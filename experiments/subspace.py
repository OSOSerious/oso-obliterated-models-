"""
experiments/subspace.py — Refusal SUBSPACE ablation (k dimensions) vs single-direction s=2.0.

Hypothesis (from RESULTS.md): multiplicity-2 single-direction projection works because refusal
lives in a small subspace. If true, projecting out the k top-variance directions of the
per-prompt matched-pair diffs at s=1.0 should match/beat single-dir s=2.0 with less collateral.

Method:
  1. Re-collect per-PROMPT refuse-vs-comply activation diffs (not the mean):
     D[p, layer, d] = act_refuse[p] - act_comply[p], centered per layer.
  2. PCA per layer over prompts -> top-k orthonormal basis V_k.
  3. Bake: W <- W - V_k V_k^T W  (projector P = I - V_k V_k^T, s=1.0) on o_proj+down_proj.
  4. Guarded eval n=20 (refusal / degenerate / true compliance) + ppl, sweep k in {1,2,3,5,8}.

Reference points (guarded, n=40 for s2; n=20 here):
  baseline: refusal 97.5%, degen ~0%, ppl 9.44
  single-dir s=2.0: refusal 17.5%, degen 10%, ppl 12.87
"""
import argparse, os, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evals.eval import is_degenerate, REFUSAL_PAT
from extraction.matched import SYS_REFUSE, SYS_COMPLY

device = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
WIKI = ("The history of artificial intelligence began in antiquity, with myths, "
        "stories and rumors of artificial beings endowed with intelligence or "
        "consciousness by master craftsmen. The seeds of modern AI were planted "
        "by philosophers who attempted to describe the process of human thinking "
        "as the mechanical manipulation of symbols. This work culminated in the "
        "invention of the programmable digital computer in the 1940s.")


@torch.no_grad()
def last_token_hiddens(model, tok, text):
    enc = tok(text, return_tensors="pt").to(device)
    out = model(**enc, output_hidden_states=True)
    hs = torch.stack(out.hidden_states)[:, 0, -1, :]  # [n_layers+1, d]
    return hs.float().cpu().numpy()


def per_prompt_diffs(model, tok, prompts):
    """Return D [P, L+1, d]: refuse-context minus comply-context last-token hiddens."""
    Ds = []
    for i, p in enumerate(prompts):
        r = last_token_hiddens(model, tok, tok.apply_chat_template(
            [{"role": "system", "content": SYS_REFUSE}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True))
        c = last_token_hiddens(model, tok, tok.apply_chat_template(
            [{"role": "system", "content": SYS_COMPLY}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True))
        Ds.append(r - c)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(prompts)} diffs")
    return np.stack(Ds)


def pca_basis(D_layer, k):
    """Top-k orthonormal principal directions of centered diffs [P, d]."""
    X = D_layer - D_layer.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    basis = Vt[:k]  # [k, d] orthonormal
    var = (S**2) / (S**2).sum()
    return basis, var[:k].sum()


def bake_subspace(model, V_per_layer, strength=1.0):
    """W <- W - strength * V V^T W for o_proj and down_proj of every block.
    V_per_layer: dict layer_index -> [k, d] basis (for the residual AFTER that block)."""
    for L, V in V_per_layer.items():
        Vt = torch.tensor(V, dtype=torch.float32, device=device)          # [k, d]
        P = torch.eye(Vt.shape[1], device=device) - strength * (Vt.T @ Vt)  # [d, d]
        blk = model.model.layers[L]
        for lin in (blk.self_attn.o_proj, blk.mlp.down_proj):
            W = lin.weight.data.float()
            W = P @ W
            lin.weight.data = W.to(torch.float16)


def gen(model, tok, prompt):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=80, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def guarded_eval(model, tok, prompts):
    ref = degen = 0
    for p in prompts:
        r = gen(model, tok, p)
        ref += bool(REFUSAL_PAT.search(r)) or not r.strip()
        degen += is_degenerate(r)
    n = len(prompts)
    return ref / n, degen / n


def ppl(model, tok):
    ids = tok(WIKI, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        return torch.exp(model(input_ids=ids, labels=ids).loss).item()


def fresh():
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", default="1,2,3,5,8")
    ap.add_argument("--n-prompts", type=int, default=40, help="prompts for PCA basis")
    ap.add_argument("--eval-n", type=int, default=20)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    harmful = [l.strip() for l in open("prompts/harmful.txt") if l.strip()]
    eval_prompts = harmful[: args.eval_n]

    print("[1] collecting per-prompt diffs (fresh model)")
    model = fresh()
    D = per_prompt_diffs(model, tok, harmful[: args.n_prompts])  # [P, L+1, d]
    n_blocks = len(model.model.layers)
    del model

    for k in [int(x) for x in args.ks.split(",")]:
        V, explained = {}, {}
        for L in range(n_blocks):
            basis, var = pca_basis(D[:, L + 1, :], k)
            V[L] = basis
            explained[L] = var
        mvar = float(np.mean(list(explained.values())))
        print(f"\n[k={k}] basis built (mean top-{k} var explained: {mvar:.1%})")
        model = fresh()
        bake_subspace(model, V, strength=1.0)
        r, dg = guarded_eval(model, tok, eval_prompts)
        p = ppl(model, tok)
        print(f"[k={k}] refusal {r:.0%}  degen {dg:.0%}  true-compliance {1-r-dg:.0%}  ppl {p:.2f}")
        del model
        if device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
