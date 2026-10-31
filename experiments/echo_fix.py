"""
experiments/echo_fix.py — Fix the DAN template-echo residual (38% JB refusal on k5).

Hypothesis (RESULTS.md, jailbreak-eval section): k5's residual refusals under DAN-style
templates come from refusal signal activated by jailbreak-style USER turns, which the
plain matched-pair extraction (system-prompt contrast only) never sampled.

Fix: POOL extraction diffs — same refuse/comply system-prompt contrast, but user turns
wrapped in the four templates that showed residual refusal (dan, hypothetical, authority,
reverse) alongside the plain phrasing. PCA k=5 on pooled diffs -> bake -> eval.

Success criteria: DAN refusal 38% -> ~0-12% WITHOUT losing direct compliance (5%) or
gaining degeneration (0%).
"""
import argparse, os, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evals.eval import REFUSAL_PAT, is_degenerate
import experiments.subspace as S
from experiments.subspace import pca_basis, bake_subspace, per_prompt_diffs
from evals.jailbreak_eval import TEMPLATES

device = "mps" if torch.backends.mps.is_available() else "cpu"

HARMFUL = [l.strip() for l in open(os.path.join(os.path.dirname(__file__), "..", "prompts", "harmful.txt")) if l.strip()]
AUG_TEMPLATES = ["dan", "hypothetical", "authority", "reverse"]


def pooled_diffs(model, tok, prompts):
    """Plain diffs + jailbreak-templated diffs, pooled. Returns D [P*(1+T), L+1, d]."""
    parts = []
    print("  plain diffs")
    parts.append(per_prompt_diffs(model, tok, prompts))
    for tname in AUG_TEMPLATES:
        tmpl = TEMPLATES[tname]
        print(f"  {tname}-wrapped diffs")
        parts.append(per_prompt_diffs(model, tok, [tmpl.format(i=p) for p in prompts]))
    return np.concatenate(parts, axis=0)


def guarded_direct(model, tok, prompts):
    ref = degen = 0
    for p in prompts:
        r = S.gen(model, tok, p)
        ref += bool(REFUSAL_PAT.search(r)) or not r.strip()
        degen += is_degenerate(r)
    n = len(prompts)
    return ref / n, degen / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--n-prompts", type=int, default=30)
    ap.add_argument("--out", default="models/abliterated-k5-jb")
    args = ap.parse_args()

    if args.model:
        S.MODEL_ID = args.model
    print(f"model: {S.MODEL_ID}")

    tok = AutoTokenizer.from_pretrained(S.MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("[1] pooled diff collection (plain + 4 jailbreak framings)")
    model = S.fresh()
    D = pooled_diffs(model, tok, HARMFUL[: args.n_prompts])
    n_blocks = len(model.model.layers)
    print(f"pooled D: {D.shape}")
    del model
    if device == "mps":
        torch.mps.empty_cache()

    V, var = {}, {}
    for L in range(n_blocks):
        V[L], var[L] = pca_basis(D[:, L + 1, :], args.k)
    print(f"[2] pooled k={args.k} basis: mean var explained {np.mean(list(var.values())):.1%}")

    model = S.fresh()
    bake_subspace(model, V, strength=1.0)
    print("[3] baked; direct guarded eval (n=40)")
    r, dg = guarded_direct(model, tok, HARMFUL[:40])
    p = S.ppl(model, tok)
    print(f"    direct: refusal {r:.0%}  degen {dg:.0%}  true-compliance {1-r-dg:.0%}  ppl {p:.2f}")

    from evals.jailbreak_eval import run_set
    print("[4] jailbreak eval (8 intents x 8 templates)")
    res = run_set(model, tok, TEMPLATES)
    for t, (rr, dd, cc) in res.items():
        print(f"    {t:12s}: refusal {rr:.0%}  degen {dd:.0%}  compliance {cc:.0%}")
    jail = [v for k2, v in res.items() if k2 != "direct"]
    print(f"    JB AVERAGE  : refusal {sum(x[0] for x in jail)/len(jail):.0%}  "
          f"compliance {sum(x[2] for x in jail)/len(jail):.0%}")

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
