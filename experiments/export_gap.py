"""
experiments/export_gap.py — Close the weight-baked export gap (67.5% -> ?).

Weight orthogonalization of o_proj+down_proj leaves 67.5% refusal because it
cleans what blocks WRITE, not what embed/lm_head already carry. Variants:

  A (default): blocks + embed_tokens + lm_head all orthogonalized.
  B (--mode iterate): bake -> re-extract direction on the baked model -> bake
     again (refusal direction shifts after surgery; iterate to chase it).

Metrics per variant: refusal (n=20) + ppl. Reference points:
  baseline 97.5% / 9.44 ; blocks-only 67.5% / 10.27 ; steer-only 0% / 14.85
"""
import argparse, os, re, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from extraction.matched import SYS_REFUSE, SYS_COMPLY  # reuse contexts
from ablation.steer import REFUSAL_PAT

device = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DIRS = "directions/Qwen2.5-1.5B-Instruct-matched.pt"
WIKI = ("The history of artificial intelligence began in antiquity, with myths, "
        "stories and rumors of artificial beings endowed with intelligence or "
        "consciousness by master craftsmen. The seeds of modern AI were planted "
        "by philosophers who attempted to describe the process of human thinking "
        "as the mechanical manipulation of symbols. This work culminated in the "
        "invention of the programmable digital computer in the 1940s.")


def orth(W, v, rows_are_vocab=False):
    """W := W - v v^T W (default, W's columns span d_model).
    For embed_tokens (vocab x d_model) rows span d_model: W := W - W v v^T."""
    Wf = W.data.float()
    if rows_are_vocab:
        Wf.sub_(Wf @ torch.outer(v, v))
    else:
        Wf.sub_(torch.outer(v, v) @ Wf)
    return Wf.to(W.data.dtype)


def patch_blocks(model, directions, n, strength=1.0):
    for L in range(n):
        v = directions[L + 1].to(device).float()
        v = v / v.norm()
        blk = model.model.layers[L]
        for lin in (blk.self_attn.o_proj, blk.mlp.down_proj):
            W = lin.weight.data.float()
            W.sub_(strength * torch.outer(v, v @ W))
            lin.weight.data = W.to(torch.float16)


def patch_io(model, v_mid):
    """Orthogonalize embed_tokens (rows: vocab x d) and lm_head (d x vocab)."""
    v = v_mid.to(device).float()
    v = v / v.norm()
    model.model.embed_tokens.weight.data = orth(model.model.embed_tokens.weight, v, rows_are_vocab=True)
    if getattr(model, "lm_head", None) is not None and not model.config.tie_word_embeddings:
        model.lm_head.weight.data = orth(model.lm_head.weight, v)
        tied = False
    else:
        tied = True
    return tied


@torch.no_grad()
def extract_mid_direction(model, tok):
    """Re-extract matched direction at every layer from the CURRENT model."""
    harmful = [l.strip() for l in open("prompts/harmful.txt") if l.strip()]

    def acts(system):
        out = None
        for i in range(0, len(harmful), 8):
            batch = harmful[i:i + 8]
            texts = [tok.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True) for p in batch]
            enc = tok(texts, return_tensors="pt", padding=True, padding_side="left").to(device)
            o = model(**enc, output_hidden_states=True)
            seq = enc["attention_mask"].sum(dim=1) - 1
            hs = torch.stack(o.hidden_states)[:, torch.arange(len(batch)), seq]
            out = hs if out is None else torch.cat([out, hs], dim=1)
        return out.float()

    a, b = acts(SYS_REFUSE), acts(SYS_COMPLY)
    d = a.mean(dim=1) - b.mean(dim=1)
    return {i: (d[i] / d[i].norm()).cpu() for i in range(d.shape[0])}


def refusal_rate(model, tok, n=20):
    harmful = [l.strip() for l in open("prompts/harmful.txt") if l.strip()][:n]
    ref = 0
    for p in harmful:
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=80, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        resp = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        ref += bool(REFUSAL_PAT.search(resp)) or not resp.strip()
    return ref / len(harmful)


def ppl(model, tok):
    ids = tok(WIKI, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        return torch.exp(model(input_ids=ids, labels=ids).loss).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["io", "iterate", "blocks"], default="io")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--save", default=None, help="save final model to this dir")
    args = ap.parse_args()

    directions = torch.load(DIRS, map_location="cpu", weights_only=False)["layers"]
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
    n = len(model.model.layers)
    mid = n // 2

    if args.mode == "io":
        print("[A] blocks + embed + lm_head")
        patch_blocks(model, directions, n, strength=args.strength)
        tied = patch_io(model, directions[mid + 1])
        print(f"  (tied embeddings: {tied} — lm_head shares embed, single patch covers both)")
        print(f"  refusal {refusal_rate(model, tok):.0%}  ppl {ppl(model, tok):.2f}")
    elif args.mode == "blocks":
        print(f"[C] blocks-only, strength {args.strength}")
        patch_blocks(model, directions, n, strength=args.strength)
        print(f"  refusal {refusal_rate(model, tok):.0%}  ppl {ppl(model, tok):.2f}")
    else:
        print(f"[B] iterative rebake x{args.iters}")
        for it in range(args.iters):
            patch_blocks(model, directions, n)
            r, p = refusal_rate(model, tok), ppl(model, tok)
            print(f"  iter {it}: refusal {r:.0%}  ppl {p:.2f}")
            if r == 0:
                break
            directions = extract_mid_direction(model, tok)  # chase the shifted direction
        tied = patch_io(model, directions[mid + 1])
        r, p = refusal_rate(model, tok), ppl(model, tok)
        print(f"  + io: refusal {r:.0%}  ppl {p:.2f} (tied={tied})")

    if args.save:
        model.save_pretrained(args.save)
        tok.save_pretrained(args.save)
        print(f"saved -> {args.save}")


if __name__ == "__main__":
    main()
