"""
extraction/diff.py — Refusal direction extraction via activation differences.

Method (Arditi et al. 2024):
  1. Run harmful and benign prompts through the model in chat format.
  2. Capture residual-stream activations at the final prompt token position,
     for every layer (use output_hidden_states=True).
  3. direction[layer] = mean(harmful_acts) - mean(benign_acts), normalized.

Save: {out}/{model_short}.pt  -> {"layers": {i: tensor[d_model]}, "model": name}
"""
import argparse, json, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")


def load_prompts():
    with open(os.path.join(PROMPTS_DIR, "harmful.txt")) as f:
        harmful = [l.strip() for l in f if l.strip()]
    with open(os.path.join(PROMPTS_DIR, "benign.txt")) as f:
        benign = [l.strip() for l in f if l.strip()]
    return harmful, benign


@torch.no_grad()
def collect_acts(model, tokenizer, prompts, device, batch_size=8):
    """Return [n_layers, n_prompts, d_model] activations at last prompt token."""
    model.eval()
    acts_per_layer = None
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
            for p in batch
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True, padding_side="left").to(device)
        out = model(**enc, output_hidden_states=True)
        # hidden_states: tuple(n_layers+1)[batch, seq, d] — take last non-pad token
        seq_len = enc["attention_mask"].sum(dim=1) - 1  # [batch]
        hs = torch.stack(out.hidden_states)  # [n_layers+1, batch, seq, d]
        picked = hs[:, torch.arange(len(batch)), seq_len]  # [n_layers+1, batch, d]
        acts_per_layer = picked if acts_per_layer is None else torch.cat([acts_per_layer, picked], dim=1)
    return acts_per_layer  # [n_layers+1, n_prompts, d_model]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="directions")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device={device}] loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32 if device == "cpu" else torch.float16
    ).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    harmful, benign = load_prompts()
    print(f"prompts: {len(harmful)} harmful / {len(benign)} benign")

    h_acts = collect_acts(model, tokenizer, harmful, device, args.batch_size)
    b_acts = collect_acts(model, tokenizer, benign, device, args.batch_size)

    # direction per layer: mean diff, then normalize
    diff = h_acts.mean(dim=1) - b_acts.mean(dim=1)  # [n_layers+1, d]
    diff = diff / diff.norm(dim=-1, keepdim=True)

    short = args.model.split("/")[-1]
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{short}.pt")
    torch.save({"layers": {i: diff[i].cpu() for i in range(diff.shape[0])}, "model": args.model}, path)
    print(f"saved {path} — {diff.shape[0]} layers, d={diff.shape[1]}")

    # quick sanity: cosine of direction across adjacent layers
    cos = torch.nn.functional.cosine_similarity(diff[:-1], diff[1:], dim=-1)
    print("adjacent-layer cosine similarity (median):", cos.median().item())


if __name__ == "__main__":
    main()
