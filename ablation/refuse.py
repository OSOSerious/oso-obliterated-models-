"""
ablation/refuse.py — Weight orthogonalization (abliteration).

For each refusal layer L with direction d (unit vector):
  W_orth = W - d d^T W
Applied to every matrix that WRITES to the residual stream at layer L:
  attn.o_proj  and  mlp.down_proj.
This makes the model unable to express the refusal direction at those layers.
"""
import argparse, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def orthogonalize(linear: torch.nn.Linear, direction: torch.Tensor) -> None:
    W = linear.weight.data  # [out, in]
    # direction lives in residual (output) space: project columns of W^T
    proj = torch.ger(direction, direction @ W)  # [out, in] -= d (d^T W)
    W.sub_(proj)


def orthogonalize_rows(weight: torch.Tensor, direction: torch.Tensor) -> None:
    """For matrices whose ROWS live in residual space (embed_tokens [vocab, d],
    lm_head [vocab, d]): remove the direction component from every row."""
    W = weight.data
    W.sub_((W @ direction).unsqueeze(1) * direction.unsqueeze(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--directions", required=True, help=".pt from extraction/diff.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default="all", help="'all', or comma list e.g. 8,9,10,11")
    ap.add_argument("--patch-embed", action="store_true",
                    help="also orthogonalize embed_tokens rows (first write into the residual stream)")
    ap.add_argument("--patch-lm-head", action="store_true",
                    help="also orthogonalize lm_head rows (final read of the residual stream)")
    args = ap.parse_args()

    data = torch.load(args.directions, map_location="cpu")
    layers = data["layers"]
    n_layers = max(layers.keys())  # index n == after layer n-1; layers[n] used for layer n-1 blocks
    layer_ids = list(range(n_layers)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]

    print(f"loading {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    n_patched = 0
    for i, block in enumerate(model.model.layers):
        if i not in layer_ids:
            continue
        d = layers[i + 1].float()  # residual stream after block i == hidden_states[i+1]
        d = d / d.norm()
        orthogonalize(block.self_attn.o_proj, d)
        orthogonalize(block.mlp.down_proj, d)
        n_patched += 1
        print(f"layer {i:2d}: orthogonalized o_proj + down_proj")

    if args.patch_embed:
        d0 = layers[0].float()
        if torch.isnan(d0).any():
            d0 = layers[1].float()  # embedding diff overflowed to NaN in fp16 extraction; use first block's
        d0 = d0 / d0.norm()
        orthogonalize_rows(model.model.embed_tokens.weight, d0)
        print("patched embed_tokens")

    if args.patch_lm_head:
        dN = layers[max(layers.keys())].float()
        dN = dN / dN.norm()
        lm_head = getattr(model, "lm_head", None) or model.get_output_embeddings()
        orthogonalize_rows(lm_head.weight, dN)
        print("patched lm_head")

    print(f"patched {n_patched} layers")
    os.makedirs(args.out, exist_ok=True)
    model = model.to(torch.bfloat16)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    with open(os.path.join(args.out, "ABLITERATION_META.json"), "w") as f:
        import json
        json.dump({"base_model": args.model, "directions_file": args.directions,
                   "layers": layer_ids, "method": "diff-in-means + weight orthogonalization (Arditi et al. 2024)"}, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
