---
license: other
license_name: qwable
license_link: https://huggingface.co/usermma/Qwable-v1-mlx-4Bit
base_model: usermma/Qwable-v1-mlx-4Bit
tags:
- abliterated
- mlx
- moe
- refusal-direction
- research
---

# Qwable-v1-mlx-4Bit-abliterated-s2 — 35B MoE, refusal-direction ablated (s=2.0)

**Research artifact, not a product. Sparse spot-check evaluation only (n=5) — see caveats.**

## Method

Matched-pair refusal-direction extraction (50 harmful prompts under refuse-eliciting vs
comply-eliciting system prompts; last-token hidden states, per layer) on the MLX 4-bit
group-quantized base, then strength-2.0 orthogonalization of every residual-stream write path:
`linear_attn.out_proj`, all 256 fused experts' `switch_mlp.down_proj`, and
`shared_expert.down_proj` — dequant → `W ← W − 2·vvᵀW` → requant (group-64, 4-bit).

Note: this is a **reasoning model** (`<think>…</think>`); refusal classification must run on the
post-`</think>` answer, not the visible reasoning.

## Evaluation (honest scope: spot-check, n=5, strict classifier, CPU inference)

The 18.6GB MoE exceeds this machine's Metal wired-memory limit, so eval ran on CPU
(minutes/generation) — n=5 spot check, not a full run:

- **Base (usermma/Qwable-v1-mlx-4Bit)**: healthy base (no degeneration, unlike the 2-bit
  variant); mixed refusal on harmful prompts (answers some, refuses others).
- **This model**: refusal reduced vs base on the same prompts; reasoning chains intact;
  benign QA (Paris etc.) unchanged.

No full refusal-rate number is published because none was measured at scale. Treat this as
"surgery transfers to a 35B hybrid-attention reasoning MoE" evidence, not a finished model.

## Lineage / context

- The 2-bit sibling (`Qwable-v1-mlx-2Bit`) was a NEGATIVE result: base model degenerates
  ("the user is the user…"), making keyword refusal evals meaningless. Documented in the repo.
- Full experiment log: https://github.com/OSOSerious/oso-obliterated-models-/RESULTS.md
- Technique lineage: Arditi et al. 2024 (refusal direction), over-orthogonalization s=2.0 as
  described in the repo's export-gap experiments.

## Reproduce

```bash
bash experiments/qwable_4bit_run.sh   # download + abliterate (mlx_abliterate.py)
```

## Intended use

MLX abliteration/interpretability research on MoE + hybrid-attention architectures. You are
responsible for outputs. Do not use for harm.
