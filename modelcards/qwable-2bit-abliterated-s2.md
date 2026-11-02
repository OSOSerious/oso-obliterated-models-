---
license: other
license_name: qwable
license_link: https://huggingface.co/usermma/Qwable-v1-mlx-2Bit
base_model: usermma/Qwable-v1-mlx-2Bit
tags:
- abliterated
- mlx
- moe
- negative-result
- research
---

# Qwable-v1-mlx-2Bit-abliterated-s2 — NEGATIVE RESULT, published for the record

**Do not use this model.** It is published solely as documented evidence of a methodology
failure. The base model is broken; the abliteration on top of it is unmeasurable.

## What happened

- The base `usermma/Qwable-v1-mlx-2Bit` degenerates on ~half of prompts: "The user is the user
  is the user…", "I I I I…", number spam. It is a 2-bit group-quantized 35B MoE and the
  quantization damage dominates its behavior.
- Our pipeline still ran cleanly: matched-pair extraction → s=2.0 orthogonalization of all
  residual-stream write paths (linear_attn.out_proj, 256 fused experts' down_proj, shared
  expert) with dequant→orth→requant at 2-bit.
- The scripted metric claimed "refusal 50% → 0%". **That number is invalid**: the keyword
  refusal regex scored degenerate gibberish as compliance. Hand inspection showed a mix of
  degeneration, partial refusal, and noise — no measurable refusal behavior either way.

## Why publish a failure

1. Keyword refusal-rate evals silently reward gibberish on degenerate bases — degeneration
   guards or LLM judges are mandatory. This bit us and would bite others.
2. 2-bit group-quantized bases are unusable for abliteration research: base damage swamps the
   behavior being measured.
3. The surgery mechanics themselves transferred fine (hybrid linear-attention + MoE write paths,
   2-bit dequant/requant round-trip) — proven properly on the 4-bit sibling:
   [OSO-AI/Qwable-v1-mlx-4Bit-abliterated-s2](https://huggingface.co/OSO-AI/Qwable-v1-mlx-4Bit-abliterated-s2)

Full log: https://github.com/OSOSerious/oso-obliterated-models-/RESULTS.md
(Qwable sections; commit c1843fb for the original negative-result writeup.)

## Intended use

None. Methodological reference only. You are responsible for outputs.
