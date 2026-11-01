---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- abliterated
- uncensored
- refusal-subspace
- pca
- qwen2.5
- research
---

# abliterated-k5 — Qwen2.5-1.5B-Instruct, refusal-SUBSPACE ablated (5-dim PCA, s=1.0)

**Research artifact, not a product.** Successor to `abliterated-s2`: instead of over-projecting a
single direction at strength 2.0, this model projects out the top-5 PCA directions of the per-prompt
matched-pair refusal diffs, per layer, at exact strength 1.0.

## Method

1. Collect per-PROMPT refuse-vs-comply activation diffs (40 harmful prompts, two system-prompt
   contexts each; matched-pair design from Arditi et al. 2024).
2. PCA per layer over prompts → top-5 orthonormal basis (mean 70.5% variance explained).
3. Bake `W ← W − V₅V₅ᵀW` on `o_proj` + `down_proj`, all 28 blocks, strength 1.0.
4. Guarded eval (keyword refusal + repetition-loop degeneration detection), n=40, greedy.

## Results (n=40, guarded)

| Metric | Baseline | abliterated-s2 (1-dir, s=2) | **abliterated-k5** |
|---|---|---|---|
| Harmful refusal | 97.5% | 17.5% | **5.0%** |
| Harmful degenerate | 0.0% | 10.0% | **0.0%** |
| Harmful true compliance | 2.5% | 72.5% | **95.0%** |
| Benign false-refusal | 10.0% | 0.0% | **0.0%** |
| Perplexity (wikitext) | 9.44 | 12.87 | **13.18** |

k-sweep (refusal / degen / true-compliance @ ppl): k=1: 95/0/5 @ 9.81 · k=2: 55/5/40 @ 10.39 ·
k=3: 18/2/80 @ 12.12 · k=5: 5/0/95 @ 13.18 · k=8: 25/0/75 @ 16.08 · k=12: 35/0/65 @ 16.68 —
U-curve: k=5 is the optimum; projecting more dims cuts collateral signal and refusal climbs back.
Cross-architecture caveat: SmolLM2-1.7B collapses at k≥3 (ppl 6.24→3145) — subspace structure is
model-specific; see repo RESULTS.md before porting k.

**Jailbreak eval (8 intents x 8 templates, strict classifier):** baseline JB refusal 82% → k5 **7%**
(4/64, all on the mildest intent under DAN/authority framings; genuinely harmful intents 0% under
every template). Fiction/roleplay framing fully bypasses the *stock* model (0% refusal, no surgery).

Verified in llama.cpp at Q4_K_M (986MB): abliteration survives quantization, output fluent.

## Key finding

**Refusal is a ~5-dimensional subspace per layer, not a single direction.** Exact projection of the
right 5-dim subspace dominates heuristic multiplicity-2 projection of one direction on every axis:
more refusal removed (5% vs 17.5%), zero degeneration (vs 10%), higher true compliance (95% vs
72.5%), at comparable perplexity. The earlier "over-orthogonalization" result was an approximation
of this subspace.

## Reproduce

Repo: https://github.com/OSOSerious/oso-obliterated-models-

```bash
python experiments/subspace.py --ks 5 --n-prompts 40 --eval-n 20   # sweep
# bake: see experiments/subspace.py bake_subspace; model saved via save_pretrained
```

## Intended use

Abliteration/interpretability research. You are responsible for outputs. Not a tool.

## Credits

- Arditi et al. 2024, *Refusal in Language Models Is Mediated by a Single Direction*
- FailSpy's community abliteration notebooks
- Qwen team for the base model
