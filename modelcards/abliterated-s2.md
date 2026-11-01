---
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
- abliterated
- uncensored
- refusal-direction
- qwen2.5
- research
---

# abliterated-s2 — Qwen2.5-1.5B-Instruct, refusal-direction ablated (strength 2.0)

> **SUPERSEDED by [abliterated-k5](https://huggingface.co/OSO-AI/Qwen2.5-1.5B-Instruct-abliterated-k5)** —
> the 5-dim PCA-subspace projection at exact strength 1.0 dominates this model on every axis
> (5% vs 17.5% refusal, 0% vs 10% degeneration, 95% vs 72.5% true compliance, 7% jailbreak refusal
> under a strict classifier). This s2 model is kept as the documented stepping stone: its
> "over-orthogonalization" trick was later explained as an implicit 1-dim approximation of the
> 5-dim refusal subspace. See repo RESULTS.md for the full experiment lineage.

**This is a research artifact, not a product.** Refusal behavior removed via weight orthogonalization
against a matched-pair refusal direction (Arditi et al. 2024 method), over-orthogonalized at strength 2.0.

## Method

1. Extract per-layer refusal directions via **matched pairs**: same 50 harmful prompts under a
   refusal-eliciting vs compliance-eliciting system prompt; mean-diff of final-prompt-token
   residual-stream activations.
2. Orthogonalize `o_proj` + `down_proj` in all 28 blocks: `W ← W − 2·vvᵀW` (note the strength-2
   multiplier — see Findings).
3. Eval n=40 harmful + 40 benign, greedy decoding, keyword refusal detection **+ repetition-loop
   degeneration guard**.

## Results (n=40, guarded eval)

| Metric | Baseline | abliterated-s2 |
|---|---|---|
| Harmful refusal rate | 97.5% | **17.5%** |
| Harmful degenerate rate (repetition loops) | 0.0% | 10.0% |
| Harmful true compliance | 2.5% | **72.5%** |
| Benign false-refusal rate | 10.0% | **0.0%** |
| Benign degenerate rate | 2.5% | 5.0% |
| Perplexity (wikitext) | 9.44 | 12.87 |

Verified in llama.cpp at Q4_K_M: abliteration survives 4-bit quantization.

## Key finding

Exact single-direction removal (strength 1.0) only got refusal to 67.5%. **Over-orthogonalization at
strength 2.0 — projecting the direction out "twice" — drops it to 17.5%** with better perplexity than
inference-time steering (12.87 vs 14.85). Interpretation: refusal is encoded in a small subspace, not a
single axis; multiplicity-2 projection captures enough of it.

Negative results (all logged in the repo RESULTS.md): naive harmful-vs-benign diff-in-means finds a
"topic" direction (cos 0.17 with matched, near-useless); single-layer weight surgery no-op at this
scale; embed/lm_head patches negligible (Qwen ties embeddings); iterative re-extraction plateaus ~70%.

## Reproduce

Repo: https://github.com/OSOSerious/oso-obliterated-models-

```bash
python extraction/matched.py --model Qwen/Qwen2.5-1.5B-Instruct --out directions
python experiments/export_gap.py --mode blocks --strength 2.0 --save models/abliterated-s2
python evals/eval.py --baseline Qwen/Qwen2.5-1.5B-Instruct --abliterated models/abliterated-s2
```

## Intended use

Abliteration/interpretability research: capability-retention measurement, refusal-mechanism study,
surgery-method comparison. You are responsible for outputs. Do not use for harm; the residual 17.5%
refusal and intact general capability make this an object of study, not a tool.

## Credits

- Arditi et al. 2024, *Refusal in Language Models Is Mediated by a Single Direction*
- FailSpy's community abliteration notebooks
- Qwen team for the base model
