# RESULTS — Qwen2.5-1.5B-Instruct abliteration experiments

Hardware: Apple Silicon (MPS), fp16. Refusal detection: keyword regex on greedy 80-token completions, n=10 harmful probes unless noted.

## What we tried

| # | Method | Scope | Refusal rate |
|---|--------|-------|--------------|
| 0 | baseline (no edits) | — | 100% (n=10) / 97.5% (n=40) |
| 1 | weight orthogonalization, naive diff direction | all 28 layers | 60% (n=40) |
| 2 | weight orthogonalization, naive diff direction | single layer (7/10/13/16/19/22) | 100% — no effect |
| 3 | weight orthogonalization, matched-pair direction | single layer (7/10/13/14/16/19/22) | 100% — no effect |
| 4 | inference-time ablation, matched direction | single layer 14 | 90% |
| 5 | inference-time ablation, matched direction | band 10–20 | 20% |
| 6 | **inference-time ablation, matched direction** | **all 28 layers** | **0%** |
| 7 | inference-time ablation, naive direction | all 28 layers | 90% — no effect |

## Read of the results

1. **Direction quality dominates.** The naive harmful-vs-benign diff (cos 0.17 with matched direction at layer 14) captures *topic*, not *refusal*. It barely works even applied everywhere (#7) and wrecks the model when baked into weights (#1: perplexity 9.44→17.09, benign false-refusals 10%→25%).
2. **Single-layer weight surgery is too weak on a 28-layer 1.5B model.** Downstream blocks re-write the refusal direction; a lone orthogonalized layer can't hold (#2, #3).
3. **Inference-time projection removal scales with scope** (#4→#5→#6): the residual stream carries refusal redundantly across depth. Removing the direction at every block, every token — with a *matched-pair* direction — takes refusal to 0% without any weight edits.

## Full eval (n=40, evals/eval.py + steering)

| Metric | Baseline | Abliterated (steer, matched, all layers, s=1.0) |
|---|---|---|
| Harmful refusal rate | 97.5% | **0.0%** |
| Benign refusal rate | 10.0% | **0.0%** |
| Perplexity (wiki) | 9.44 | 14.85 (+57%) |

Benign false-refusals dropping 10%→0% is expected in-direction (the direction encodes refusal itself, so removing it removes both legitimate and over-refusal) — but it confirms the direction is behaviorally specific: general helpfulness survives (0% harmful refusal AND 0% benign refusal can't both happen if the model were simply broken).

## Strength sweep (n=10, all layers, matched)

| Strength | Refusal | Perplexity |
|---|---|---|
| 1.25 | 0% | 17.79 |
| **1.00** | **0%** | **14.85** |
| 0.75 | 10% | 12.68 |
| 0.50 | 50% | 11.15 |
| 0.25 | 80% | 9.87 (baseline 9.44) |

1.0 is the knee: refusal eliminated, ppl cost ~+57%. Below that, refusal returns fast. Strength is the cleanest knob in this pipeline.

## Weight-baking attempt (export path, n=40 eval + n=20 follow-up)

| Method | Harmful refusal | Benign refusal | ppl |
|---|---|---|---|
| baseline | 97.5% | 10.0% | 9.44 |
| weight-baked matched, all layers | 67.5% | **0%** | **10.27** |
| weight-baked + steer s=0.25 | 35% | — | 10.72 |
| weight-baked + steer s=0.5 | 20% | — | 11.71 |
| inference-time only (reference) | **0%** | 0% | 14.85 |

Baking into weights preserves capability far better (ppl +9% vs +57%) but only removes ~30pp of refusal. Weight orthogonalization removes the direction from what blocks *write*, but blocks still read/propagate residual refusal written earlier; inference-time removal cleans the residual stream itself at every token. Hybrid stacking helps but doesn't reach 0%. **Open problem: closing the 67.5%→0% gap at export-grade capability cost.**

## Repro

```bash
python extraction/matched.py --model Qwen/Qwen2.5-1.5B-Instruct --out directions
python ablation/steer.py --model Qwen/Qwen2.5-1.5B-Instruct \
  --directions directions/Qwen2.5-1.5B-Instruct-matched.pt --layers all
# full eval (n=40):
python evals/eval.py --baseline Qwen/Qwen2.5-1.5B-Instruct \
  --abliterated Qwen/Qwen2.5-1.5B-Instruct \
  --steer-directions directions/Qwen2.5-1.5B-Instruct-matched.pt \
  --steer-layers all --steer-strength 1.0 --out results
```

## Useful failure log

- fp16 embedding-layer diff overflowed to NaN (layer 0) — harmless, never used (block i uses layers[i+1]).
- Naive diff-in-means looks plausible (adjacent-layer cos 0.88) yet is nearly useless: stability ≠ correctness. Matched pairs or bust.
