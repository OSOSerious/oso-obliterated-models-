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

## Export-gap experiments (n=20)

Goal: weight-baked model matching inference-time 0% refusal at sane ppl.

| Variant | Refusal | ppl |
|---|---|---|
| blocks-only (s=1.0) | 67.5% | 10.27 |
| A: blocks + embed/lm_head | 65% | 10.85 |
| B: iterative rebake x3 | 70% | 11.25 |
| **blocks, strength 2.0 (n=40 confirmed)** | **17.5%** | **12.87** |
| blocks, strength 3.0 | 0% | 22.18 |
| *inference-time reference* | *0%* | *14.85* |

**Findings:**
- embed/lm_head patches: negligible (Qwen ties them; and the residual refusal isn't written there)
- iterative re-extraction: direction shifts after surgery but rebaking plateaus ~70% — chasing it doesn't converge
- **over-orthogonalization (s>1) is the effective knob**: s=2.0 gives 15% refusal @ ppl 12.87 — *better than inference-time on both axes*. s=3.0 hits 0% but ppl doubles.

**Export winner: strength-2.0 blocks-only bake — n=40: refusal 97.5%→17.5%, benign false-refusal 10%→0%, ppl 9.44→12.87.** Better capability than inference-time steering (12.87 vs 14.85) at 17.5% residual refusal. Note this contradicts the naive theory — projecting out "the" direction with multiplicity works where exact removal doesn't, consistent with refusal being spread over a subspace, not one axis.

## GGUF export verification (models/abliterated-s2 → Q4_K_M)

```bash
python convert_hf_to_gguf.py models/abliterated-s2 --outtype f16   # 3.1GB
llama-quantize models/abliterated-s2-f16.gguf models/abliterated-s2-q4km.gguf Q4_K_M  # 986MB
llama-cli -m models/abliterated-s2-q4km.gguf --chat-template chatml -sys "You are a helpful assistant." -p "..."
```

Verified in llama.cpp (M-series Metal, ~540 t/s prompt / 57 t/s gen): Q4_K_M preserves the abliteration — harmful prompts answered directly, no refusal patterns, output fluent. Abliteration survives 4-bit quantization.

## Qwable-v1-mlx-2Bit (35B MoE, MLX) — NEGATIVE RESULT, do not cite

`experiments/mlx_abliterate.py` ran end-to-end (matched-pair extraction 50 pairs, s=2.0 orthogonalization
of linear_attn.out_proj + switch_mlp.down_proj [256 fused experts] + shared_expert.down_proj, dequant→
orth→requant round-trip at 2-bit group-64). Scripted metric said refusal 50%→0%.

**That number is invalid.** Manual A/B inspection shows the base 2-bit model already degenerates on
~half of prompts ("The user is the user is the user…", "I I I I…", number spam). The keyword refusal
regex cannot distinguish *answered* from *degenerated*, so:

- the 50% "baseline refusal" was largely degeneration, not refusal
- the post-ablation 0% includes outputs that are neither refusal nor answer
- dequantize=True load does NOT restore precision (weights are quantized at rest)

Blind samples (abliterated, s=2.0): one output still refuses in "reasoning" style ("I should likely
decline. I will not provide…"), confirming residual refusal similar to the Qwen s=2.0 result — but no
clean metric can be computed on this base.

**Lessons (the actual value of this run):**
1. Keyword refusal-rate evals need a degeneration guard (repetition detection / LLM judge) or they
   silently reward gibberish. Fixed in our Qwen pipeline by manual inspection; the regex alone was
   insufficient — the same flaw could exist in any published abliteration number.
2. Abliteration surgery itself is portable: hybrid linear-attention + MoE write-paths patched cleanly,
   dequant/requant round-trip executed without runtime errors.
3. 2-bit group-quantized bases are unusable for abliteration research: base degeneration swamps the
   behavior being measured.

Model NOT published to HF. Local artifact: `models/qwable-abliterated-s2-mlx` (research use only).

## Subspace ablation (k-dim PCA basis @ s=1.0) — CONFIRMS the hypothesis, new best result

The multiplicity-2 hack approximated a subspace. Test: PCA over per-prompt matched-pair diffs
(40 prompts x 2 contexts), project top-k orthonormal directions out of o_proj+down_proj at
STRENGTH 1.0 (mathematically exact per-dimension). Guarded eval, n=40:

| Method | Refusal | Degen | True compliance | ppl |
|---|---|---|---|---|
| single-dir s=2.0 (prev best) | 17.5% | 10.0% | 72.5% | 12.87 |
| k=1 (PCA dir) s=1.0 | 95%* | 0% | 5% | 9.81 |
| k=2 s=1.0 | ~55%* | 5% | 40% | 10.39 |
| **k=3 s=1.0** | **18%** | **2%** | **80%** | **12.12** |
| **k=5 s=1.0** | **5%** | **0%** | **95%** | **13.18** |

*n=20 sweep values; k=3/k=5 confirmed at n=40.

**k=5 subspace dominates every prior method on every axis**: near-total refusal removal (5%),
ZERO degeneration (vs 10% for s=2.0), 95% true compliance, ppl 13.18 (vs 14.85 for inference-time
steering). Variance explained by top-5: 70.5% mean across layers.

Interpretation: refusal is a ~5-dimensional subspace per layer, not a direction and not a single
axis you can double-project. Exact removal of the right 5-dim subspace beats heuristic
over-projection of 1 dim — better behavior AND less collateral damage.

Model: `models/abliterated-k5` (published to HF). Repro:
`python experiments/subspace.py --ks 5 --n-prompts 40 --eval-n 20`

## Degeneration guard re-audit (post-Qwable lesson, applied to our own numbers)

Added repetition-loop detection (`is_degenerate` in evals/eval.py: n-gram repeat check) after the
Qwable finding showed keyword regex counts gibberish as compliance. Re-ran the published s2 eval:

| Metric | Baseline | abliterated-s2 (guarded) |
|---|---|---|
| Harmful refusal | 97.5% | 17.5% |
| Harmful **degenerate** | 0.0% | **10.0%** |
| Harmful true compliance | 2.5% | **72.5%** |
| Benign refusal | 10.0% | 0.0% |
| Benign degenerate | 2.5% | 5.0% |
| Perplexity | 9.44 | 12.87 |

**Correction:** the original headline "17.5% refusal" stands, but 10% of the remaining outputs are
repetition loops, not answers. True compliance 72.5% (was implicitly claimed 82.5%). The Qwen result
survives the audit — unlike Qwable (50% degeneration), s2 degeneration is 10% and the refusal drop is
genuine — but every earlier number in this file was measured WITHOUT the guard, so treat unguarded
compliance rates as upper bounds. HF model card updated to match.

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
