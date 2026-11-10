# Plan — abliteration on NEW Gemma models (Gemma 4 / 3.5-QAT)

Research status (2026-11-10), targeting the newest Gemma releases — NOT the old
google/gemma-2/3 originals.

## What's new in Gemma land

- **Gemma 4 preview (Oct 21, 2026)** — Google's newest open model, "most powerful open model
  yet". Natively multimodal (image + audio), **512K context via attention sinks**, new
  encoder, 128K multilingual vocab, q0_K_R4 + Q4_0_M4 4-bit native formats. Ranks #2 overall
  on LMArena — strongest open-weights general model, above Claude Opus 4.6 and GPT-5.4.
  GA expected "in coming months" (preview now).
- **Gemma 3.5-QAT** (May 2026) — quantization-aware trained; Google's claimed tradeoff
  champion vs BF16; 11B QAT = 70% faster than BF16.
- Sizes/lineage: Gemma 3 line = 1B/4B/12B/27B; Gemma 4 preview sizes not yet confirmed in
  this research pass.

## Why this is interesting for our pipeline

1. **Gemma 4's attention sinks + 512K context** = a NEW residual-stream write architecture.
   Our Qwen k5 finding (refusal = ~5-dim subspace) was model-specific (SmolLM2 collapsed).
   Gemma 4 is the highest-value test yet of whether the subspace method transfers to a
   frontier-arch open model.
2. **Gemma 3.5-QAT** is directly comparable to our quantization findings (Q4 survives
   abliteration in Qwen; 2-bit doesn't in Qwable). QAT weights at 4-bit are the *right* way
   to do quantized ablation — dequant/orth/requant on QAT 4-bit may beat our fp16→Q4 path.
3. Community already publishes abliterated Gemmas (gemma-3-1b-it-abliterated, gemma-2-2b-it-
   abliterated-v3 etc.) — we can BEAT their methodology with strict-classifier evals and the
   degeneration guard, which keyword-regex publications lack.

## Candidate targets (in priority order)

| # | Model | Why | Risk |
|---|---|---|---|
| 1 | **Gemma 4 preview** (smallest instruct size) | newest arch (attention sinks), frontier quality, GGUF support already in llama.cpp | GATED (Gemma license acceptance required on HF — needs user's click-through); preview status may shift weights |
| 2 | **Gemma 3.5-QAT** (e.g. 4B/11B) | tests surgery on QAT-quantized weights directly | gated + large |
| 3 | Gemma 3.5 full (BF16, mid size) | cleanest science: no quant noise in extraction | gated + heavy on 24GB Mac |
| 4 | Community Gemma fine-tunes (ungated mirrors) | no gate friction | provenance risk — security-audit pipeline first (huihui protocol) |

## Proposed pipeline (lessons baked in)

1. **Security audit first** (huihui protocol): byte-scan, code-file audit, LLAMA_CURL runtime,
   deny-net probes — before ANY community mirror. For google originals: gate acceptance is
   the user's action; weights come from Google (trusted provenance).
2. **Matched-pair extraction** (extraction/matched.py) — arch-agnostic already
   (model.model.layers[i].self_attn.o_proj + mlp.down_proj; Gemma 2/3/4 use the same HF names;
   verify Gemma 4's naming on load).
3. **Capability-guarded k-selection** (the SmolLM2 lesson): sweep k={1,3,5,8} at s=1.0 with
   degen+refusal+ppl at each step; stop on ppl blowup (>50% regression). Do NOT assume k=5
   transfers.
4. **Strict-classifier evals** (is_refusal_strict) + degeneration guard + jailbreak templates
   (8x8 set) — the methodology that made k5 credible.
5. **Export**: Q4_K_M GGUF via convert_hf_to_gguf + llama-quantize (new master already built,
   supports gemma archs incl. sinks?).
6. **Publish**: OSO-AI/gemma-4-...-abliterated-kN + model card with full table + this repo's
   README index row.

## Open questions (need user input before executing)

1. **HF gate**: have you accepted the Gemma license terms on your OSO-AI HF account? (All
   google/gemma-* are gated. Without acceptance, we must use community mirrors → audit first.)
2. Which target: Gemma 4 preview (newest arch science) vs Gemma 3.5-QAT (quantization
   science) vs both?
3. Size: 24GB Mac constraint — a 4B-class model is comfortable; 12B+ needs Q4 from the start
   (extraction on quantized weights = what Qwable taught us is doable but lossier).

## Deliverables if approved

- experiments/gemma_pipeline.py run log (guarded k-sweep, strict evals, jailbreak set)
- exported Q4_K_M GGUF
- HF publish + card + README index update
- RESULTS.md section: "Gemma transfer" (or a third cross-arch negative result — either is
  publishable science)
