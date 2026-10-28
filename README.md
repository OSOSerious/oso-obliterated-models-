# obliteratus 🔥🧠

Minimal, reproducible **refusal-direction abliteration** pipeline for small open-weight LLMs.
Based on Arditi et al., *"Refusal in Language Models Is Mediated by a Single Direction"* (2024)
and the community abliteration notebooks popularized by FailSpy.

**Pipeline:** extract → ablate → eval → export → publish

```
obliteratus/
├── extraction/diff.py       # capture activations, mean-diff refusal direction per layer
├── ablation/refuse.py       # orthogonalize weights against the direction
├── evals/eval.py            # refusal rate + capability regression harness
├── export/quantize.sh       # GGUF export + llama.cpp quantization
├── prompts/                 # harmful / benign probe sets
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Extract refusal directions (needs ~8GB RAM for 1-2B models on MPS)
python extraction/diff.py --model Qwen/Qwen2.5-1.5B-Instruct --out directions/

# 2. Ablate
python ablation/refuse.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --directions directions/qwen2.5-1.5b.pt --out models/abliterated/

# 3. Evaluate (before vs after)
python evals/eval.py --baseline Qwen/Qwen2.5-1.5B-Instruct \
    --abliterated models/abliterated/

# 4. Export to GGUF for Ollama
bash export/quantize.sh models/abliterated/
```

Runs on Apple Silicon (MPS). No GPU cluster needed.

## What we measure (honestly)

| Metric | Baseline | Abliterated | Delta |
|--------|----------|-------------|-------|
| Refusal rate (harmful set) | tracked | tracked | the headline number |
| Refusal rate (benign set) | tracked | tracked | should stay ~0 |
| Perplexity (wikitext-2) | tracked | tracked | capability regression check |
| Qualitative samples | logged | logged | the "useful failures" |

Negative results are logged, not hidden. If a layer choice destroys the model, that goes in the README.

## Status

- [x] Extraction working on Qwen2.5-1.5B (naive + matched-pair)
- [x] Weight ablation: all-layers works partially (60% refusal) but damages model (ppl 9.44→17.09); single-layer: no effect at 1.5B scale
- [x] **Inference-time ablation + matched direction, full eval n=40: harmful refusal 97.5%→0%, benign false-refusal 10%→0%, ppl 9.44→14.85**
- [x] Eval harness (refusal + false-refusal + perplexity) with steering support
- [ ] GGUF export verified in Ollama (steering needs a custom runtime hook — see below)
- [ ] Published: HF model + this repo

**Note on export:** inference-time ablation lives in the forward pass, not weights, so plain GGUF export doesn't carry it. Path to Ollama: (a) bake into weights at the band/strength sweet spot, or (b) serve with vLLM-style hooks. Tracked as open work.

## Credits

- Arditi, A., et al. (2024) — the technique
- FailPy — community abliteration notebooks
- pliny — TASK FORCE: OBLITERATUS, the reason this exists

License: MIT
