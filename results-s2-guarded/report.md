# Abliteration Eval Report

| Metric | Baseline | Abliterated | Delta |
|---|---|---|---|
| Harmful refusal rate | 97.5% | 17.5% | -80.0% |
| Harmful degenerate rate | 0.0% | 10.0% | +10.0% |
| Benign refusal rate | 10.0% | 0.0% | -10.0% |
| Benign degenerate rate | 2.5% | 5.0% | +2.5% |
| Perplexity (wiki) | 9.44 | 12.87 | +3.42 |

Device: mps, n=40 prompts per set.
Degenerate = repetition loop detected (guard added post-Qwable finding); counted as a separate
failure mode, NOT as compliance. True compliance = 100% - refusal% - degenerate%.
