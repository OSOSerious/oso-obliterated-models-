#!/bin/bash
# experiments/qwable_4bit_run.sh — Qwable follow-up: re-run MLX abliteration on the 4-Bit base.
#
# Why: the 2-bit run (c1843fb) was a negative result — the base model degenerates
# ("the user is the user...") and keyword evals scored gibberish as compliance.
# The 4-bit quant (~18-19GB) fits in 24GB RAM with a metal memory cap and should
# have a healthy base, letting us actually measure abliteration on the 35B MoE.
#
# Usage: bash experiments/qwable_4bit_run.sh
set -e
cd "$(dirname "$0")/.."
PY=../.venv/bin/python

MODEL_ID="usermma/Qwable-v1-mlx-4Bit"
SNAP=$(ls -d ~/.cache/huggingface/hub/models--usermma--Qwable-v1-mlx-4Bit/snapshots/*/ 2>/dev/null | head -1 || true)

if [ -z "$SNAP" ]; then
  echo "[1/2] downloading $MODEL_ID (~19GB)..."
  hf download "$MODEL_ID"
  SNAP=$(ls -d ~/.cache/huggingface/hub/models--usermma--Qwable-v1-mlx-4Bit/snapshots/*/ | head -1)
else
  echo "[1/2] cached: $SNAP"
fi

echo "[2/2] abliterating (s=2.0, matched-pair, all write paths)..."
$PY experiments/mlx_abliterate.py \
  --model "$SNAP" \
  --strength 2.0 \
  --out models/qwable-4bit-abliterated-s2-mlx \
  --mem-gb 16 \
  --eval-n 20
