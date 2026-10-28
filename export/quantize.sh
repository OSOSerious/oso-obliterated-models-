#!/usr/bin/env bash
# export/quantize.sh — convert abliterated HF model to GGUF + quantize for llama.cpp/Ollama
# Usage: bash export/quantize.sh models/abliterated
set -euo pipefail
MODEL_DIR="${1:?usage: quantize.sh <model_dir>}"
OUT_GGUF="${MODEL_DIR}/../$(basename "$MODEL_DIR").gguf"

# requires llama.cpp repo sibling: git clone https://github.com/ggerganov/llama.cpp
LLAMA_CPP="${LLAMA_CPP:-../llama.cpp}"

python "$LLAMA_CPP/convert_hf_to_gguf.py" "$MODEL_DIR" --outtype f16 --outfile "$OUT_GGUF"
"$LLAMA_CPP/build/bin/llama-quantize" "$OUT_GGUF" "${OUT_GGUF%.gguf}-Q4_K_M.gguf" Q4_K_M

cat <<EOF
Done:
  f16:     $OUT_GGUF
  Q4_K_M:  ${OUT_GGUF%.gguf}-Q4_K_M.gguf

Test in Ollama — create Modelfile:
  FROM ${OUT_GGUF%.gguf}-Q4_K_M.gguf
then: ollama create obliterated -f Modelfile && ollama run obliterated
EOF
