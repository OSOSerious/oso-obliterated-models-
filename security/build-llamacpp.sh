#!/bin/bash
# Build current llama.cpp master (has qwen35moe/MTP support) for the huihui behavior test.
# Output: /tmp/llamacpp-build/build/bin/llama-cli
# -DLLAMA_CURL=OFF: compile-time network removal (isolation layer 2).
# -DLLAMA_BUILD_UI=OFF: web-UI asset bundler is broken in shallow clone (stale HF stamp
#   -> missing dist assets -> ui-assets.cmake file(ARCHIVE_CREATE) fails). CLI doesn't need it.
set -e
cd /tmp/llamacpp-build
cmake -B build -DGGML_METAL=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON
# Workaround: llama-ui-assets gzip step requires asset files that the HF-stamp skip left
# missing (dist/_app/immutable/assets/*). The CLI never serves the UI. Create any asset
# files referenced in the asset list but absent on disk, so file(ARCHIVE_CREATE) succeeds.
if [ -d build/tools/ui/dist ]; then
  # The complete asset set (incl. content-hashed bundles) lives in dist.tar.gz.
  # Provisioner left dist/ partial; extract the tarball over it so emit_files finds everything.
  if [ -f build/tools/ui/dist.tar.gz ]; then
    tar -xzf build/tools/ui/dist.tar.gz -C build/tools/ui/dist/ 2>/dev/null || true
    echo "extracted dist.tar.gz over dist/"
  fi
  grep -rhoE "dist/[A-Za-z0-9_./-]+\.(css|js|html|svg|png|json|woff2?|ico)" \
      build/tools/ui/CMakeFiles build/tools/ui 2>/dev/null | sort -u | while read -r rel; do
    [ -f "build/tools/ui/$rel" ] || { mkdir -p "build/tools/ui/$(dirname "$rel")"; touch "build/tools/ui/$rel"; echo "stubbed: $rel"; }
  done
fi
cmake --build build --config Release -j "$(sysctl -n hw.ncpu)" --target llama-cli llama-gguf llama-gguf-hash
echo "BUILD_OK: $(ls -la build/bin/llama-cli)"
