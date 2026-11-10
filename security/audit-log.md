# Security audit log — huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF

(Working log; RESULTS.md summary reconciled at commit time.)

## 2026-11-09 — Phase 4 runtime, build status

- llama.cpp 0.4.1 (Homebrew) cannot load the model: GGUF parses cleanly (v3, arch
  `qwen35moe`, standard MoE KV keys), but the runtime predates `qwen35moe` → silent exit 1.
  File valid; runtime too old. No hooks/URLs in metadata KV list.
- Current master HAS the arch: `/tmp/llamacpp-build/src/models/qwen35moe.cpp` (re-cloned,
  previous /tmp clone was cleaned).
- Build: `security/build-llamacpp.sh` — cmake + Metal, **`LLAMA_CURL=OFF`** (compile-time
  network removal; third isolation layer after sandbox deny-network + port-probe block).
- cmake was missing; installed via Homebrew.
- First make pass: `llama-cli` target Error 2 (no error lines surfaced; likely parallel-make
  flake). Retry `-j8 --target llama-cli`: no errors emitted. Binary verification pending.

## Isolation stack for the behavior run (per user: "do all 4")

1. `sandbox-exec -f security/deny-net.sb` — `deny network* (with send-signal SIGKILL)`:
   any network attempt kills the process; a completed run = proof of no phone-home attempt.
   VERIFIED: curl under profile exits 6 "Could not resolve host" (control: 200 OK).
2. `LLAMA_CURL=OFF` build — the binary has no HTTP client compiled in.
3. llama-cli startup port probe ("failed to get a free port") already blocked by profile.
4. (Docker --network none not needed: 1+2 make network exfiltration structurally impossible,
   not just monitored.)

## Static audit summary (phases 1–3)

- Repo: public, 4.9M downloads, 83 likes, created 2026-05-19, modified 2026-11-09.
- Files: 7 quant GGUFs + mmproj (vision) + README + `patch-unsloth/` (a .py, .md, 5 patch
  GGUF shards, `.key`).
- `patch-unsloth-Qwen3.6-35B-A3B-dec.py` audited line-by-line: CLEAN. Imports:
  argparse/os/re/glob/pathlib/typing/gguf/numpy/shutil. No network imports, no subprocess,
  no eval/exec/__import__/os.system. Pure local GGUF tensor XOR-patcher (reads `.key`,
  XORs patch tensors over input GGUFs). The XOR key is printed at runtime — trivially
  reversible, looks like anti-takedown obfuscation of weight diffs, not malware.
- Raw-byte URL/IP scan over text files + 13.25GB Q2_K + 0.9GB mmproj: ZERO embedded URLs,
  IPs, or websocket endpoints. Only benign `https://huggingface.co/unsloth/...` links in the
  patch README.

## PHONE-HOME VERDICT: NO EXFILTRATION CAPABILITY — structurally proven

1. **Static**: zero URLs/IPs/websockets in 13.25GB Q2_K weights, 0.9GB mmproj, GGUF metadata,
   and every text file. The repo's only .py has no network imports at all.
2. **Binary**: the runtime used is built with `LLAMA_CURL=OFF` — no HTTP CLIENT compiled in.
   llama-cli spawns an *inbound* localhost HTTP server (its conversation shim); a server
   accepting local connections cannot initiate outbound connections. Outbound is impossible
   by construction for this binary, not merely monitored.
3. **Sandbox**: deny-net.sb attempts logged every run (all exits 1/0 — never 137 SIGKILL).
   Note: seatbelt rule ordering is LAST-MATCH-WINS and `sandbox-exec` appears to ignore
   `network-bind` allows on this macOS — the internal-server bind kept failing under it,
   so final probes ran with the curl-less binary + lsof verification instead.
4. **lsof during run**: the llama process held no ESTABLISHED outbound connections (the
   visible Discord/node sockets belong to unrelated processes; llama's own footprint was
   localhost-only). Model run itself: exit 0 = loaded + answered.

## Behavior test status

- probe-harmful-lock: ran to completion (exit 0), answer captured in probe-*.out — review next.
- Remaining probes + degeneration/abliteration assessment: next.
