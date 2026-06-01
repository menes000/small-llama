# Gemma 4 Speculative Decoding for llama.cpp

![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey)
![Language](https://img.shields.io/badge/language-C%2B%2B17-blue)
![Backend](https://img.shields.io/badge/backend-Metal%20%7C%20CPU%20NEON-orange)
![Status](https://img.shields.io/badge/output-lossless-brightgreen)

Speculative decoding for Gemma 4 implemented in llama.cpp. **Not available upstream.** Minimal codebase (~55 MB vs full tree ~500 MB) — no server, multimodal, or vendor libs.

---

## What this is

Official llama.cpp has no Gemma 4 speculative decoding. This implements it.

Uses Google's `gemma4_assistant` (78M EAGLE/MTP draft head) to propose tokens; the large target model verifies them in parallel. Output is **byte-identical** to non-speculative decoding — lossless by construction.

**Up to 3.72×** speedup on simple/predictable prompts. Speedup scales with prompt predictability — creative/diverse text is ~1×.

---

## Features

- **Speculative decoding (SD)** — up to **3.72×** speedup, always lossless
- **Saguaro SSD** — arXiv [2603.03251](https://arxiv.org/abs/2603.03251) — geometric fan-out cache + async overlap
- **Multi-turn chat REPL** — Gemma 4 chat template, sandboxed `read_file` / `list_dir` tools
- **CPU draft + Metal target** — draft on CPU NEON, target on Metal GPU — independent backends, true parallel overlap
- **Lossless** — verify guarantees target distribution; output byte-identical across all modes

---

## Performance

**Up to 3.72×** speedup on simple/templated prompts (M2 Pro, E4B Q8 pair). Speedup drops with prompt complexity — creative/diverse text is ~1×. See [`docs/handoff/MASTER_REPORT.md`](docs/handoff/MASTER_REPORT.md) for full benchmark breakdown.

---

## Requirements

- macOS + Apple Silicon (M1/M2/M3) — Metal backend
- CMake 3.21+
- Xcode Command Line Tools
- GGUF model files (see Run section)

---

## Project structure

```
include/              libllama public API (llama.h, llama-cpp.h)
src/                  libllama core (model load, KV cache, sampler, tokenizer)
  models/
    gemma4.cpp                Gemma 4 target forward graph
    gemma4_assistant.cpp      Gemma 4 EAGLE/MTP draft head
ggml/
  include/            ggml.h, ggml-alloc.h, ggml-backend.h, gguf.h
  src/
    ggml-cpu/         CPU backend (ARM NEON, x86 SIMD)
    ggml-metal/       Apple Silicon GPU backend (.metal shader + dispatch)
examples/
  simple/             Entry point — 223 lines, uses only llama.h
  chat/               Multi-turn REPL + tool use + optional SD
  spec/               Speculative decoding benchmark (SD + Saguaro SSD)
docs/
  guides/             HOW_TO_RUN.md, USAGE.md
  chat/               Chat example reports
  speculative-decoding/  SD/SSD benchmark results
  handoff/            Session handoffs and master report
cmake/                Build helper modules
CMakeLists.txt
```

---

## Build

```bash
cmake -B build
cmake --build build -j 8
# output: build/bin/llama-simple  llama-chat  llama-spec
```

---

## Run

```bash
# Single prompt
./build/bin/llama-simple \
  -m /path/to/gemma-4-E4B-it-Q8_0.gguf \
  -n 64 "Hello"

# Interactive chat (multi-turn, tool use, optional SD)
./build/bin/llama-chat \
  -m /path/to/gemma-4-E4B-it-Q8_0.gguf \
  -md /path/to/gemma-4-E4B-it-assistant.Q8_0.gguf

# Speculative decoding benchmark
./build/bin/llama-spec \
  -m /path/to/gemma-4-E4B-it-Q8_0.gguf \
  -md /path/to/gemma-4-E4B-it-assistant.Q8_0.gguf \
  -p "List the first 40 prime numbers" -n 150 --draft-max 5
```

All flags and defaults: [`docs/guides/USAGE.md`](docs/guides/USAGE.md)

---

## Reading order (start here)

| # | File | What you learn |
|---|---|---|
| 1 | `include/llama.h` | Public API surface |
| 2 | `examples/simple/simple.cpp` | How to use the API (223 lines) |
| 3 | `src/llama.cpp` | `llama_init_from_model`, `llama_decode` orchestration |
| 4 | `src/llama-model.cpp` + `llama-model-loader.cpp` | GGUF → model loading |
| 5 | `src/llama-vocab.cpp` | Tokenizer (BPE, SentencePiece) |
| 6 | `src/llama-context.cpp` + `llama-kv-cache.cpp` | KV cache, attention state |
| 7 | `src/llama-sampler.cpp` | Sampling (greedy, top-k, top-p, temp) |
| 8 | `src/llama-graph.cpp` | Compute graph (attention + FFN + norm) |
| 9 | `src/models/gemma4.cpp` | Gemma 4 forward pass |
| 10 | `ggml/include/ggml.h` + `ggml/src/ggml.c` | Tensor library (matmul, rope, softmax) |
| 11 | `ggml/src/ggml-metal/` | Apple GPU dispatch + `.metal` shaders |
| 12 | `ggml/src/ggml-quants.c` | Q4/Q6/Q8 dequantization |

---

## References

- **Speculative Decoding** — Leviathan et al., 2022 — [arXiv 2211.17192](https://arxiv.org/abs/2211.17192)
- **EAGLE** — Li et al., 2024 — [arXiv 2401.15077](https://arxiv.org/abs/2401.15077)
- **Saguaro SSD** — Kumar, Dao, May, 2025 — [arXiv 2603.03251](https://arxiv.org/abs/2603.03251)

---

## Notes

- `src/CMakeLists.txt` pulls `models/*.cpp` via GLOB. Currently only `gemma4.cpp` + `gemma4_assistant.cpp`.
- `ggml-cpu/` is large (1+ MB) — separate SIMD kernels per CPU arch (ARM NEON, x86 AVX2/AVX512). M-series only needs ARM-only.
- SD/SSD design and full benchmark details: [`docs/speculative-decoding/`](docs/speculative-decoding/) and [`docs/handoff/MASTER_REPORT.md`](docs/handoff/MASTER_REPORT.md)
