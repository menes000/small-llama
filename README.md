# Minimal llama.cpp — Gemma 4 Speculative Decoding

![Platform](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey)
![Language](https://img.shields.io/badge/language-C%2B%2B17-blue)
![Backend](https://img.shields.io/badge/backend-Metal%20%7C%20CPU%20NEON-orange)
![Status](https://img.shields.io/badge/output-lossless-brightgreen)

llama.cpp'nin Gemma 4 ile çalışmak için gereken **en az kodu**. Server, multimodal, vendor (httplib/json/stb), Python script, doc/test/bench yok. Okuyup anlamak ve deney yapmak için.

**Full tree:** ~500 MB — **Bu tree:** ~55 MB

---

## Özellikler

- **Speculative decoding (SD):** Gemma 4 E4B-it Q8 üzerinde **1.69× speedup** (M2 Pro, lossless)
- **Saguaro SSD:** arXiv [2603.03251](https://arxiv.org/abs/2603.03251) implementasyonu — geometric fan-out cache + async overlap
- **Multi-turn chat REPL:** Gemma 4 chat template, `read_file` / `list_dir` sandboxed tool'ları
- **CPU draft + Metal target:** draft CPU NEON, target Metal GPU — ayrı backend paralel overlap
- **Lossless:** verify her zaman target dağılımını garantiler, output token-identical

---

## Performans (M2 Pro, Gemma 4 E4B-it Q8)

| Prompt | Baseline | SD (CPU draft) | Speedup |
|---|---|---|---|
| primes (40 prime) | ~30 t/s | 54.3 t/s | **1.81×** |
| count (1–60) | ~30 t/s | 29.1 t/s | ~1× |
| counting task (acc %85) | ~30 t/s | 49.8 t/s | **1.69×** |

Draft: `gemma-4-E4B-it-assistant.Q8_0.gguf` (96 MB EAGLE/MTP head). Output her testte byte-identical.

---

## Gereksinimler

- macOS + Apple Silicon (M1/M2/M3) — Metal backend için
- CMake 3.21+
- Xcode Command Line Tools
- GGUF model dosyaları (aşağıda)

---

## Yapı

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
  simple/             ENTRY POINT — 223 satır, sadece llama.h kullanır
  chat/               Multi-turn REPL + tool use + opsiyonel SD
  spec/               Speculative decoding benchmark (SD + Saguaro SSD)
docs/
  guides/             HOW_TO_RUN.md, USAGE.md
  chat/               Chat örnek raporları
  speculative-decoding/  SD/SSD benchmark sonuçları
  handoff/            Oturum handoff'ları ve master rapor
cmake/                build helper modülleri
CMakeLists.txt
```

---

## Build

```bash
cd /Users/enes/Desktop/all/less-llama-cpp/only-needed-files
cmake -B build
cmake --build build -j 8
```

Çıktı: `./build/bin/llama-simple`, `llama-chat`, `llama-spec` + `libllama.dylib`, `libggml*.dylib`

---

## Çalıştır

```bash
# Tek prompt
./build/bin/llama-simple \
  -m /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -n 64 "Merhaba"

# Interaktif chat (multi-turn, tool use, opsiyonel SD)
./build/bin/llama-chat \
  -m /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf

# Speculative decoding benchmark
./build/bin/llama-spec \
  -m /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf \
  -p "Count from 1 to 60:" -n 128
```

Tüm flag'ler ve default değerler: [`docs/guides/USAGE.md`](docs/guides/USAGE.md)

---

## Okuma Sırası (yeni başlayan için)

| # | Dosya | Ne öğrenirsin |
|---|---|---|
| 1 | `include/llama.h` | Public API surface |
| 2 | `examples/simple/simple.cpp` | API nasıl kullanılır (223 satır) |
| 3 | `src/llama.cpp` | `llama_init_from_model`, `llama_decode` orchestration |
| 4 | `src/llama-model.cpp` + `llama-model-loader.cpp` | GGUF → model yükleme |
| 5 | `src/llama-vocab.cpp` | Tokenizer (BPE, SentencePiece) |
| 6 | `src/llama-context.cpp` + `llama-kv-cache.cpp` | KV cache, attention state |
| 7 | `src/llama-sampler.cpp` | Sampling (greedy, top-k, top-p, temp) |
| 8 | `src/llama-graph.cpp` | Compute graph (attention + FFN + norm) |
| 9 | `src/models/gemma4.cpp` | Gemma 4 forward pass |
| 10 | `ggml/include/ggml.h` + `ggml/src/ggml.c` | Tensor library (matmul, rope, softmax) |
| 11 | `ggml/src/ggml-metal/` | Apple GPU dispatch + `.metal` shader'lar |
| 12 | `ggml/src/ggml-quants.c` | Q4/Q6/Q8 dequantization |

---

## Notlar

- `src/CMakeLists.txt` `models/*.cpp` GLOB ile pull ediyor. Şu an sadece `gemma4.cpp` + `gemma4_assistant.cpp` var.
- `ggml-cpu/` büyük (1+ MB) — her CPU arch için ayrı SIMD kernel (ARM NEON, x86 AVX2/AVX512). Sadece M-series için ARM-only patch lazım.
- SD/SSD tasarımı ve benchmark detayları: [`docs/speculative-decoding/`](docs/speculative-decoding/) ve [`docs/handoff/MASTER_REPORT.md`](docs/handoff/MASTER_REPORT.md)
