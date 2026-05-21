# Minimal llama.cpp — okuma için

Bu klasör llama.cpp'nin **terminal'den tek bir GGUF modelle text üretmek için
gereken en az kod**ı içerir. Server, multimodal, common util, vendor (httplib/json/stb),
GPU backend'leri (CUDA/Vulkan/SYCL...), Python conversion script'leri, doc/test/bench
yok. Sadece okuyup anlamak için.

## Yapı

```
include/              libllama public API (llama.h, llama-cpp.h)
src/                  libllama core (model load, KV cache, sampler, tokenizer)
  models/             her LLM mimarisi için forward graph (gemma, llama, qwen, ...)
ggml/                 tensor library
  include/            ggml.h, ggml-alloc.h, ggml-backend.h, gguf.h
  src/                core tensor ops + quantization + GGUF reader
    ggml-cpu/         CPU backend (ARM NEON, x86 SIMD)
    ggml-metal/       Apple Silicon GPU backend (.metal shader + dispatch)
examples/simple/      ENTRY POINT — 223 satır, sadece llama.h kullanır
cmake/                build helper modülleri
CMakeLists.txt        bu klasördeki kendi minimal build script'i
```

Toplam ~10 MB. Orijinal fork ~500 MB.

## Build

```bash
cd /Users/enes/Desktop/all/attention/only-needed-files
cmake -B build
cmake --build build -j 8
```

Çıktı: `./build/bin/llama-simple` ve `./build/bin/libllama.dylib`, `libggml*.dylib`.

## Çalıştır

```bash
./build/bin/llama-simple -m /Users/enes/Desktop/all/llms/Gemma-4-E4B-Uncensored-Q6.gguf -n 64 "Merhaba"
```

## Okuma sırası (yeni başlayan için)

1. **`include/llama.h`** — Public API surface. Burada ne fonksiyonlar var, ne yapıyorlar.
2. **`examples/simple/simple.cpp`** — API'nin nasıl kullanıldığını gör. 223 satır, baştan sona oku. Buradan diğer dosyalara nasıl atladığını gözle.
3. **`src/llama.cpp`** — `llama_init_from_model`, `llama_decode`, ana orchestration.
4. **`src/llama-model.cpp`** + **`llama-model-loader.cpp`** — GGUF dosyasından model nasıl yükleniyor.
5. **`src/llama-vocab.cpp`** — Tokenize / detokenize. SentencePiece, BPE, vb.
6. **`src/llama-context.cpp`** + **`llama-kv-cache.cpp`** — KV cache, attention state yönetimi.
7. **`src/llama-sampler.cpp`** — Sampling pipeline (greedy, top-k, top-p, temperature).
8. **`src/llama-graph.cpp`** — Compute graph oluşturma (attention + FFN + norm).
9. **`src/models/gemma3.cpp`** (veya istediğin mimari) — O modelin forward pass'i nasıl.
10. **`ggml/include/ggml.h`** + **`ggml/src/ggml.c`** — Tensor library. matmul, softmax, rope, vb.
11. **`ggml/src/ggml-metal/`** — Apple GPU dispatch + `.metal` shader dosyaları.
12. **`ggml/src/ggml-quants.c`** — Q4/Q6/Q8 dequantization (Gemma Q6_K nasıl float'a açılıyor).

## Notlar

- `src/CMakeLists.txt` `models/*.cpp` GLOB ile bütün mimarileri pull ediyor. Tek mimari (Gemma) istiyorsan src/models/ içindeki diğer .cpp'leri silebilirsin — sonra `cmake -B build` yenile.
- `ggml-cpu/` çok büyük (1+ MB) çünkü her CPU arch için ayrı SIMD kernel var (arm64 NEON, x86 AVX2/AVX512, vb). Sadece M-series için sade tutmak istersen ARM-only patch lazım.
- Mimari listesini görmek: `ls src/models/` — ~80 model variant.
