# Chat'te Yeni Yapı (SSD + CPU draft + async overlap)

`examples/chat/chat.cpp` 858 → 1019 satır. spec.cpp'deki tüm SSD altyapısı portlandı:
- CPU/GPU split per-model (`-ngl-draft`)
- Geometric fan-out cache (`--ssd-fan-out`, `--ssd-r`)
- Saguaro sampling scaffold (`--ssd-sampling-C`)
- `std::async` verify/post-pass overlap (`--ssd-async`)
- Cache hit lookup → sonraki round'ta seed
- Per-turn stats (hits, hidden%, vb.)

## Defaults

```
-ngl 99           target Metal GPU
-ngl-draft 0      draft CPU NEON          ← yeni default
--draft-max 3     chat default (spec'te 5)
--ssd-fan-out 1   SSD KAPALI              ← default
--ssd-async       off                     ← default
```

→ Default = **SD + draft CPU + target Metal** (E4B target için en hızlı).

## Test Sonuçları (E4B Q8 çifti, "list first 30 primes")

### `--draft-max 3` (chat default)
| Mode | t/s | hidden% | hit% |
|---|---|---|---|
| SD Metal (both GPU) | 65.2 | — | — |
| **SD CPU** | **72.8** | — | — |
| SSD serial CPU | 64.5 | 0% | 0 |
| **SSD async CPU** | **72.5** | **18.9%** | 0 |

→ SSD async = SD CPU (overhead tam gizlendi).  
→ Hit=0 çünkü K=3 + accept %98.9 → bonus K pozisyonunda, cache K-1'e kadar.

### `--draft-max 5` (SSD için daha iyi)
| Mode | t/s | hidden% | timing |
|---|---|---|---|
| SD CPU | 69.8 | — | — |
| **SSD async CPU** | **69.8** | **29.3%** | verify=48ms post=79ms overlap=89ms |

→ Tam eşit. SSD ekleme tamamen bedava.  
→ Hidden %29.3 — spec.cpp'deki %20-22'den daha iyi.

### Output Lossless ✓
4-mode × 1 prompt = 4 çalıştırma. Tüm `*.out` byte-identical.

## Yeni Stats Hattı

SSD aktifken (B>K) per-turn:

```
[stats] gen=119 tok / 1.71s = 69.8 t/s | prefill=31 tok / 0.23s = 135 t/s | rounds=22 ...
[stats] SSD: hits=0 misses=5 hit=0.0% extra=41 saved=0 | verify=48ms post=79ms overlap=89ms hidden=29.3% async=on draft=CPU
```

Alanlar:
- `hits/misses/hit%` — speculation cache hit oranı
- `extra` — SSD'nin yaptığı ekstra draft decode (post-pass + restore)
- `saved` — cache hit ile atlanan draft step
- `verify/post/overlap` — wall-time'lar (ms)
- `hidden%` — async overlap ile gizlenen toplam sürenin yüzdesi
- `async` — `--ssd-async` aktif mi
- `draft` — draft model'in backend'i (CPU/GPU)

## Standart Komutlar

### Default (en hızlı E4B):
```bash
./build/bin/llama-chat \
  -m  /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf \
  --no-tools
```

### SSD aktif (paper algoritması, async overlap):
```bash
./build/bin/llama-chat \
  -m  /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf \
  --no-tools --draft-max 5 --ssd-fan-out 7 --ssd-async
```

### Eski varsayılan (her ikisi GPU):
```bash
./build/bin/llama-chat -m TARGET -md DRAFT --no-tools -ngl-draft 99
```

## Önemli Notlar

1. **Per-turn reset**: `ssd_have_seed` ve `spec_cache` her yeni user turn başlangıcında temizlenir. Prefill pending_feat'i sıfırladığı için eski round'ların seed'i geçersiz.

2. **`reset_chat()` SSD state temizler**: KV overflow veya decode hatasında SSD state de reset edilir (tutarlılık için).

3. **Tool calls SSD'yi etkilemez**: Tool hop'ları arasında yeni run_inference çağrıları aynı turn'de devam eder. SSD seed turn boyunca round-arası persist eder.

4. **Hit rate K-bağımlı**: Chat default `--draft-max 3` küçük → bonus genelde K=3 pozisyonuna düşer → cache miss. Yüksek hit için `--draft-max 5+` öner.

5. **Lossless garantisi**: Greedy target deterministik. SSD mode'u bile aktif olsa diff output her zaman boş.

## Implementation Detayları

Değişen kısımlar (`examples/chat/chat.cpp`):
- `struct args` — yeni alanlar (line ~41)
- `usage()` — yeni flag listesi (line ~57)
- `parse_args()` — yeni parser case'leri (line ~85)
- `masked_topf` + `geometric_fanout` helpers (line ~250)
- `mp_tgt`/`mp_drf` split (line ~340)
- SSD state + reset_chat hook (line ~497, 540)
- Draft loop 2-phase + post-pass + async dispatch (line ~673)
- Per-turn SSD stats (line ~1080)
- `<future>` + `<cmath>` include (line ~24)

Hiçbir header/API değişikliği yok. Sadece `chat.cpp` ve daha önce yaptığımız `spec.cpp`. `llama-ext.h`, `llama.h`, CMakeLists değişmedi.
