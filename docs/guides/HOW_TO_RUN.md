# Nasıl Çalıştırılır — Hızlı Referans

İki binary var:
- `llama-spec` — tek prompt, batch çalıştırma (benchmark, test)
- `llama-chat` — interaktif REPL (multi-turn chat, tools)

İkisi de aynı SSD/CPU-draft altyapısını kullanır.

---

## Modellerin Yeri

```
/Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf              # target
/Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf    # draft
```

E2B çifti de var ama büyük target'larda CPU draft daha iyi sonuç verdiği için E4B kullan.

---

## Build (gerekirse)

```bash
cd /Users/enes/Desktop/all/less-llama-cpp/only-needed-files
cmake --build build --target llama-spec llama-chat
```

---

## llama-chat (interaktif)

### En Hızlı Mod (Default — SD + CPU draft)

```bash
./build/bin/llama-chat \
  -m  /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf \
  --no-tools
```

**~73 t/s** primes prompt, E4B Q8 çiftinde. Hiç ekstra flag yok.

### SSD Aktif (paper algoritması, async overlap)

```bash
./build/bin/llama-chat \
  -m  /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf \
  --no-tools --draft-max 5 --ssd-fan-out 7 --ssd-async
```

**~70 t/s** + lossless. SSD overhead async overlap ile tamamen gizleniyor (hidden=%29).

### Tool'lar Aktif (read_file, list_dir)

```bash
./build/bin/llama-chat \
  -m  /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf \
  --root /Users/enes/Desktop/all --yolo
```

`--yolo` = her tool çağrısını otomatik onayla. Çıkar `--yolo` interaktif onay için.

### Eski Default (her ikisi GPU — SD Metal)

```bash
./build/bin/llama-chat -m TARGET -md DRAFT --no-tools -ngl-draft 99
```

E4B'de bu mod en yavaş. E2B çiftinde hâlâ en hızlı.

---

## llama-spec (benchmark)

### Default (SD CPU draft)

```bash
./build/bin/llama-spec \
  -m  /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf \
  -p "list the first 40 prime numbers" -n 150 --draft-max 5
```

### SSD Aktif

```bash
./build/bin/llama-spec \
  -m  /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf \
  -p "list the first 40 prime numbers" -n 150 --draft-max 5 \
  --ssd-fan-out 7 --ssd-async
```

---

## Flag Referans

### Backend split (ikisinde de aynı)
| Flag | Default | Açıklama |
|---|---|---|
| `-ngl N` | 99 | Target GPU layer sayısı (99 = full Metal) |
| `-ngl-draft N` | **0** | Draft GPU layer sayısı (0 = CPU NEON) |

### SSD (paper 2603.03251v3)
| Flag | Default | Açıklama |
|---|---|---|
| `--ssd-fan-out B` | 1 | Toplam fan-out bütçesi (B≤K = kapalı; B>K aktif) |
| `--ssd-r R` | 1.0 | Geometric fan-out üs (Thm 12) |
| `--ssd-sampling-C C` | 1.0 | Saguaro sampling downweight (greedy mode'da etkisiz) |
| `--ssd-async` | off | Target verify ile post-pass paralel |

### Genel
| Flag | Default | Açıklama |
|---|---|---|
| `--draft-max N` | spec=5, chat=3 | Round başına draft token sayısı (K) |
| `-n N` | spec=128, chat=1024 | Max generation token |
| `-c N` | 4096 | Context size |
| `-p "..."` | "hello" (spec) | Tek prompt (sadece spec) |

### Chat'e özel
| Flag | Default | Açıklama |
|---|---|---|
| `--no-tools` | off | Tool'ları devre dışı bırak |
| `--no-sd` | off | SD'yi devre dışı bırak (-md verilse bile) |
| `--thinking` | off | Thinking mode |
| `--kv-f16` | off | K/V cache f16 (default q8) |
| `--root DIR` | $HOME | Tool sandbox kökü |
| `--yolo` | off | Tool onaylarını atla |

---

## Hangi Mode Ne Kazandırır?

| Senaryo | Komut eki | Beklenen kazanç |
|---|---|---|
| **Sadece hız** | (hiçbir şey) | E4B'de SD Metal'den %12 hızlı |
| Paper SSD demo | `--ssd-fan-out 7 --ssd-async` | SD CPU ile eşit, lossless |
| Hit rate ölçmek | `--ssd-fan-out 10 --ssd-async` | Hit %20-40, t/s biraz düşer |
| Eski (her ikisi GPU) | `-ngl-draft 99` | E4B'de yavaş, E2B'de hızlı |

---

## Output Yorumlama

### Spec stats:
```
[spec] emitted=150 in 1.36s = 110.2 t/s | rounds=30 drafted=150 accepted=119 accept_rate=79.3% acc_per_round=3.97
[spec] SSD: hits=0 misses=19 hit_rate=0.0% extra_draft_decodes=77 steps_saved=0 net_extra=77
[spec] timing: verify=42ms post_pass=256ms overlap_region=261ms hidden=12.5%   async=on draft_backend=CPU
```

### Chat per-turn stats:
```
[stats] gen=119 tok / 1.71s = 69.8 t/s | prefill=31 tok / 0.23s = 135 t/s | rounds=22 drafted=110 accepted=99 acc_rate=90.0% | total=1.92s
[stats] SSD: hits=0 misses=5 hit=0.0% extra=41 saved=0 | verify=48ms post=79ms overlap=89ms hidden=29.3% async=on draft=CPU
```

### Ana metrikler
- `t/s` → ana hız (yüksek = iyi)
- `acc_rate` → draft kalitesi (yüksek = iyi)
- `hidden%` → async overlap ne kadar gizledi (yüksek = iyi)
- `net_extra` → SSD'nin net maliyeti (düşük/negatif = iyi)

---

## Lossless Doğrulama

Her mode aynı output'u üretir (greedy target deterministik):

```bash
TGT=/Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf
DRF=/Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf
P="list 30 primes"

./build/bin/llama-spec -m $TGT -md $DRF -p "$P" -n 100                          > /tmp/sd.out
./build/bin/llama-spec -m $TGT -md $DRF -p "$P" -n 100 --ssd-fan-out 7 --ssd-async > /tmp/ssd.out
diff /tmp/sd.out /tmp/ssd.out  # boş olmalı
```

---

## Hızlı Tek-Tıklık Karşılaştırma

```bash
TGT=/Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf
DRF=/Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf
P="list the first 40 prime numbers"

echo "=== SD CPU draft (default, en hızlı) ==="
./build/bin/llama-spec -m $TGT -md $DRF -p "$P" -n 150 --draft-max 5 2>&1 \
  | grep -E "^\[spec\] (emitted|timing)"

echo "=== SSD async ==="
./build/bin/llama-spec -m $TGT -md $DRF -p "$P" -n 150 --draft-max 5 \
  --ssd-fan-out 7 --ssd-async 2>&1 \
  | grep -E "^\[spec\] (emitted|SSD: hits|timing)"

echo "=== Chat default ==="
printf "list the first 30 primes\n" | ./build/bin/llama-chat \
  -m $TGT -md $DRF --no-tools -n 200 2>&1 \
  | grep -E "^\[stats\]"
```

---

## İlgili Diğer Dokümanlar

- `SSD_REPORT.md` — Algoritma detay + bug fix history
- `SSD_COMPARISON.md` — E2B SD vs SSD sweep
- `SSD_ASYNC_REPORT.md` — E2B async overlap analizi
- `SSD_E4B_REPORT.md` — E4B Q8 çifti karşılaştırma + sürpriz CPU bulgular
- `CHAT_SSD_REPORT.md` — Chat'e port detayı
- `USAGE.md` — spec.cpp odaklı eski kılavuz
