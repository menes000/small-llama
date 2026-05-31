# llama-spec Kullanım Kılavuzu

`examples/spec/spec.cpp` — Gemma4 speculative decoding (SD) + opsiyonel Saguaro SSD (paper arXiv 2603.03251v3).

## Defaults (Hiçbir Flag Vermezsen)

```
-ngl 99           target Metal GPU (full)
-ngl-draft 0      draft CPU NEON         ← yeni default
--ssd-fan-out 1   SSD KAPALI (saf SD)    ← default
--ssd-async       off                    ← default
```

→ **Default mode = SD + CPU draft + Metal target.** E4B Q8 çiftinde en hızlı.

## Standart Komutlar

### 1. Default (en hızlı E4B Q8)
```bash
./build/bin/llama-spec \
  -m  /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf \
  -p "list the first 40 prime numbers" -n 150 --draft-max 5
```
SSD yok, sadece SD. Draft CPU, target Metal. **38 t/s** (primes, ortalama).

### 2. SSD aktif (paper algoritması)
```bash
./build/bin/llama-spec \
  -m  /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf \
  -p "list the first 40 prime numbers" -n 150 --draft-max 5 \
  --ssd-fan-out 7 --ssd-async
```
SSD speculation cache + async overlap aktif. **~35 t/s** (default'tan %8 yavaş, çünkü post-pass overhead tam kapanmıyor).

### 3. Tüm SSD opsiyonları
```bash
--ssd-fan-out B      bütçe (default 1 = kapalı; >K ile aktif, tipik 7-15)
--ssd-r R            geometric fan-out exponent (default 1.0)
--ssd-sampling-C C   Saguaro sampling (default 1.0; greedy mode warns)
--ssd-async          target verify ile post-pass paralel (sadece farklı backend'de)
```

## Hangi Modu Ne Zaman Kullan?

| Amaç | Komut eki | Beklenen |
|---|---|---|
| **Maks hız** | (hiçbir şey) | SD CPU draft, E4B'de 38 t/s ort |
| Paper SSD'yi göster | `--ssd-fan-out 7 --ssd-async` | SSD aktif, lossless, ~35 t/s |
| Eski varsayılan (SD Metal) | `-ngl-draft 99` | Draft GPU'da, E4B'de 34 t/s |
| SSD without overlap | `--ssd-fan-out 7` | Serial SSD, ölçüm için |
| Hit rate ölçmek | `--ssd-fan-out 10 --ssd-async` | Daha yüksek hit ama daha yavaş |

## Output Yorumlama

### Banner
```
[spec] backends: target ngl=99 (GPU)   draft ngl=0 (CPU)
[spec] prompt tokens=35 n_predict=150 n_draft_max=5 n_centroids=2048 top_k=32 backbone=2560
[spec] SSD: fan_out_budget=1 r=1.00 sampling_C=1.00 -> OFF (budget <= K, no alternatives)
```

### Stats
```
[spec] emitted=150 in 2.76s = 54.3 t/s | rounds=41 drafted=205 accepted=108 accept_rate=52.7% acc_per_round=2.63
[spec] SSD: hits=0 misses=0 hit_rate=0.0% extra_draft_decodes=0 steps_saved=0 net_extra=0
[spec] timing: verify=64ms post_pass=0ms overlap_region=64ms hidden=-0.0%   async=off draft_backend=CPU
```

Alan açıklamaları:
- `t/s` — emitted tokens / wall time
- `accept_rate` — kabul oranı (draft kalitesi göstergesi)
- `acc_per_round` — round başına kabul edilen token sayısı
- `hits/misses/hit_rate` — SSD cache hit oranı
- `extra_draft_decodes` — SSD'nin yaptığı ekstra draft compute
- `steps_saved` — cache hit ile atlanan draft step sayısı
- `verify` — total target Metal verify wall-time
- `post_pass` — total SSD alt-decode wall-time
- `hidden%` — async overlap ile gizlenen sürenin yüzdesi

## Lossless Garantisi

Her mod aynı output'u üretir (greedy target deterministik). Aşağıdaki diff her zaman 0 olmalı:

```bash
./build/bin/llama-spec -m T -md D -p "X" -n 100                              > /tmp/sd.out
./build/bin/llama-spec -m T -md D -p "X" -n 100 --ssd-fan-out 7 --ssd-async  > /tmp/ssd.out
diff /tmp/sd.out /tmp/ssd.out  # boş olmalı
```

## Önemli Notlar

1. **E4B target + E4B assistant** çifti — assistant ve target model UYUMLU olmalı. Karışım acceptance %0 verir.
2. **E2B çiftinde** SD Metal (her ikisi GPU) hala en hızlı. CPU draft kazancı sadece BÜYÜK target'larda (E4B+) var.
3. **SSD net kazanç** Mac M2 Pro'da yok — paper'ın hız iddiası ayrı GPU'lar gerektiriyor. SSD async sadece SD Metal'den biraz iyi (~+3%), SD CPU'dan daha yavaş.

## Tek Tıklık Karşılaştırma

```bash
TGT=/Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf
DRF=/Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf
P="list the first 40 prime numbers"

echo "=== Default (SD CPU draft) ==="
./build/bin/llama-spec -m $TGT -md $DRF -p "$P" -n 150 --draft-max 5 2>&1 \
  | grep -E "^\[spec\] (emitted|timing)" | head -2

echo "=== SSD aktif ==="
./build/bin/llama-spec -m $TGT -md $DRF -p "$P" -n 150 --draft-max 5 \
  --ssd-fan-out 7 --ssd-async 2>&1 \
  | grep -E "^\[spec\] (emitted|SSD: hits|timing)" | head -3
```
