# E2B Speculative Decoding Benchmark

Tarih: 2026-05-26
Model çifti:
- target: `/Users/enes/Desktop/all/llms/gemma-4-E2B-it-UD-Q8_K_XL.gguf`
- draft:  `/Users/enes/Desktop/all/llms/gemma-4-E2B-it-assistant.F16.gguf`

Ortam: Metal `-ngl 99`, `--temp 0` (greedy), `-n 300`, `-st`.
Komut şablonu (baseline):
```bash
./build/bin/llama-cli -m <target> -p "<prompt>" -n 300 --temp 0 -st -ngl 99
```
Spec:
```bash
./build/bin/llama-cli -m <target> -md <draft> \
  --spec-type draft-mtp --spec-draft-n-max <N> \
  -p "<prompt>" -n 300 --temp 0 -st -ngl 99
```

Tüm spec çıktıları baseline ile **byte-byte aynı** (lossless garantisi).

---

## Sonuç tablosu

| # | Prompt | baseline (t/s) | spec n=3 (t/s · acc% · acc/round) | spec n=5 (t/s · acc% · acc/round) | En iyi | Hızlanma |
|---|--------|---------------:|----------------------------------:|----------------------------------:|:------:|---------:|
| 1 | "hello" | 50.9 | 53.2 · 59.0% · 1.77 | **58.1** · 50.8% · 2.54 | n=5 | **1.14×** |
| 2 | "What is 2+2? Answer in one word." | 50.6 | **52.6** · 56.5% · 1.70 | 51.0 · 42.0% · 2.10 | n=3 | 1.04× |
| 3 | "Translate to French: The quick brown fox jumps over the lazy dog." | 50.5 | 48.9 · 50.4% · 1.51 | 45.5 · 35.4% · 1.77 | baseline | 0.97× |
| 4 | "List the first 50 prime numbers." | 50.0 | 68.2 · 80.8% · 2.43 | **74.8** · 72.0% · 3.60 | n=5 | **1.50×** |
| 5 | "Count from 1 to 100, one number per line." | 50.2 | **52.6** · 56.5% · 1.69 | 50.0 · 41.0% · 2.05 | n=3 | 1.05× |
| 6 | "Explain how a transistor works in simple terms." | 50.5 | 44.2 · 42.2% · 1.27 | 40.3 · 30.3% · 1.51 | baseline | 0.88× |
| 7 | "Write a Python function to compute the factorial of n." | 50.9 | **53.7** · 59.5% · 1.79 | 51.6 · 46.4% · 2.32 | n=3 | 1.05× |
| 8 | "Write a short story about a robot learning to paint." | 50.4 | 41.5 · 39.4% · 1.18 | 35.9 · 27.8% · 1.39 | baseline | 0.82× |

Baseline ortalaması: **50.4 t/s** (her prompt'ta neredeyse sabit).

---

## Tipe göre özet

| Tip | Örnek | Hızlanma | Notu |
|-----|-------|----------|------|
| **Yapılı/şablonlu liste** | primes | **1.50×** | en yüksek acceptance (80%+), n=5 ideal |
| Çok kısa / soru-cevap | hello, 2+2 | 1.04–1.14× | thinking template yapılı çıktı üretir |
| Sıralı sayma | 1→100 | 1.05× | beklediğimden düşük (her sayı için ayrı argmax) |
| Kod yazımı | python factorial | 1.05× | kod şablonlu ama açıklamalı → orta acc |
| Çeviri (kısa) | French | 0.97× | breakeven civarı |
| Açıklayıcı metin | transistor | 0.88× | yaratıcı → kayıp |
| Yaratıcı düzyazı | robot hikayesi | 0.82× | en çeşitli → en büyük kayıp |

---

## Gözlemler

### 1. avg_accepted/round n_max'i geçemiyor
n=5'te bile çoğu prompt 2-3'te doyuyor → daha büyük n_max boşa overhead. Sadece
yapısal görevde (primes 3.60) yüksek n_max ödüllendi.

### 2. acceptance ≠ hızlanma değil, **acc/round** belirleyici
n=5 acc% her zaman n=3'ten düşük (matematik gereği — daha fazla token daha sık reddedilir),
ama gerçek metrik `accepted/round`. n=5 primes'da 3.60 → büyük kazanç.

### 3. Sweet-spot n_max heuristic
- avg_accepted/round n_max'a yaklaşıyor → **n_max'i artır** (daha kazanılacak)
- avg_accepted/round ≤ n_max/2 → **n_max'i düşür** (boşa draft)
- avg_accepted/round < 1.3 → spec **muhtemelen kaybediyor**, baseline kullan

### 4. Thinking template etkisi
Gemma-4-it gguf'unun chat template'i thinking modunu açıyor (`<|channel>thought`, "Thinking
Process:"). Kısa prompt'larda bile uzun thinking bloğu üretiyor — bu yapılı kısım
acceptance'ı yukarı çekiyor (hello %59, 2+2 %57). Yaratıcı son cevap düşük acceptance.

### 5. Loss patterns (neden kayıp)
- "Write a short story" %39 acc, 1.18 acc/round, 0.82× hız
- 1 token/round ortalamada bile round başına sabit overhead (graph rebuild + N draft +
  verify + tap copy + cluster mask) baseline'dan 1-1.5× pahalı → net kayıp
- Bu **lossless korunarak** olur — çıktı doğru, sadece daha yavaş

---

## Öneri kullanım

| Görev | Komut |
|-------|-------|
| Yapısal liste / tablo / format | `--spec-draft-n-max 5` veya `6` |
| Kısa Q&A / kod / çeviri | `--spec-draft-n-max 3` |
| Uzun yaratıcı metin | baseline (spec kaybeder) veya `--spec-draft-n-max 2` |

İlk çalıştırmada acceptance'a bak (`G4A STATS:` stderr'de), n_max'i ayarla.

---

## Ham komutlar (yeniden üretmek için)

```bash
cd /Users/enes/Desktop/all/less-llama-cpp/llama.cpp
M=/Users/enes/Desktop/all/llms/gemma-4-E2B-it-UD-Q8_K_XL.gguf
D=/Users/enes/Desktop/all/llms/gemma-4-E2B-it-assistant.F16.gguf

# baseline
./build/bin/llama-cli -m $M -p "<prompt>" -n 300 --temp 0 -st -ngl 99

# spec
./build/bin/llama-cli -m $M -md $D \
  --spec-type draft-mtp --spec-draft-n-max <N> \
  -p "<prompt>" -n 300 --temp 0 -st -ngl 99
```
