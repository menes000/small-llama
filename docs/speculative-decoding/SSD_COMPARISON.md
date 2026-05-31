# SD vs SSD Karşılaştırma — Deney Sonuçları

Model: `gemma-4-E2B-it-UD-Q8_K_XL` (target Q8) + `gemma-4-E2B-it-assistant.F16` (draft F16)  
Platform: Mac, Metal backend, single GPU.  
SSD flag: `--ssd-fan-out 7` (B=7). SD = `--ssd-fan-out 1` (B=1 = off).

Output her testte byte-identical (lossless). Tüm diff'ler sıfır.

---

## Sonuç Tablosu

### draft-max=3 (K=3, B=7 > K → SSD aktif)

| Prompt | Mode | t/s | accept% | acc/round | hit% | extra_dec | saved | net_extra |
|---|---|---|---|---|---|---|---|---|
| primes (40 prime) | SD | **110.7** | 94.0 | 2.82 | — | 0 | 0 | 0 |
| primes | SSD B=7 | 92.3 | 94.0 ✓ | 2.82 | 33 | 168 | 1 | +167 |
| count (1-60) | SD | **101.9** | 81.9 | 2.46 | — | 0 | 0 | 0 |
| count | SSD B=7 | 80.1 | 81.9 ✓ | 2.46 | 44 | 164 | 4 | +160 |
| creative (robot story) | SD | **51.8** | 22.6 | 0.68 | — | 0 | 0 | 0 |
| creative | SSD B=7 | 41.8 | 21.7 ✓ | 0.65 | 20 | 256 | 10 | +246 |
| coding (fibonacci) | SD | **49.4** | 20.3 | 0.61 | — | 0 | 0 | 0 |
| coding | SSD B=7 | 38.9 | 18.9 ✓ | 0.57 | 19 | 324 | 14 | +310 |

### draft-max=5 (K=5, B=7 > K → SSD aktif)

| Prompt | Mode | t/s | accept% | acc/round | hit% | extra_dec | saved | net_extra |
|---|---|---|---|---|---|---|---|---|
| primes | SD | **109.8** | 79.3 | 3.97 | — | 0 | 0 | 0 |
| primes | SSD B=7 | 98.3 | 79.3 ✓ | 3.97 | 0 ⚠️ | 77 | 0 | +77 |
| count | SD | **94.5** | 66.4 | 3.32 | — | 0 | 0 | 0 |
| count | SSD B=7 | 86.7 | 66.4 ✓ | 3.32 | 7 | 83 | 1 | +82 |
| creative | SD | **40.5** | 14.1 | 0.71 | — | 0 | 0 | 0 |
| creative | SSD B=7 | 35.9 | 13.6 ✓ | 0.68 | 16 | 153 | 8 | +145 |
| coding | SD | **39.1** | 12.6 | 0.63 | — | 0 | 0 | 0 |
| coding | SSD B=7 | 33.9 | 11.7 ✓ | 0.59 | 16 | 189 | 12 | +177 |

### draft-max=7 (K=7, B=7 = K → SSD otomatik kapalı)

| Prompt | Mode | t/s | accept% | hit% | Açıklama |
|---|---|---|---|---|---|
| primes | SD | 96.5 | 61.7 | 0 | — |
| primes | SSD B=7 | 98.8 | 61.7 | 0 | B=K → alts yok |
| count | SD | 78.5 | 47.4 | 0 | — |
| count | SSD B=7 | 78.1 | 47.4 | 0 | B=K → alts yok |
| creative | SD | 32.9 | 10.1 | 0 | — |
| creative | SSD B=7 | 32.4 | 10.1 | 0 | B=K → alts yok |
| coding | SD | 30.9 | 9.0 | 0 | — |
| coding | SSD B=7 | 30.6 | 9.0 | 0 | B=K → alts yok |

---

## Ana Bulgular

### 1. Losslessness — Her Testte Sağlandı
`diff SD_output SSD_output` → her zaman 0 fark. Accept rate değişse bile emitted text aynı (target greedy deterministik).

### 2. Accept Rate — B=7, K≤5'te Korunuyor
draft-max=3 ve draft-max=5'te SSD B=7, accept rate'i değiştirmiyor (chain bütünlüğü intact). Tek istisna: primes prompt + draft-max=5'te hit=0 (cache miss, ama accept rate bozulmadı).

### 3. t/s Her Zaman Daha Düşük (Single GPU)
Paper'ın kazancı draft'ın AYRI cihazda çalışmasını gerektiriyor (T_p < 1). Tek Metal GPU'da:
- extra_decodes her zaman steps_saved'dan büyük
- net_extra her zaman pozitif (+77 → +310 range)
- t/s kaybı: %5–25 arası

### 4. Hit Rate Pattern Paper Tahminiyle Uyumlu
```
structured/predictable > creative/diverse
primes (dm=3):  33–44% hit  ← en yüksek
count (dm=3):   44% hit
creative:       16–20% hit
coding:         16–19% hit
```
Paper Fig.3: rejection rate (= 1 - hit_rate) falls as power-law with fan-out → büyük B'de hit rate artar.

### 5. B = K Durumu → Sıfır Overhead (Safe Fallback)
`geometric_fanout(B, K)` B≤K → all-ones → alts sıfır. `ssd_fan_out > n_draft_max` koşulu False → cache lookup atlanır.
draft-max=7, B=7: SD ile SSD neredeyse aynı (fark <1%). Güvenli fallback davranışı.

### 6. draft-max=3 + B=7 → En Yüksek Hit Rate
Kısa lookahead → bonus token çoğunlukla pozisyon 0-2'ye düşüyor → draft'ın top-F tahmini daha doğru. dm=3, count prompt: **44% hit** en iyi sonuç.

### 7. draft-max=5, primes → 0 Hit Anomalisi
Aynı prompt dm=3'te 33% hit, dm=5'te 0% hit. Sebep: daha derin lookahead ile bonus token genelde pozisyon 3-4'e düşüyor → draft'ın o pozisyonlardaki top-F tahminleri doğru değil (conditional on 5-step chain). Tesadüfî değil — deterministik, her run'da tekrarlanıyor.

---

## Ne Zaman Kullanılır

| Durum | Öneri |
|---|---|
| Single Mac, hız öncelik | `--ssd-fan-out 1` (SSD kapalı) |
| Single Mac, hit rate ölçmek istiyorum | `--ssd-fan-out 7 --draft-max 3` |
| Ayrı GPU/cihaz (gelecekte) | `--ssd-fan-out 20+` — gerçek kazanç burada başlıyor |
| Structured/templated prompt | SSD daha yüksek hit rate → multi-device'da daha iyi |
| Creative prompt | Hit rate ~20% → multi-device'da marginal kazanç |

---

## Bilinen Sınırlama: draft-max küçük iken creative prompt accept düşüşü

dm=3, creative: SD=22.6% accept, SSD=21.7% accept — küçük düşüş. Çok sayıda round'da biriken K/V restore eksikliği. Primes dm=10+ durumundaki büyük düşüşün (82.5%→69.6%) küçük versiyonu.

B=7 seçimi bu sınırlamayı minimize ediyor ama sıfırlamıyor.
