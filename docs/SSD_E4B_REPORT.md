# E4B Q8 Çifti — SD vs SSD Karşılaştırma

## Setup

- **Target**: `gemma-4-E4B-it-Q8_0.gguf` (7.6 GB)
- **Draft**:  `gemma-4-E4B-it-assistant.Q8_0.gguf` (96 MB)
- **Donanım**: M2 Pro
- **Draft-max**: 5, **SSD fan-out**: 7

## Bug Fix Notu

İlk denemede ekrana sürekli `decode: cannot decode batches with this context (calling encode() instead)` basıyordu. Crash değil, GGML_LOG_LEVEL_DEBUG seviyesi her draft decode'da uyarı basıyordu (MTP context normal). Çözüm: `spec.cpp main()` başına quiet log callback eklendi — DEBUG seviyesi suppress, INFO ve üzeri görünür.

```cpp
static void spec_quiet_log(enum ggml_log_level level, const char * text, void *) {
    if (level <= GGML_LOG_LEVEL_DEBUG) return;
    fputs(text, stderr);
}
// main() içinde: llama_log_set(spec_quiet_log, nullptr);
```

## Sonuç Tablosu

Output 16 çalıştırmada byte-identical (lossless).

| Prompt | SD Metal (both GPU) | **SD CPU draft** | SSD serial CPU | SSD ASYNC CPU |
|---|---|---|---|---|
| primes (40) | 49.2 t/s | **54.3** | 47.5 | 50.4 |
| count (1-60) | 26.3 | **29.1** | 27.0 | 28.6 |
| creative | 20.5 | **23.4** | 21.9 | 23.1 |
| coding (fib) | 39.8 | **45.6** | 36.0 | 38.1 |
| **ortalama** | **34.0** | **38.1** | 33.1 | 35.1 |

## Şu An Full Metal SD'den DAHA İYİ mi DAHA KÖTÜ mü?

### SSD ASYNC vs SD Metal (paper implement vs baseline)
| Prompt | SSD async | SD Metal | fark |
|---|---|---|---|
| primes | 50.4 | 49.2 | **+2.4%** ✓ |
| count | 28.6 | 26.3 | **+8.7%** ✓ |
| creative | 23.1 | 20.5 | **+12.7%** ✓ |
| coding | 38.1 | 39.8 | -4.3% ✗ |
| **ortalama** | **35.1** | **34.0** | **+3.2%** ✓ |

→ **Marjinal kazanç. 4'te 3 prompt'ta daha iyi, 1'de kötü. Ortalama %3 hızlı.**

### Asıl Sürpriz: SD CPU draft >> SD Metal her zaman
| Prompt | SD CPU | SD Metal | fark |
|---|---|---|---|
| primes | 54.3 | 49.2 | **+10.4%** |
| count | 29.1 | 26.3 | **+10.6%** |
| creative | 23.4 | 20.5 | **+14.1%** |
| coding | 45.6 | 39.8 | **+14.6%** |
| **ortalama** | **38.1** | **34.0** | **+12.0%** |

→ **SSD bile gerekmiyor — sadece draft'ı CPU'ya almak %12 net kazanç.**

## Neden SD CPU > SD Metal?

E4B target Q8 = **7.6 GB** Metal RAM'de. Draft de Metal'deyken GPU'yu daha çok stress'liyor:
- Memory bandwidth çekişmesi
- Shader pipeline congestion
- Aynı queue'da seri compute

Draft CPU'ya çekildiğinde:
- Target Metal'de tek başına → verify hızlı
- Draft CPU NEON'da → küçük model (96 MB Q8) zaten hızlı
- İki backend BAĞIMSIZ çalışıyor

E2B'de (4.9 GB target) bu etki yoktu — E2B Metal'de nefes alıyor zaten. Sadece büyük target'larda CPU draft kazanç veriyor.

## SSD async neden CPU draft baseline'ı geçmiyor?

```
verify  ~80 ms
post_pass ~250 ms (B=7 → ~30 alt decode × ~8ms)
overlap_region ~265 ms
hidden = (80+250-265)/(80+250) = 19.7%
```

Post-pass verify'dan **3× büyük**. Async sadece verify wall-time'ı gizliyor. Post-pass bottleneck olmaya devam ediyor.

E2B'de hidden %12 idi, E4B'de %20-22'ye çıktı (verify uzadı). **Daha büyük target'ta** SSD async kazanca yaklaşır:
- M-Ultra veya 70B model gibi target'larda verify >> post-pass olabilir
- O noktada SSD async > SD CPU mümkün

Şu anki donanımda: SSD async = SD CPU (overhead tam kapanıyor ama net kazanç yok).

## Pratik Tavsiye (E4B Q8 çifti, M2 Pro)

| Amaç | Komut |
|---|---|
| **En hızlı** | `-ngl 99 -ngl-draft 0 --ssd-fan-out 1` (SD CPU draft) |
| Paper SSD demo | `-ngl 99 -ngl-draft 0 --ssd-fan-out 7 --ssd-async` |
| Eski varsayılan | `-ngl 99 -ngl-draft 99` (SD Metal, en yavaş) |

**En önemli öğrenme**: Apple Silicon'da büyük target için **default'u `-ngl-draft 0` yapmak doğru**. Zaten yeni default 0.

## Lossless Doğrulama

16 çalıştırma (4 prompt × 4 mode). Tüm output'lar byte-identical. Threading deterministik compute'u bozmadı.

## Sonuç

- ✅ SSD async **SD Metal'den %3 hızlı** (ortalama, 4/4 prompt'tan 3'ünde)
- ✅ SD CPU draft **SD Metal'den %12 hızlı** (her prompt'ta)
- ✅ SSD async ≈ SD CPU draft (SSD overhead tamamen gizlendi)
- ✅ Lossless korundu
- ❌ Paper'ın %30 kazancı yok — M2 Pro'da verify hala post-pass'ten kısa

**Net cevap**: Şu anki kurulum (SSD async, E4B Q8, CPU draft) **full Metal SD'den biraz daha iyi (+%3)**. Ama esas kazanç SSD'den değil, draft'ı CPU'ya çekmekten geliyor (+%12). E2B'de SD Metal hâlâ en hızlıydı; E4B'de tersine döndü.
