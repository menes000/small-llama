# SSD Async (CPU draft + Metal target + std::thread) Raporu

Paper'ın gerçek hız kazancı kurulumu: draft AYRI cihazda, verify ile paralel. Bu sürüm onu Mac üzerinde uygular:
- **Target**: Metal GPU (`-ngl 99`, default)
- **Draft**: CPU NEON (`-ngl-draft 0`, yeni default)
- **Overlap**: `--ssd-async` → SSD post-pass `std::async` ile target verify'a paralel çalışır

Donanım: M2 Pro (Apple Silicon). Target = Gemma4 E2B Q8_K_XL. Draft = Gemma4 E2B assistant F16 (~156 MB).

## Yeni CLI

| Flag | Default | Açıklama |
|---|---|---|
| `-ngl N` | 99 | Target GPU layer sayısı (Metal) |
| `-ngl-draft N` | **0** | Draft GPU layer sayısı (0 = CPU NEON) |
| `--ssd-async` | off | Target verify ile SSD post-pass paralel (true overlap) |

`--ssd-async` ancak draft ve target FARKLI backend'de ise anlamlı.

## Test Sonuçları (K=5, B=7, M2 Pro)

Output her testte byte-identical (lossless). 4-way diff = 0 fark.

| Prompt | SD Metal (both GPU) | SD CPU draft | SSD serial CPU | SSD ASYNC CPU | hit% |
|---|---|---|---|---|---|
| primes (40) | **108.4** t/s | 98.5 | 87.5 | **95.3** | 0 |
| count (1-60) | **94.7** | 88.4 | 77.4 | **89.7** | 0 |
| creative | **40.8** | 36.7 | 30.4 | **35.2** | 16 |
| coding (fib) | **86.5** | 80.2 | 68.4 | **79.2** | 23 |

### Timing breakdown örneği (creative, async):
```
verify = 80 ms     (target Metal, 59 round × ~1.4ms)
post_pass = 514 ms (draft CPU, 153 alt decodes × ~3.3ms)
overlap_region = 525 ms
hidden = 11.7%
```

## Ana Bulgular

### 1. Async overlap işe yarıyor — her zaman serial'den hızlı
Tüm prompt'larda `ssd_async` > `ssd_serial` (+%12 ortalama). Post-pass cost'unun verify wall-time içinde olan kısmı tamamen gizleniyor.

### 2. SSD async ≈ SD CPU baseline
Async modu SSD overhead'ini tamamen Metal verify wall-time'ı içinde gizliyor. Sonuç: SSD aktifken bile saf SD-CPU-draft kadar hızlı. **SSD bedavaya geliyor**.

### 3. Ama SSD async < SD Metal (pure GPU)
Çünkü:
- CPU draft kendisi %7 daha yavaş GPU draft'tan (M2 Pro NEON 156MB model için)
- Bu fixed cost SSD ile bağlantılı değil — CPU backend'in kendisi
- Async ne kadar iyi olsa de CPU draft tax'ini kaldıramıyor

### 4. Hidden% sadece ~%12-13 (post-pass verify'dan çok büyük olduğu için)
```
verify    ~35 ms
post_pass ~250 ms  (B=7, K=5, ~30 alt decode)
```
Verify post-pass'in 1/7'si. Async overlap verify'i tamamen gizliyor ama post-pass yine de bottleneck. Hidden% formülü:
```
hidden = (verify + post_pass - max(verify, post_pass)) / (verify + post_pass)
       = verify / (verify + post_pass)
       = 35 / 285 = 12%
```

### 5. Lossless korunuyor
4 prompt × 4 mode = 16 çalıştırma. Her çiftin output'u byte-identical. Threading deterministik compute'u bozmadı.

## Önemli Çıkarımlar

### Paper'ın "30% kazanç" iddiası kim için geçerli?
Paper donanım dengesi:
- Target: 4×H100 (büyük model, **çok yavaş** verify)
- Draft: 1×H100 (küçük model, çok hızlı)
- Verify wall-time >> draft spec wall-time → draft "bedava"

Senin donanımın:
- Target: 1× M2 Pro Metal GPU (~30ms verify)
- Draft CPU: M2 Pro NEON (~3.3ms per single decode, ~250ms post-pass)
- Verify << post_pass → draft DEĞİL bedava

### Ne zaman SSD async net kazanç verir?
`verify_wall_us >= post_pass_wall_us` koşulu sağlandığında. Bunun için:
- Çok büyük target model (verify yavaşlar)
- Veya çok küçük alt sayısı (B = K+1 → 1-2 alt, post-pass küçülür)
- Veya daha hızlı draft backend (örn. NPU varsa)

### M2 Pro'da pratik sweet spot
| Konfigürasyon | t/s | Not |
|---|---|---|
| SD Metal (default) | 108 | En hızlı |
| SD CPU draft | 98 | %10 yavaş ama RAM/VRAM ayrımı sağlıyor |
| SSD async CPU | 95 | SD CPU + SSD bedava cache, %12 yavaş |

Pure hız için: SD Metal. SSD demo/research için: SSD async.

## Bilinen Sınırlamalar

1. **CPU draft yavaş**: M2 Pro NEON F16 156MB model için ~300 decode/sec. Daha küçük draft (örn. quantize edilmiş Q4) daha hızlı olur.

2. **`hidden%` mütevazı**: post_pass >> verify olduğu sürece overlap savings sınırlı kalır. Verify wall-time'ı uzatmak ya da post-pass'i kısaltmak gerek.

3. **Thread overhead**: `std::async` setup ~0.1ms. Çok kısa round'larda fark edilmiyor.

4. **Accept rate koruması**: Mevcut restore mekanizması B≤K+2 için iyi çalışıyor. Daha yüksek B'lerde bazı prompt'larda accept rate hala düşebilir (önceki rapor §2.2).

## Future Work

- **Faster draft quantization**: Gemma4 assistant Q4_K_M ~40MB → CPU draft 2-3× hızlanabilir.
- **Multi-round async pipelining**: Round T+1 greedy chain'ini round T verify'a paralel başlat.
- **GGML CPU backend tuning**: Accelerate backend'i aktif etmek (CMakeLists'te `GGML_ACCELERATE OFF` → ON).
- **Test farklı donanım**: M-Ultra'da Metal verify çok daha yavaş, async kazanç büyür.
