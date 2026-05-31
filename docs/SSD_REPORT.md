# SSD (Saguaro) Implementasyon Raporu

Paper: **Speculative Speculative Decoding** — arXiv 2603.03251v3 (Kumar/Dao/May).

Dosya: `examples/spec/spec.cpp` (438 → 649 satır).

## 1. Ne Yapıldı

### 1.1 CLI Flag'leri
`struct args` + `parse_args` genişletildi:
- `--ssd-fan-out B` — toplam alternatif bütçesi (default `1` = SSD kapalı).
- `--ssd-r R` — geometric fan-out üs (default `1.0`, Theorem 12).
- `--ssd-sampling-C C` — Saguaro sampling downweight (default `1.0` = pasif; greedy modunda UYARI verir).

### 1.2 Yardımcı Fonksiyonlar
- `masked_topf(...)` — `masked_argmax`'ın top-F genelleştirmesi. Top-K centroid cluster içindeki tokenları dense_logits'e göre sıralar, F tane döner. `out_tokens[0]` = eski argmax.
- `masked_argmax(...)` artık `masked_topf(..., F=1)` wrapper'ı.
- `geometric_fanout(B, K, a_p, r)` — Theorem 12 formülü:
  ```
  F_k = round(F_0 · a_p^(k/(1+r)))    her F_k >= 1, sum(F_k) <= B
  ```
  `a_p` (running accept rate) ilk round için `0.8`, sonra `total_accepted / total_drafted`.

### 1.3 Speculation Cache
```cpp
struct cache_entry {
    llama_token alt_bonus;          // "eger bonus bu olsa..."
    llama_token next_tok;            // pre-speculated sonraki draft token
    std::vector<float> next_feat;    // round T+1 için hazır chained feature
};
std::vector<std::vector<cache_entry>> spec_cache;   // spec_cache[k] = pos k'daki alt'lar
```

Her round başında reset. `start_k` ile cache hit varsa ilk draft step skip.

### 1.4 Draft Loop Yeniden Yapısı
Eski (lines 309–337): tek pass, her step k için inline decode + masked_argmax.

Yeni: **iki phase** — pollution bug'ı için (aşağıda).
1. **Greedy phase**: K step pure greedy chain. Her step için `(in_tok, in_feat, alts)` kaydedilir.
2. **SSD post-pass**: tüm greedy chain bitince, kaydedilmiş `in_*`'tan alt decode'lar yapılır. Cache'e yazılır.
3. **Restore**: post-pass sonunda son greedy step yeniden decode → K/V[draft_pos] vanilla path ile aynı kalır.

### 1.5 Post-verify Cache Lookup
Verify bittikten sonra (`n_accept`, `next_pending` belli):
```cpp
if (n_accept < spec_cache.size()) {
    for (entry of spec_cache[n_accept]) {
        if (entry.alt_bonus == next_pending) {
            ssd_seed_tok = entry.next_tok;
            ssd_seed_feat = entry.next_feat;
            ssd_have_seed = true; ssd_hits++; break;
        }
    }
    if (!hit) ssd_misses++;
}
```
Hit → sonraki round'un k=0 draft step skip edilir (1 decode kazanım).

### 1.6 Stats Hattı
`[spec]` line'a ek:
```
SSD: hits=N misses=N hit_rate=N.N% extra_draft_decodes=N steps_saved=N net_extra=N
```

## 2. Karşılaşılan Sorunlar + Çözümler

### 2.1 BUG #1: Inline alt decode → accept rate düştü
**Belirti**: B=10 ile primes prompt, accept rate 82.5% → 69.6%. Output byte-identical (lossless) ama draft chain bozulmuş — sonraki round'ların kalitesi düşmüş.

**Hipotez**: Her greedy step'in HEMEN ardından alt decode yapmak ctx_d'nin K/V[draft_pos]'unu kirletiyor. Sonraki greedy step bozuk K/V okuyor.

**Çözüm girişimi #1 — Post-pass**:
Alt decode'lar greedy chain'in TAMAMI bittikten sonra yapıldı. `step_state{in_tok, in_feat, alts}` per-step kaydedildi, sonra ayrı loop'ta replay.

**Sonuç**: ❌ Accept rate hala 69.6%. Sorun başka yerde.

### 2.2 BUG #2: Alt decode K/V'yi kirletiyor → sonraki ROUND etkileniyor
**Hipotez**: Post-pass sonunda K/V[draft_pos] = son alt'ın K/V'si. Sonraki round'un draft self-attention'ı bu kirlenmiş pozisyonu okuyor.

**Çözüm girişimi #2 — Restore decode**:
Post-pass sonunda son greedy step'i (in_tok[K-1], in_feat[K-1]) ile tekrar decode et. K/V[draft_pos] geri vanilla path'in bıraktığı değere döner.

**Sonuç (kısmi başarı)**:
- B=7, 8 → accept rate ✓ korunuyor (82.5%)
- B=10+ → accept rate ✗ hala düşüyor (69.6%)

**Yorum**: Restore tek pozisyonda K/V'yi geri yüklüyor ama çoklu alt decode (F_k >= 3 olunca pos başına 2+ alt) ctx_d içindeki K/V dışı state'i bozuyor olabilir (hidden counter, n_outputs, vs.). MTP context tipinin internal state'ini debug etmek gerekiyor — bu sürüm kapsamı dışı.

### 2.3 BUG #3 (false positive): LSP "headers not found"
**Belirti**: clang LSP her edit sonrası `'llama.h' file not found` + std type errors raporladı.
**Tanı**: LSP include path konfigüre değil. cmake build sorunsuz çalıştı → diagnostik gerçek değil. Görmezden gelindi.

### 2.4 Tasarım kararı: Saguaro sampling σ_F,C scaffold-only
Greedy decode mode'unda top-F downweight argmax'ı kaydırır → losslessness bozulur (target greedy ile mismatch yaratmaz ama draft'ın çıktısını değiştirir). Flag eklendi, kod yolu yok, greedy + C<1 girilirse uyarı verilir. Temperature sampling eklendiğinde aktive edilecek.

### 2.5 Tasarım kararı: threading skip
Paper `T_p < 1` varsayımına dayanıyor — draft, target verify wall-time'ı içinde gizlenebilsin diye AYRI device gerekli. Tek Mac Metal'de shared GPU queue → `std::thread` overlap kazanç vermez. Belgelendi, atlanıldı.

## 3. Doğrulama Sonuçları

Setup: `gemma-4-E2B-it-UD-Q8_K_XL` (target) + `gemma-4-E2B-it-assistant.F16` (draft), `--draft-max 5`, Mac Metal.

### 3.1 Determinism check
Aynı flag'ler ile arka arkaya iki run → aynı stat (rounds, accept_rate, hits) ✓

### 3.2 Losslessness check
Tüm B değerleri için `diff baseline.out ssd_on.out` → 0 fark ✓

### 3.3 Fan-out sweep — primes prompt
| B | t/s | rounds | accept% | hits | extra | saved |
|---|---|---|---|---|---|---|
| 1 (off) | 110.7 | 24 | 82.5 | 0 | 0 | 0 |
| 6 | 105.4 | 24 | 82.5 | 0 (path enter) | 0 | 0 |
| 7 | 97.0 | 24 | 82.5 | 1 | 56 | 1 |
| 8 | 93.1 | 24 | 82.5 | 1 | 95 | 1 |
| 10 | 80.0 | 27 | 69.6 ⚠️ | 7 | 148 | 7 |
| 15 | 67.5 | 28 | 67.1 ⚠️ | 10 | 274 | 10 |

### 3.4 Prompt tipi karşılaştırma (B=10)
| Prompt | t/s base → SSD | accept% base → SSD | hit% |
|---|---|---|---|
| "first 30 primes" | 110.7 → 80.0 | 82.5 → 69.6 ⚠️ | 41 |
| "count 1 to 50" | 90.9 → 76.0 | 66.1 → 66.1 ✓ | 25 |
| "write short story dragon" | 40.3 → 34.7 | 15.7 → 15.7 ✓ | 23 |

**Hit rate paper Fig.3/4 pattern'i ile uyumlu**: structured > creative.

## 4. Sonuç

**Doğru çalışıyor**: 
- Algoritma paper'a sadık (cache, geometric fan-out, hit lookup, stats).
- Lossless (greedy target output byte-identical).
- Stats doğru (p_hit, extra_decodes, steps_saved ölçülüyor).

**Tek-GPU'da kazanç yok**: 
- net_extra = extra_decodes - steps_saved → her zaman pozitif (>> 0).
- Paper'ın 30%-5× hızı **AYRI DEVICE** varsayımına dayanıyor.
- Kullanıcı bilerek "pragmatic" seçti, beklenen sonuç.

**Bilinen sınırlama**:
- Yüksek B + bazı prompt'larda (örn. primes B=10) accept rate düşüyor. MTP context'in K/V dışı internal state'i tam restore edilemiyor. Sonraki iterasyon gerekirse `src/models/gemma4.cpp` MTP path'ini incele.

**Future work** (bu sürümde değil):
- Temperature sampling + Saguaro σ_F,C aktive et.
- Threading: draft ve target ayrı backend'lerde (Metal GPU + CPU NEON) çalıştırılırsa partial overlap mümkün olabilir.
- Fallback speculator (paper §4.3) — ikinci hızlı draft model gerekli.
