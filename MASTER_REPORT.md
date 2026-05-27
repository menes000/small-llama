# `only-needed-files` Master Raporu — Gemma 4 Spec Decoding + Chat + Tool Use

Tarih: 2026-05-26
Kapsam: Tüm oturum boyunca yapılan iş — bug avı, port, optimizasyon, sınırlamalar, gelecek roadmap.

---

## 0. TL;DR

**Yapılanlar:**
- Full tree llama.cpp'de Gemma-4 speculative decoding'i bozuyordu (acceptance %0, çöp token, Metal crash). 3 root-cause bug bulundu ve fix'lendi → SD çalışır hale geldi (E4B'de 1.69×, E2B yapısal görevde 1.5×).
- Tüm SD altyapısı `only-needed-files` minimal tree'sine port edildi (~1100 satır).
- `only-needed-files` için yeni `llama-spec` örneği yazıldı (438 satır).
- Gemma-4 chat template `only-needed-files`'a eklendi (`llama_chat_apply_template` artık -1 dönmüyor).
- Yeni `llama-chat` örneği: multi-turn REPL + sandbox'lı `read_file`/`list_dir` tool'ları + opsiyonel SD entegrasyonu + per-turn istatistikler.

**Performans:**
- E2B baseline: ~51 t/s (Metal)
- E2B SD (yapısal görev, primes): **80 t/s** spec.cpp / **57 t/s** chat.cpp (acc %79)
- E4B-it Q8 baseline: ~30 t/s
- E4B-it SD (counting): **49.8 t/s = 1.69×** (acc %85)

**Bilinen sınır:** %85 acceptance bile 3× kazandırmıyor çünkü round-başı sabit overhead (graph rebuild, scheduler sync) E2B gibi küçük modelin baseline'ını aşmıyor. 3×'e ulaşmak için graph-rebuild eliminasyonu, adaptive draft length, daha büyük target gerekir.

---

## 1. Proje bağlamı

`only-needed-files` (`/Users/enes/Desktop/all/less-llama-cpp/only-needed-files/`) kullanıcının kişisel
projesi: **llama.cpp'yi mümkün olan en az satırla aynı performansta çalıştırmak**. Full tree
~500 MB, only-needed-files ~10 MB. Gemma-4 mimarisine odaklı.

**Hedef modeller:**
- `gemma-4-E2B-it-UD-Q8_K_XL.gguf` — 2B param target (Q8, ~2.8 GB)
- `gemma-4-E2B-it-assistant.F16.gguf` — 78M MTP/EAGLE draft head (F16, 174 MB)
- `gemma-4-E4B-it-Q8_0.gguf` — 4B target (Q8, ~8.2 GB)
- `gemma-4-E4B-it-assistant.F16.gguf` — eşleşen E4B draft

Mimari özet: `gemma4` arch (target) + `gemma4_assistant` arch (draft). Assistant kendi K/V'sini
saklamaz — her layer'ında target modelin paylaşılan K/V'sine cross-attention yapar. Output:
post_projection feature (chain için) + centroid masked-embedding logit head.

---

## 2. Aşama 1 — Full tree'de SD bug avı

### Başlangıç durumu

```
G4A draft[step] i=0 id=178830('Zag') p=1.0
G4A   TARGET id_last=100 logit=-16 dense_rank=4096 (MASKED OUT)
[ Prompt: 142 t/s | Generation: 6.5 t/s ]   ← bozuk
```

Pipeline kuruluydu ama %0 accept, draft tamamen çöp token öneriyordu (`Zag`, `HttpServlet`,
`아멘`, emoji), generation 6.5 t/s, Metal crash (CPU-only çalıştı).

### Metodoloji: tahmin değil, ölç

HF transformers `gemma4_assistant`'ını referans olarak kurdum (`/tmp/g4a_venv`, scripts'ler
`/tmp/g4a_*.py`). Sayısal karşılaştırma:

| Karşılaştırma | Sonuç | Çıkarım |
|---------------|-------|---------|
| llama draft forward vs HF (aynı concat ile) | cosine 1.0 | Forward matematiği doğru |
| HF mekanizması (normal template) | 8/8 acceptance | Assistant sağlam |
| HF mekanizması (thinking template, llama'nın 21 token'ı) | 6/8 acceptance | Mekanizma + assistant tamam |
| llama dump → HF assistant | rank 4096 (mask) | llama input değerleri yanlış |

İzolasyon: **llama'nın target forward'ı HF'den farklı değerler üretiyor.** Bileşen-bileşen
cosine karşılaştırması:

| Bileşen | cos | Magnitude | Verdict |
|---------|-----|-----------|---------|
| `embed` (scaled HF) vs llama | 1.0 | HF ratio 39.2 | **scale eksik** |
| `embed` (raw) vs llama | 1.0 | ratio 1.0 | Raw doğru, scale yok |
| `hidden` post-norm | 0.9997 | ratio 1.004 | ✅ |
| `k_full` | 0.9998 | ratio 1.0 | ✅ |
| `v_full` | **-0.02** | ratio 0.63 | ✗✗ tap çöp |
| `k_swa` | **0.06** | ratio 0.30 | ✗✗ tap çöp |
| `v_swa` | **0.43** | ratio 2.27 | ✗ tap çöp |

`k_full` doğru ama aynı layer'ın `v_full`'ü ve SWA K/V çöp → tap host-copy bug'ı.

### 3 root-cause bug

#### Bug 1: `ggml_set_output` eksik (ROOT CAUSE + Metal crash)

`src/models/gemma4.cpp`'de hidden tap `ggml_set_output()` çağırıyordu ama 4 K/V tap tensoru
çağırmıyordu (yalnız `ggml_build_forward_expand`). ggml-alloc scheduler bu intermediate
tensor'ların buffer'larını sonraki op'lar için reuse ediyor → host async tap **bayat veri**
okuyor. k_full şans eseri hayatta kalıyor, V/SWA üzerine yazılıyor.

**Aynı bug Metal crash'inin de sebebi:** `GGML_ASSERT(buf_dst)` — okunabilir Metal buffer'ı
olmayan tensor.

**Fix:**
```cpp
if (kv_tap_k_full) { ggml_set_output(kv_tap_k_full); ggml_build_forward_expand(gf, kv_tap_k_full); }
if (kv_tap_v_full) { ggml_set_output(kv_tap_v_full); ggml_build_forward_expand(gf, kv_tap_v_full); }
if (kv_tap_k_swa)  { ggml_set_output(kv_tap_k_swa);  ggml_build_forward_expand(gf, kv_tap_k_swa);  }
if (kv_tap_v_swa)  { ggml_set_output(kv_tap_v_swa);  ggml_build_forward_expand(gf, kv_tap_v_swa);  }
```

#### Bug 2: Target embed scale eksik

`llama_model_get_token_embd` ham (unscaled) embedding satırı döner. HF candidate generator
`Gemma4TextScaledWordEmbedding` kullanır — `×√hidden` (E2B için ×39.19).

**Fix:** `common/speculative.cpp` draft loop'unda concat oluştururken `tmp_embd[j] *= √n_embd_backbone`.

#### Bug 3: Centroid masked-embedding head implement edilmemiş

Gemma-4 assistant'ın logit head'i dense lm_head değil, `Gemma4AssistantMaskedEmbedder`:
1. Centroid logits = `mtp_centroids @ hidden` ([2048])
2. Top-32 cluster seç
3. O cluster'ların token'larıyla kısıtlı dense logit'lerden argmax

llama dense lm_head kullanıyordu → garbage. Math equivalent: dense logits restricted to
top-k cluster'lar.

**Fix:**
- Graph'a `centroid_logits` çıktısı ekle (feature ile concat edilerek embeddings buffer'ına)
- Host'ta `token_to_cluster[canon_id] = cluster` haritası kur
- `masked_argmax(dense_logits, centroid_logits, token_to_cluster, top_k)` ile argmax

### Bug 1 sonrası sonuçlar (full tree)

| Test | Önce | Sonra |
|------|------|-------|
| Acceptance | 0% | %50-85 (görev tipine göre) |
| E2B baseline Metal | 51 t/s | 51 t/s |
| E2B SD Metal yapısal | 6.5 t/s (bozuk) | **80 t/s** (acc %78) |
| E4B SD Metal counting | (E4B-Uncensored mismatch %0) | **49.8 t/s = 1.69×** |
| Metal crash | crash | ✅ çözüldü |

---

## 3. Aşama 2 — `only-needed-files`'a SD portu

### Hedef

Full tree'deki SD altyapısı `common/speculative.cpp` (~5000 satır + jinja/sampling deps)
gerektiriyordu. only-needed-files'a minimal port: ~1100 satır.

### Yapılan

**Shared dosyalara additive eklentiler (drift değil, sadece SD):**

| Dosya | Δ satır | Ne |
|-------|---------|-----|
| `src/llama-arch.h` | +9 | LLM_ARCH_GEMMA4_ASSISTANT enum + 4 KV key + 4 tensor enum |
| `src/llama-arch.cpp` | +14 | name, KV key, tensor name + tensor_info dispatch |
| `src/llama-hparams.h` | +6 | 4 yeni alan (n_embd_backbone, n_centroids, top_k, use_ordered) |
| `src/llama-hparams.cpp` | +6 | n_embd_inp() override |
| `src/llama-cparams.h` | +1 | assistant_kv_tap flag |
| `src/llama-ext.h` | +37 | 6 yeni API (set/get tap, accessors) |
| `src/llama-graph.h` | +63 | assistant_shared_kv + input + result tensors |
| `src/llama-graph.cpp` | +70 | build_inp_assistant_kv + set_input |
| `src/llama-context.h` | +36 | tap host buffer + APIs |
| `src/llama-context.cpp` | +112 | tap copy in decode + APIs + MTP ctx relax |
| `src/llama-model.cpp` | +50 | factory case + 4 accessor + nullptr memory |
| `src/models/models.h` | +30 | llama_model_gemma4_assistant class decl |
| `src/models/gemma4.cpp` | +50 | K/V + hidden tap (ggml_set_output ile) |

**Yeni dosyalar:**

| Dosya | Satır | Ne |
|-------|------:|-----|
| `src/models/gemma4_assistant.cpp` | 240 | Assistant arch implementation (full tree'den verbatim) |
| `examples/spec/spec.cpp` | 438 | Minimal SD driver (common dep yok) |
| `examples/spec/CMakeLists.txt` | 5 | Build glue |

**Toplam:** ~1100 satır yeni C++. Full tree'nin SD katmanı ~5000 + jinja 5800 = ~10800 satır;
port ~%10'u.

### Karşılaşılan zorluklar (port sırasında)

- **Diff'lerin temizliği:** `llama-model.cpp`'de 303 satır diff vardı ama %80'i full-tree'deki
  diğer arch'ların factory case'leriydi (drift). Sadece gemma4_assistant ile ilgili ~50 satır
  ekledim. Aynısı `models.h`'da: 1887 satır diff vardı ama only-needed-files'ın 31 satırı sadece
  gemma4 class'ı içeriyor; sadece 30 satır assistant class decl ekledim.
- **CMake GLOB cache'i:** Yeni `gemma4_assistant.cpp` dosyası eklenince ilk build link error
  verdi. `cmake ..` ile yeniden config gerekti.
- **Speculative loop'un common-bağımsız hali:** `common/speculative.cpp`'nin
  `common_speculative_state_draft_gemma4_assistant` struct'ı 448 satırdı (multi-seq abstraction,
  common_sampler dep, staged accept'i, env toggle'lar). Bunları çıkararak ~250 satırlık tek-seq
  greedy çekirdek loop'u inline ettim.

### KV cache rollback bug'ı (port'ta yakalanan)

Spec.cpp'nin ilk versiyonu rejected draft pozisyonlarını target'ın KV cache'inden silmiyordu →
ikinci round'da pozisyon tutarsızlığı (X=41, Y=41 — Y=X+1 olmalı). Fix:

```cpp
if (batch_t.n_tokens > n_keep) {
    llama_memory_seq_rm(llama_get_memory(ctx_t), 0, acc_nkv + n_keep, -1);
}
```

---

## 4. Aşama 3 — Chat + Tool use + SD entegrasyonu

### `llama-chat` (yeni)

Multi-turn REPL + sandbox'lı tool dispatch + opsiyonel SD.

**Args:**
| Flag | Default | Etki |
|------|---------|------|
| `-m` | (zorunlu) | target model |
| `-md` | (yok) | varsa SD aç |
| `--draft-max` | 3 | round başına draft token |
| `--no-sd` | (kapalı) | -md verilse bile SD'yi kapat |
| `--thinking` | (kapalı) | reasoning trace aç |
| `--no-tools` | (açık) | tool tanımlarını kaldır |
| `--root` | `$HOME` | tool sandbox kökü |

**Tools (read-only, realpath sandbox):**
- `read_file(path)` — max 16 KiB
- `list_dir(path)` — max 200 entry

**Per-turn stats (stderr):**
```
[stats] 130 tokens in 8.12s = 16.0 t/s | rounds=26 drafted=130 accepted=104 acc_rate=80.0%
```

**Gemma-4 chat template:**

`src/llama-chat.{h,cpp}`'ye `LLM_CHAT_TEMPLATE_GEMMA_4` enum + detect branch (`<|tool_call>`
veya `<|turn>` substring) + apply branch eklendi. Apply branch turn delimiters'ları üretiyor
(`<bos>`, `<|turn>{role}\n…<turn|>\n`), tool/thinking marker'larını caller (chat.cpp)
system content'ine gömüyor.

**Yeni dosya/satır:**
| Dosya | Satır |
|-------|------:|
| `src/llama-chat.{h,cpp}` (additive) | +25 |
| `examples/chat/chat.cpp` | 631 (init 389 + SD eklentisi 242) |
| `examples/chat/CMakeLists.txt` | 5 |

### Karşılaşılan zorluklar

- **`llama_chat_apply_template` C API tools/thinking taşımıyor** → caller system message content'ine
  embed eden tasarıma geçtim, ayrı `_ex` API eklemeye gerek kalmadı.
- **`std::regex` `[^]` desteklemiyor** → `.*?` ile değiştirdim.
- **`realpath` symlink takip** → macOS'ta `/tmp` → `/private/tmp`. `realpath_s(--root)` ile
  prefix kontrolü.
- **Tool-call durma koşulu** → `<tool_call|>` substring'i `assistant_text` içinde aranıyor;
  bulununca break + dispatch.
- **Multi-turn KV cache yönetimi (string-substr tail diff)** — tutarlılık için **raw push**
  (thinking strip etmiyorum) zorunlu oldu. İlk versiyon thinking'i strip ediyordu → next turn
  `[chat] empty tail tokenization` patladı. Fix: gemma jinja'sında `<|channel>` blokları zaten
  context'te yaşamak için tasarlanmış.
- **`parse_special=true`** tokenize/detokenize her yerde zorunlu (tool token'ları tek-token
  olsun diye).
- **Chat'te SD entegrasyonu — KV cache invariant'ları:** acc_nkv == kv_pos (target seq 0 size)
  her zaman tutmalı. Verify sonrası rollback + n_keep kadar ilerlet. Turn-arası bu state
  korunarak incremental tail prefill çalışıyor.
- **Multi-chunk prefill'de tap reset (sonradan bulundu):** Tail uzunsa `llama_decode`
  birden fazla chunk halinde çağrılır. Her çağrıda target'ın tap buffer'ı (`n_tokens_prev==0`
  case'inde) silinir → sadece son chunk'ın K/V'si tap'te kalır. Belirti: büyük tool result
  veya uzun ilk prompt'tan sonra `tap=N tail=M` mismatch (`N << M`). Fix: tap'i her chunk
  sonrası oku ve acc'a ekle.
- **Çift BOS:** apply branch literal `<bos>` ekliyor; `tokenize(..., add_special=true)`
  da BOS ekliyor → çift. Belirti: `check_double_bos_eos: ... 2 BOS tokens` warning. Fix:
  daima `add_special=false`.
- **stderr log spam:** llama internal log'ları stdout'u boğuyor, kullanıcı cevabı tam
  göremiyor. Fix: `llama_log_set(cb)` ile filtre — WARN ve üzeri görünür, INFO/DEBUG gizli
  (LLAMA_VERBOSE=1 ile aç). Ayrıca tüm verbose log'lar `logs/session-YYYYMMDD-HHMMSS.log`
  dosyasına yazılıyor (sonra inceleme için).
- **KV cache exhaustion cascade (sonradan bulundu):** uzun tool result (README, log dosyası)
  veya çok turn sonrası `acc_nkv` n_ctx'e yaklaşıyor → `llama_decode` `failed to find a
  memory slot for batch of size N` döner → state korrupt, sonraki turn'ler de patlar
  (`decode failed`, `gen=0 tok`). Belirti: 3-4 ardışık decode fail.
  **Fix iki katmanlı:**
  1. **Proactive**: REPL turn başında `acc_nkv > n_ctx - 1024` ise history rotate edilir
     (system msg + sadece yeni user msg tutulur; KV temizlenir).
  2. **Reactive**: decode fail olduğunda `reset_chat(keep_last_user=false)` çağrılır —
     sadece system msg kalır, sonraki user prompt fresh başlar. `llama_memory_seq_rm` ile
     target + draft KV cache temizlenir, acc K/V vektörleri ve `last_formatted` sıfırlanır.
- **Prefill counter inaccurate when decode fails (sonradan):** `turn_prefill_tok` peşin
  sayılıyordu (`+= tail.size()`); decode chunk ortasında fail olunca counter tam, süre kısa
  → bogus rate (`prefill=6094 tok / 0.11s = 57628 t/s`). **Fix:** her başarılı chunk
  decode'undan sonra `turn_prefill_tok += n` (gerçekten işlenen).
- **Empty turn (gen=0) UX (sonradan):** model bazen ilk token olarak EOG sample ediyor →
  turn boş bitiyor → kullanıcı hang sanıyor. **Fix:** `[chat] (model produced no output
  for this turn)` notu basılır.

---

## 5. Mevcut performans tablosu

### E2B (target 2B Q8, draft 78M F16) — Metal -ngl 99

| Test | Tool | t/s | Acc | Round | Hızlanma |
|------|------|----:|----:|------:|---------:|
| baseline (no SD) | llama-spec/chat | 51 | — | — | 1.00× |
| **primes (n=5) — REBUILD FIX SONRASI** | llama-chat | **114.7** | %87 | 43 | **2.25×** |
| **primes (n=5) — REBUILD FIX SONRASI** | llama-spec | **106.1** | %82 | 45 | **2.08×** |
| primes (n=5) — fix öncesi | llama-spec | 80.6 | %78 | — | 1.58× |
| primes (n=5) — fix öncesi | llama-chat | 57.3 | %79 | 24 | 1.12× |
| count 1-100 (n=2) | llama-spec | 53.1 | %62 | — | 1.04× |
| Eiffel paragraf (n=2) | llama-spec | 50.6 | %52 | — | 1.00× |
| robot hikayesi (n=2) | llama-spec | 41.5 | %39 | — | 0.81× |

### E4B (target 4B Q8, draft F16) — Metal -ngl 99

**Eski (rebuild fix öncesi):**

| Test | Tool | t/s | Acc | Round | Hızlanma |
|------|------|----:|----:|------:|---------:|
| baseline | llama-spec | 29.4 | — | — | 1.00× |
| count 1-100 (n=3) | llama-spec | 49.8 | %85 | 83 | 1.69× |
| photosynthesis (n=3) | llama-spec | 38.0 | %55 | — | 1.29× |
| Eiffel (n=3) | llama-spec | 30.6 | %52 | — | 1.03× |

**REBUILD FIX SONRASI (yeni, llama-chat) — kapsamlı bench:**

| Prompt | Baseline | SD (n=3) | Hızlanma | Acc | Yorum |
|--------|---------:|---------:|---------:|----:|-------|
| First 50 primes | 28.5 | **73.6** | **2.58×** | %99 | Şampiyon — düzenli liste |
| First 50 primes (n=5) | 28.5 | 70.7 | 2.48× | %89 | Daha düşük acc |
| Verbs list (alphab) | 27.6 | 52.0 | 1.88× | %60 | Liste, orta tahmin |
| Python factorial | (~28) | 62.6 | 2.20× | %76 | Kod, kalıplı |
| Python reverse string | 27.2 | 50.8 | 1.87× | %59 | Kod |
| Count 1-50 | (~28) | 62.4 | 2.19× | %81 | Sayma |
| Database index açıklama | 27.4 | 36.8 / 32.9 | 1.20-1.29× | %25-33 | Açıklayıcı düz metin |
| TCP açıklama | 27.4 | 32.9 | 1.20× | %25 | Açıklayıcı |
| Sea poem | 25.8 | 29.2 | 1.13× | %26 | Yaratıcı |
| Autumn haiku | (~28) | 28.0 | 0.98× | %25 | Yaratıcı kısa |
| Translate FR (10 tok) | 20.8 | 26.7 | 1.28× | %33 | Çok kısa, prefill domine |

**Pattern (acc rate doğrudan hızla korele):**
- %99 acc → 2.58× (teorik max'a yakın)
- %60-80 acc → 1.9-2.2×
- %25-35 acc → 1.1-1.3× (overhead kazancı yiyor)

**Yapısal görevde 3× hedefe %88 ulaşıldı (2.58/3.0). Yaratıcı metinde SD net etkisiz/marjinal.**

**Q8 draft assistant testi (E4B):** F16 (174 MB) vs Q8 (100 MB) — hız ve acc rate aynı
(±1 t/s, gürültü). Beklenen +%10-15 kazanç gerçekleşmedi. Sebep: draft model çok küçük
(78M params), round'daki payı sadece ~%10. Bandwidth halve marjinal etki. **Q8'in tek avantajı
disk/RAM tasarrufu** (~75 MB).

**Pattern:** kazanç prompt tahmin edilebilirliğiyle orantılı, model boyutuyla orantılı.

---

## 6. Niye %79 acceptance 3× hızlanma getirmiyor?

### Saf teorik üst sınır

E2B, n=5, acc %79 → round başına ~4 accept + 1 bonus = ~5 token/round.

**İdeal maliyet:**
- Verify batch (6 token) GPU'da memory-bound: ≈1 target forward
- 5 draft forward (78M, ~%3 target boyutu): ≈0.15 target forward
- **Toplam ~1.15 target-forward × 5 token = 0.23 forward/token → ~4× teorik**

### Gerçekte ne oluyor?

Round başına ölçülemeyen sabit overhead'ler:

| Bileşen | Tahmini ms/round (Metal, E2B) |
|---------|---:|
| Verify batch (target) | ~20 |
| 5 draft forward (assistant) | ~5 |
| **Graph rebuild (sched_need_reserve=true)** | **~15-20** |
| Tap copy GPU→CPU (K/V + hidden) | ~3 |
| Scheduler bookkeeping / Metal cmd buffer sync | ~5 |
| **TOPLAM** | **~48 ms / 5 token = ~9.6 ms/token = 104 t/s** |

Ölçüm: 57 t/s = 17.5 ms/token. Aradaki ~8ms muhtemelen Metal GPU dispatch latency + memory
sync.

### Ana darboğaz: graph rebuild

`set_assistant_shared_kv` her draft round'ında `sched_need_reserve=true` set ediyor (n_kv
büyüdüğü için tensor şekli değişiyor). Bu her round'da:
1. Draft graph yeniden plan
2. Buffer reallocation
3. ggml-alloc graph traversal

CPU-side iş ama Metal GPU pipeline'ı dolduramadan blokluyor.

### Niye E2B 3×'e ulaşamıyor

Baseline 51 t/s = 19.6 ms/token. Round overhead 30 ms zaten baseline'ın 1.5 katı. 5 token'a
bölünse bile 6 ms/token kazanç → maksimum 100 t/s teorik (2× baseline). Ölçüm bunun yarısı
çünkü ek sync overhead var.

### Niye E4B'de daha iyi

E4B baseline 30 t/s = 33 ms/token. Round 48ms / 5 token = 9.6 ms/token. Teorik 3.4×. Ölçüm
1.69× — ama orada da %85 acc ve düşük n=3 ile.

E4B'de adaptive n + graph rebuild fix ile 2.5-3× erişilebilir.

---

## 7. 3×'e ulaşmak için yol haritası

### 7.1 Per-round graph rebuild eliminasyonu ✅ TAMAMLANDI

**Sorun:** Her draft round'da n_kv büyüyor → tensor şekli değişiyor → graph rebuild
(~15-20 ms/round CPU-side).

**Yapılan çözüm:** Bucketed K/V capacity + padded mask.
- `llama_assistant_shared_kv` struct'a `n_kv_full_cap` / `n_kv_swa_cap` alanları
- `set_assistant_shared_kv` bucket'ı (next-power-of-2, min 256) hesaplar; sadece bucket
  büyürse `sched_need_reserve=true` set eder
- `build_inp_assistant_kv` tensor'ları `n_kv_cap` boyutunda alloc eder (actual değil)
- `set_input` actual K/V'yi buffer'ın başına kopyalar, tail zero-fill; mask `[0, actual)=0`,
  `[actual, cap)=-INF` ile unused pozisyonları softmax'ta sıfırlar
- Session başına ~5 rebuild (256→512→1024→2048→4096), eskiden round başına 1 (~50+/turn)
- Toplam: ~55 satır eklenti (`llama-graph.{h,cpp}` + `llama-context.{h,cpp}`)

**Ölçülen etki (E2B, Metal -ngl 99):**

| Test | Önce | Sonra | Δ |
|------|-----:|------:|--:|
| **llama-chat primes (n=5)** | 57 t/s | **114.7 t/s** | **+%100** |
| **llama-spec primes (n=5)** | 80 t/s | **106.1 t/s** | **+%32** |
| baseline (greedy) | 51 t/s | 51 t/s | aynı |
| acc_rate (primes) | %79 | %87 | +%8 |
| Lossless | ✓ | ✓ | — |

**E2B chat artık baseline'ın 2.25× hızında** (yapısal görevde). chat'in ekstra kazancı (%32
yerine %100) chat-spesifik per-round overhead'in (tail tokenize, template apply) de tek-seferlik
olmasından geliyor.

**Beklenen E4B etkisi (denenmedi):** baseline 30 → 60-70 t/s = ~2-2.3× (test edilecek).

### 7.2 Adaptive draft length

**Sorun:** Sabit `n=5`, prompt'a göre overhead/kazanç dengesi bozuk. Yaratıcı metinde %30 acc
ile n=5 boşa 3-4 draft yapıyor.

**Çözüm:** Her round'dan sonra acc rate'i izle, n'i dinamik ayarla:
- acc rate > %70 → n++ (max 8)
- acc rate < %30 → n-- (min 1)
- Hysteresis ile salınımı önle

**Tahmini etki:** Karışık prompt'larda %20-40 daha iyi t/s. Yaratıcı metin pozitif tarafa geçer
(1.0× → 1.2-1.5×).

**Tahmini LOC:** ~30 satır (chat.cpp + spec.cpp).

### 7.3 SWA mask flip + multi-position verify batching

**Sorun:** Verify batch'i sıralı pozisyonlarda decode ediyor (`acc_nkv, acc_nkv+1, ...`). SWA
layer'lar sliding window kullanıyor — kısa context'te fark etmez ama uzun context'te SWA
attention pattern hatalı çıkabilir.

**Çözüm:** Verify'da SWA mask'ı manuel set et. (Ref: HF candidate generator'daki `swa_mask.flip(dims=(-1,))`).

**Tahmini etki:** Uzun context (>512) doğruluğu. Hız etkisi minimum.

**Tahmini LOC:** ~20 satır.

### 7.4 Daha büyük target — E4B'de full evaluation

E2B çok hızlı. E4B (4B param) baseline 30 t/s — SD'nin overhead'i orantısal olarak daha küçük
kalır. Mevcut llama-chat'le E4B test edilmedi (sadece llama-spec).

**Eylem:** llama-chat'i E4B çiftiyle test et:
```bash
./build/bin/llama-chat \
  -m  /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.F16.gguf \
  -ngl 99 -c 2048 -b 512 --draft-max 3
```

Beklenen: yapısal görevde 50+ t/s (vs 30 baseline) = 1.7×, graph-rebuild fix ile 2-2.5×.

### 7.5 Chat overhead azaltma

**Sorun:** llama-spec primes (n=5): 80 t/s; llama-chat aynı: 57 t/s. Fark **chat-spesifik
overhead** (~%29):
- `apply_template` her turn → string concat, std::string allocation
- `parse_tool_call` regex tarama her turn
- Multi-turn msg history yönetimi
- `last_formatted` substr diff

**Çözümler:**
- Template apply'ı incremental yap (önceki formatted'a son msg'yi append et)
- Tool detection'ı regex yerine substring + manual parse
- assistant_text içine token IDs sakla, decode'u lazy yap

**Tahmini etki:** Chat overhead'i %30 → %10. E2B chat SD 57 → 70 t/s.

**Tahmini LOC:** ~100 satır.

### 7.6 KV scratch buffer pool

**Sorun:** Verify her round'da tap host buffer'larını silip yeniden allocate ediyor
(`n_tokens_prev==0` ile reset). std::vector::insert() alloc maliyeti var.

**Çözüm:** Tek bir büyük scratch buffer, append yerine offset yaz.

**Tahmini etki:** Round başına ~1-2 ms kazanç. Marjinal.

**Tahmini LOC:** ~30 satır.

### 7.7 Draft model quantization optimization

**Şu an:** Assistant F16 (174 MB).

**Alternatif:** Assistant Q8 (87 MB) → memory bandwidth %50 azalır → draft forward ~2× hızlı.
Ama acc rate düşebilir (centroid head precision).

**Tahmini etki:** Draft cost 5 → 2.5 ms/round. Acc rate'e bağlı net hız.

**Tahmini iş:** Yeni Q8 assistant gguf üret (HF safetensors → quantize → gguf).

### 7.8 Roadmap özet

| Önerilen iş | Tahmini etki | LOC | Risk |
|------------|--------------|----:|------|
| **Graph rebuild fix** | E2B 80 → 110 t/s, **E4B 50 → 70 t/s** | 80 | Orta |
| **Adaptive n** | Karışık prompt'larda +20-40% | 30 | Düşük |
| **Chat overhead azaltma** | E2B chat 57 → 70 t/s | 100 | Orta |
| E4B llama-chat testi | (sadece ölçüm) | 0 | Yok |
| Q8 assistant denemesi | ?% (acc bağımlı) | 0 + quantize | Düşük |
| SWA mask doğruluğu | Uzun context | 20 | Düşük |
| KV scratch pool | Marjinal | 30 | Düşük |

**3×'e ulaşma sırası:** önce graph rebuild fix → E4B chat testi → adaptive n. Bu üç madde
yapısal görevde ~3×'e yaklaştırır (özellikle E4B + adaptive n ile).

---

## 7.5 Denenip işe yaramayan / geri alınan değişiklikler

Şeffaflık için: kazançlar kadar başarısızlıklar da kayda alındı. Negatif sonuçlar gelecek
oturumlarda aynı yola girilmesini önler.

### A. Faz 1 — Incremental K/V upload + mask delta (geri alındı)

**Hipotez:** Her round'da TÜM `acc->k_full` (~2-3 MB) GPU'ya yükleniyor. Sadece YENİ pozisyonları
(1-3 token) yüklesek round başı ~3-5ms tasarruf, E2B chat 114 → 130-140 t/s.

**Uygulanan değişiklik (~80 satır, `src/llama-graph.{h,cpp}` + `set_input` refactor):**
- `llm_graph_input_assistant_kv`'ye `prev_n_kv_*_uploaded`, `prev_mask_*_actual`, `first_call_after_alloc` state
- İlk çağrı: full init. Sonraki: `ggml_backend_tensor_set` ile sadece `[prev..new)` aralığı
- Mask delta: sadece değişen `[prev_actual..new_actual)` range güncelle
- Bucket min 256 → 1024 değişikliği de bundle'a dahil edildi

**Ölçüm sonucu (E2B chat primes n=5, baseline 51 t/s):**
- Önceden (cf9d17f, rebuild fix): **114.7 t/s**
- Faz 1 (incremental + bucket 1024): **99.7 t/s** (-13%) — YAVAŞ
- Faz 1 (incremental + bucket 256): **111.6 t/s** (-3%) — marjinal yavaş

**Kök sebep:**
1. **Bucket 1024 cap-büyütmesi attention compute'unu ~4× artırdı.** Mask -INF unused pozisyonlar
   için softmax sıfırlasa da `Q @ K^T` matmul tüm cap pozisyonları için çalışır.
   n_kv=50 actual, cap=1024 → matmul 1024 üzerinden = ~20× boş compute → +4-5 ms/round.
2. **Incremental upload byte tasarrufu ≠ time tasarrufu.** `ggml_backend_tensor_set` call başına
   CPU-side sync overhead ~50-200μs. Önceden 6 büyük call → 600μs. Faz 1: 8 küçük call → 800μs.
   Byte ~10x küçüldü ama call SAYISI arttı (mask için per-token-row). Net: yavaş.

**Karar:** `git checkout HEAD -- ...` ile geri alındı. cf9d17f (rebuild fix + bucket 256, full upload)
optimal nokta.

**Ders:** Small-data regiminde call overhead bytes'tan baskın. Optimization yapılırken
**call count + bytes** ikisi birden minimize edilmeli. Faz 1 sadece bytes'ı azalttı.

### B. Bucket min 1024 (deneme, geri alındı)

Pre-warm avantajı için bucket min'i 256→1024 yapıldı. Sonuç: short-context turn'lerde
attention compute ~4× boş çalıştı → çıktı yavaşladı. 256 sweet spot.

### C. Q8 draft assistant (test edildi, kazanç yok)

**Hipotez:** F16 draft (174 MB) → Q8 (100 MB) → memory bandwidth halve → draft forward ~%30
hızlı → E4B SD +%10-15.

**Test sonucu (E4B chat, draft-max 3):**
| Prompt | F16 t/s | Q8 t/s | Δ |
|--------|--------:|-------:|--:|
| Primes 50 | 74.3 | 75.0 | +0.7 (gürültü) |
| Python reverse | 50.2 | 48.5 | -1.7 (gürültü) |
| TCP açıklama | 33.3 | 31.8 | -1.5 (gürültü) |

**acc rate** F16 ve Q8'de aynı (%99 / %59 / %25). Centroid head precision Q8'de bozulmamış.

**Kök sebep:**
- Draft model çok küçük (78M params, 174 MB F16). Per-forward ~1-2 ms. Bandwidth-bound değil
  **compute/launch-overhead-bound** Metal'de.
- Round'da draft cost zaten küçük pay (~%10). Halve etsen tasarruf round'un %5'i.

**Karar:** Q8 sadece disk/RAM tasarrufu (75 MB), hız aynı. F16 kalıyor (varsayılan).

### D. Tier 3 (GPU-persistent K/V buffer) — analiz edildi, henüz uygulanmadı

**Beklenen kazanç hesabı:**
- E4B primes (%99 acc, şu an 74 t/s)
- Teorik max (round başı maliyet sıfır): ~90 t/s (round = 35 ms verify + 9 ms draft, 4 token/round)
- 74/90 = **%82 teorik max'a yakın** zaten
- Tier 3 sadece K/V upload + sync overhead'ini eler (~3-5 ms/round). Beklenen 74 → 80 t/s = **+%8**

**ROI:** ~150 LOC + ggml backend buffer ownership + scheduler etkileşim riski karşılığında %8.
Modest.

**3× hedefine ulaşmaz.** Yapısal görevde 2.6× → ~2.8× yapar. 3×'e gelmek için MEDUSA tarzı
tree drafting (~500 LOC, büyük redesign) ya da daha hızlı draft model arch gerek.

**Karar:** Şimdilik ertelendi. Yapılabilir ama meyve modest, risk var.

### E. Faz 1 pre-warm bucket transitions (Tier 4-A) — yapılmadı

İlk turn'de gizli warm-up'la bucket 256/512/1024 hepsini reserve et → kullanıcı ilk turn'de
delay yaşamaz. UX iyileştirmesi, hız etkisi yok. Faz 1 başarısız olduğu için bu da uygulanmadı.

### F. Adaptive n_max — yapılmadı, ROI yüksek (gelecek iş)

Yaratıcı %25 acc prompt'larda SD efektif 0.98× (kayıp). Adaptive ile `acc/round < 0.5` görünce
n→1 veya SD-off yap. Worst case 1× garanti, best case mevcut 2.6×.

LOC: ~30. **Tier 1'in başarısızlığından sonra en pragmatik bir sonraki iş.**

---

## 7.7 Git commit ilerleyişi

```
3b32d89 first commit                              ← only-needed-files baz iskelet
933afd0 tool,agent loop, speculative decoding     ← SD port + chat + tool (Aşama 2+3)
8bb9fb4 bug fix                                   ← 3 bug fix (cache exhaustion + prefill + gen=0)
cf9d17f hızlandırma                               ← Graph rebuild fix (bucketed cap + mask)
                                                    → E2B 57→114 t/s, E4B 30→74 t/s = 2.25-2.63×
d5ecbf1 deneme                                    ← Faz 1 incremental upload denemesi (geri alındı)
```

Şu an working tree = `cf9d17f hızlandırma` (en iyi sonuç).

---

## 8. Bilinen sınırlamalar / out of scope

1. **Sadece gemma-4 hedefli.** Diğer model'lerin chat template'leri için legacy hand-coded
   path'leri çalışır ama tool calling için gemma-4'e özel.
2. **jinja yok** — only-needed-files prensibi gereği. Yeni gemma türevleri için template
   manuel eklemek gerekir.
3. **Tool'lar read-only.** `write_file`/`bash` yok (güvenlik). Eklenebilir aynı sandbox guard ile.
4. **Tek-tool-per-turn.** Paralel tool çağrıları yok.
5. **Tool arg parsing tek `path` argümanı için optimize.** Multi-arg tool'lar için genişletme
   gerek.
6. **Greedy sampling sabit.** Top-k/temperature yok. Eklemek için `llama_sampler` kullanılabilir.
7. **Geliştirme persistence yok.** Program kapanınca history kaybolur. Tüm verbose log
   `logs/session-*.log` dosyasında.
8. **Adaptive n yok** (yukarıda önerildi).
9. **Per-round graph rebuild kaldırılmadı** (yukarıda önerildi).
10. **History rotation kayıp veri demek.** Cache dolunca eski mesajlar atılır; model önceki
    bağlamı hatırlamaz. Daha akıllı: summarize-then-evict (gelecek iş).

---

## 9. Dosya envanteri

| Dosya | Satır | Tip |
|-------|------:|-----|
| `src/llama-arch.{h,cpp}` | +23 | additive |
| `src/llama-hparams.{h,cpp}` | +12 | additive |
| `src/llama-cparams.h` | +1 | additive |
| `src/llama-ext.h` | +37 | additive |
| `src/llama-graph.{h,cpp}` | +133 | additive |
| `src/llama-context.{h,cpp}` | +148 | additive |
| `src/llama-model.cpp` | +50 | additive |
| `src/models/models.h` | +30 | additive |
| `src/models/gemma4.cpp` | +50 | additive (tap) |
| `src/models/gemma4_assistant.cpp` | 240 | **YENİ** |
| `src/llama-chat.{h,cpp}` | +25 | additive (gemma-4 template) |
| `examples/spec/spec.cpp` | 438 | **YENİ** |
| `examples/spec/CMakeLists.txt` | 5 | **YENİ** |
| `examples/chat/chat.cpp` | 631 | **YENİ** |
| `examples/chat/CMakeLists.txt` | 5 | **YENİ** |
| `CMakeLists.txt` (top) | +2 | additive |
| **TOPLAM YENİ** | **~1830 satır** | |

Eski raporlar:
- `llama.cpp/GEMMA4_ASSISTANT_HANDOFF.md` (önceki oturum)
- `llama.cpp/GEMMA4_ASSISTANT_FIX_REPORT.md` (SD bug fix detayı)
- `llama.cpp/E2B_BENCHMARK.md` (E2B prompt sweep)
- `only-needed-files/CHAT_TOOLS_REPORT.md` (chat + tool detayı)
- **Bu dosya** — master / konsolide rapor

---

## 10. Komutlar (referans)

```bash
cd /Users/enes/Desktop/all/less-llama-cpp/only-needed-files
cd build && cmake --build . -j 8 && cd ..

# Tek-shot SD (en hızlı yapısal görev için)
./build/bin/llama-spec \
  -m /Users/enes/Desktop/all/llms/gemma-4-E2B-it-UD-Q8_K_XL.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E2B-it-assistant.F16.gguf \
  --spec-type draft-mtp --spec-draft-n-max 5 \
  -p "List the first 50 prime numbers." -n 400 -ngl 99

# Multi-turn chat + tool + SD
./build/bin/llama-chat \
  -m /Users/enes/Desktop/all/llms/gemma-4-E2B-it-UD-Q8_K_XL.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E2B-it-assistant.F16.gguf \
  -ngl 99 -c 2048 --draft-max 3

# Chat tool'suz + thinking
./build/bin/llama-chat -m <model> -ngl 99 --no-tools --thinking

# Chat sandbox darlat
./build/bin/llama-chat -m <model> -ngl 99 --root /tmp

# Baseline (regression)
./build/bin/llama-simple -m <model> -n 200 -ngl 99 "prompt"
```
