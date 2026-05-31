# Gemma-4 Assistant Speculative Decoding — Bug Avı & Fix Raporu

Tarih: 2026-05-26
Model çifti: `gemma-4-E2B-it-UD-Q8_K_XL.gguf` (target) + `gemma-4-E2B-it-assistant.F16.gguf` (draft/MTP head)
Çalışma ağacı: `/Users/enes/Desktop/all/less-llama-cpp/llama.cpp`

---

## 0. TL;DR

Speculative decoding tamamen bozuktu: **%0 acceptance**, draft saçma token üretiyordu
(`HttpServlet`, `아멘`, emoji), 6.5 t/s, sadece CPU (Metal crash ediyordu).

3 bug bulundu ve düzeltildi:

1. **K/V tap tensor'larında `ggml_set_output()` eksik** (ROOT CAUSE) — `src/models/gemma4.cpp`
2. **Target embed scale eksik** (×√hidden) — `common/speculative.cpp`
3. **Centroid (masked-embedding) logit head implement edilmemiş** — `gemma4_assistant.cpp` + `speculative.cpp`

Sonuç — hızlanma görev tipine göre değişir (her ikisi de matched, clean pair):

| Görev | E2B (50 t/s baseline) | E4B (29 t/s baseline) |
|-------|----------------------|----------------------|
| Liste (primes) | **1.61×** (80.6 t/s, n=5) | (benzer beklenir) |
| Sayma | 1.05× (n=2) | **1.69×** (49.8 t/s, n=3) |
| Açıklayıcı metin | ~0.97× | 1.29× |
| Yaratıcı paragraf | ~0.92× | 1.03× |

Hepsi lossless. Optimum n_max prompt'a göre değişir (yaratıcı için 2-3, şablonlu liste için 5-6).

---

## 1. Başlangıç durumu (bozuk)

İlk çalıştırma:

```bash
cd /Users/enes/Desktop/all/less-llama-cpp/llama.cpp
./build/bin/llama-cli \
  -m /Users/enes/Desktop/all/llms/gemma-4-E2B-it-UD-Q8_K_XL.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E2B-it-assistant.F16.gguf \
  --spec-type draft-mtp -p "The capital of France is" -n 16 --temp 0 -st -ngl 0
```

Gözlem:
- Çıktı **lossless** (target verify garantisi) → "The capital ... The"
- Ama draft her token'da çöp öneriyordu: `HttpServlet`, `lename`, `아멘`, `🎂`, `Hiện`...
- Debug satırı: `TARGET id_last=16499(' Request') logit=32.756 dense_rank=5324`
  → doğru token, draft'ın dense lm_head'inde rank 5324 (yani draft hiçbir şey kabul ettirmiyor)
- Generation: **6.5 t/s** (acceptance 0, hız kazancı yok)
- Metal (`-ngl 99`) crash ediyordu (eski handoff bug #11), bu yüzden CPU-only

`GEMMA4_ASSISTANT_HANDOFF.md` (önceki oturumlar) durumu: pipeline kurulu, derleniyor,
lossless, ama acceptance 0; blocker'lar açık (centroid head implement edilmemiş, E4B target
uyuşmazlığı, pre/post-norm doğrulanmamış).

Bu oturumda yeni olan: kullanıcı **eşleşen E2B çiftini** indirdi (E2B target + E2B assistant).
Önceki oturumlarda E4B assistant + E2B target geometri uyuşmazlığı vardı. Artık geometri uyuyor
→ pipeline çalışıyor ama acceptance hâlâ 0.

---

## 2. Build & Run komutları

Build:

```bash
cd /Users/enes/Desktop/all/less-llama-cpp/llama.cpp/build
cmake --build . --target llama-cli -j 8
```

Çalıştır (düzeltmelerden sonra, önerilen):

```bash
cd /Users/enes/Desktop/all/less-llama-cpp/llama.cpp
./build/bin/llama-cli \
  -m  /Users/enes/Desktop/all/llms/gemma-4-E2B-it-UD-Q8_K_XL.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E2B-it-assistant.F16.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 4 \
  -p "The capital of France is" -n 128 --temp 0 -st -ngl 99
```

Notlar:
- `-st` = single-turn (yoksa interaktif mod EOF'a kadar `> ` basıyor)
- `--spec-draft-n-max N` = round başına kaç draft token (eski `--draft-max` kaldırılmış)
- Acceptance istatistiği: `G4A STATS:` satırı stderr'e basılıyor (rounds / drafted / accepted)
- Debug ayrıntısı için: `G4A_DEBUG=1` env (varsayılan kapalı)

---

## 3. Bug avı metodolojisi (adım adım)

Anahtar prensip: **tahmin etme, ölç.** llama.cpp çıktısını authoritative HF (transformers)
reference'ına karşı sayısal karşılaştır.

Ortam (önceki oturumdan mevcuttu):
- venv: `/tmp/g4a_venv` (torch 2.12 + transformers 5.8.1, `gemma4_assistant` native destekli)
- HF reference dizinleri:
  - `gemma4-assistant-e2b/` (assistant, model-4.safetensors)
  - `gemma4-e2b/` (target, multimodal Gemma4ForConditionalGeneration, model-3.safetensors 10GB)
- Standart isim symlink'leri: `/tmp/g4asst_e2b`, `/tmp/g4tgt_e2b`

### Adım 1 — draft forward doğru mu? (cosine testi)

llama'nın draft input'larını dump'la (`G4A_DUMP=1`), HF reference assistant'a ver, çıktıyı
karşılaştır (`/tmp/g4a_ref_e2b.py`).

Sonuç:
```
FEATURE (post_projection): REF |feat|=152.10  LLAMA |feat|=152.14  cosine=1.0000
```
**Cosine 1.0** → llama'nın draft forward'ı aynı input'larla referansla AYNI sonucu üretiyor.
Forward matematiği (cross-attention, rope, norm) **doğru**. Ama referans da llama'nın
dump'ıyla doğru token'ı bulamıyor (rank 4096) → bug forward'da değil, **input'larda**.

Çıkarım: cosine 1.0 sadece "aynı input → aynı çıktı" der; input'ların KENDİSİ yanlışsa
ikisi de aynı yanlış cevabı verir.

### Adım 2 — candidate generator'ın tam tarifi (HF kaynak okuma)

`transformers/generation/candidate_generator.py` → `SinglePositionMultiTokenCandidateGenerator`:
```python
last_hidden_state = model_outputs.hidden_states[-1]        # target hidden
last_token_id     = input_ids[:, -1:]                       # son token
position_ids      = [[input_ids.shape[1] - 1]]             # seq_len - 1
inputs_embeds     = cat([embed(last_token_id), last_hidden_state])
```
`hidden_states[-1]` pre-norm mu post-norm mu? → küçük rastgele Gemma4TextModel ile
empirik test: norm weight randomize edilince `hidden_states[-1] == last_hidden_state` (POST-norm).
→ llama da post-norm tap'liyor, **tutarlı**.

`store_full_length_kv` (modeling_gemma4.py:1252): her layer-type'ın son non-shared layer'ının
K/V'sini saklıyor. K = k_norm+rope, **V = v_norm (scale'siz, rope'suz)**. llama tap'ı
(gemma4.cpp) bununla **birebir uyuyor**.

→ Recipe seviyesinde her şey uyuyor: hidden post-norm, K/V transform/layer, pozisyon, token,
concat sırası. Yine de acceptance 0.

### Adım 3 — ground truth: mekanizma HF'de çalışıyor mu? (10GB target yükle)

`/tmp/g4a_groundtruth2.py`: HF target + HF assistant, gerçek pipeline, çok adımlı.

Normal template (14 token):
```
target: "The capital of France is **Paris**."
ASSISTANT acceptance: 8/8  (her token doğru)
```
Thinking template (`enable_thinking=True`, 21 token — llama'nın ürettiği rejimle AYNI):
```
prompt: [2,105,9731,107,98,...]   (llama'nın 21 token'ıyla aynı)
target: '<|channel>','thought','\n','Thinking',' Process',':','\n\n','1'
ASSISTANT acceptance: 6/8
```

**KESİN İZOLASYON:** Mekanizma HF'de çalışıyor (thinking rejiminde bile 6/8).
llama AYNI rejimde 0/8. → **Bug llama'da**, model/regime'de değil.

### Adım 4 — bileşen bileşen sayısal karşılaştırma (`/tmp/g4a_cmp.py`)

llama'nın dump'ını (round-0, n_kv=21, cur_tok=100) HF'nin AYNI token dizisi için hesapladığı
değerlerle karşılaştır:

```
EMBED:  cos=1.0000 RAW ile (ratio 1.0) | HF SCALED kullanıyor (ratio 39.188 = √1536)
HIDDEN: cos=0.9997  ✓
k_full: cos=0.9998  ✓
v_full: cos=-0.0243  ✗✗  (anti-correlated — tamamen yanlış)
k_swa:  cos=0.0621   ✗✗
v_swa:  cos=0.4326   ✗
```

→ İki sorun net:
1. **embed scale eksik** (ratio 39 = √1536)
2. **v_full / k_swa / v_swa tap'leri çöp** (k_full şans eseri doğru)

### Adım 5 — root cause: tap host-copy

`src/llama-context.cpp` decode tap'i `ggml_backend_tensor_get_async` ile intermediate
graph tensor'larını okuyor. Bir tensor `ggml_set_output()` ile işaretlenmemişse ggml-alloc
scheduler onun buffer'ını sonraki op'lar için reuse eder → async okuma **bayat/çöp** veri alır.

`src/models/gemma4.cpp`'de hidden tap `ggml_set_output` çağırıyordu (415), ama 4 K/V tap'i
ÇAĞIRMIYORDU (sadece `ggml_build_forward_expand`). k_full şans eseri (allocation sırası)
hayatta kalıyor, V_full ve SWA K/V üzerine yazılıyor.

**Bu aynı zamanda Metal crash'inin de sebebiydi** (`GGML_ASSERT(buf_dst)` — okunabilir Metal
buffer'ı olmayan tensor).

---

## 4. Yapılan değişiklikler (dosya bazında)

### Fix #1 (ROOT CAUSE) — `src/models/gemma4.cpp`

K/V tap tensor'larını graph output olarak işaretle:

```cpp
// önce: sadece ggml_build_forward_expand
// sonra:
if (kv_tap_k_full) { ggml_set_output(kv_tap_k_full); ggml_build_forward_expand(gf, kv_tap_k_full); }
if (kv_tap_v_full) { ggml_set_output(kv_tap_v_full); ggml_build_forward_expand(gf, kv_tap_v_full); }
if (kv_tap_k_swa)  { ggml_set_output(kv_tap_k_swa);  ggml_build_forward_expand(gf, kv_tap_k_swa);  }
if (kv_tap_v_swa)  { ggml_set_output(kv_tap_v_swa);  ggml_build_forward_expand(gf, kv_tap_v_swa);  }
```

Etki: V_full + SWA K/V artık doğru okunuyor. **Metal crash de çözüldü** (GPU artık çalışıyor).

### Fix #2 — `common/speculative.cpp` (draft loop)

Target embed'e √(backbone) scale uygula (candidate generator `ScaledWordEmbedding` kullanıyor):

```cpp
// llama_model_get_token_embd ham satır döner; Gemma4TextScaledWordEmbedding ×√hidden uygular
static const bool g4a_no_escale = getenv("G4A_NO_ESCALE") != nullptr;
if (!g4a_no_escale) {
    const float es = sqrtf((float) n_embd_backbone);   // √1536 ≈ 39.19
    for (auto & x : tmp_embd) { x *= es; }
}
```

Etki: |embd| 1.3 → 51.8 (HF ile eşleşiyor).

### Fix #3 — centroid (masked-embedding) logit head

`src/models/gemma4_assistant.cpp` (build graph):
- `cur` (256-d post-norm hidden) → `centroid_logits = mtp_centroids @ cur` ([n_centroids=2048])
- centroid logits, embeddings çıktısına concat edilerek host'a açıldı:
  `res->t_embd = concat(feature[1536], centroid_logits[2048])`
- `hparams.n_embd_out_impl = n_embd_backbone + n_centroids` (ordered head varsa)

`src/models/gemma4_assistant.cpp` (load hparams):
```cpp
hparams.n_embd_out_impl = hparams.n_embd_backbone +
    (hparams.use_ordered_embeddings ? hparams.n_centroids : 0);
```

`common/speculative.cpp`:
- ctor'da `token_to_cluster[canonical_id] = i / vocab_per_centroid` haritası kuruldu
  (`token_ordering.view(n_centroids, vocab_per_centroid)` → cluster = flat_index / 128)
- draft loop'ta dense lm_head argmax yerine **masked argmax**: top-32 centroid cluster seç,
  dense logit'leri sadece o cluster'ların token'larına kısıtla, argmax al.
  (Referans masked_embedding ile matematiksel eşdeğer: per-token logit aynı dot product.)

`src/llama-model.cpp` + `src/llama-ext.h`:
- `llama_model_get_centroid_params(model, &n_centroids, &top_k)` accessor eklendi.

### Yardımcı / temizlik
- Dump gate'i tek-sefer (round 0) yapıldı (`dbg == 0`) — temiz reference karşılaştırması için.
- Hot-path debug print'leri kaldırıldı: `llama-graph.cpp` pos set_input fprintf (her step),
  `llama-context.cpp` tap-copy LLAMA_LOG_ERROR (her ubatch).
- Draft debug blokları `G4A_DEBUG` env'ine alındı (varsayılan kapalı).
- `G4A STATS:` (rounds/drafted/accepted/accept_rate) destructor'da basılıyor.

---

## 5. Doğrulama scriptleri (`/tmp`, yeniden üretilebilir)

| Script | Ne yapar |
|--------|----------|
| `/tmp/g4a_ref_e2b.py` | llama dump → HF assistant, feature cosine + token rank |
| `/tmp/g4a_groundtruth.py` | HF target+assistant, ilk draft, tek adım |
| `/tmp/g4a_groundtruth2.py` | HF, çok adımlı acceptance (normal template) |
| `/tmp/g4a_gt_think.py` | aynısı `enable_thinking=True` (21 token rejimi) |
| `/tmp/g4a_postest.py` | pozisyon (L-2/L-1/L) hangisi doğru token'ı verir |
| `/tmp/g4a_cmp.py` | llama dump vs HF, bileşen bazında cos + magnitude |

Dump üretmek için:
```bash
G4A_DUMP=1 ./build/bin/llama-cli -m <target> -md <draft> --spec-type draft-mtp \
  -p "The capital of France is" -n 4 --temp 0 -st -ngl 0
# /tmp/g4a_llama_{concat,kfull,vfull,kswa,vswa,logits,feat}.bin yazılır (round 0)
/tmp/g4a_venv/bin/python /tmp/g4a_cmp.py
```

---

## 6. Sonuçlar

### Acceptance (E2B, Metal, "The capital of France is", n=128)

| n_max | accept_rate | avg accepted/round |
|-------|-------------|--------------------|
| 2 | %62.2 | 1.24 |
| 3 | %50.0 | 1.50 |
| 4 | %45.5 | 1.82 |
| 6 | %38.5 | 2.31 |

(HF reference bf16 ile thinking rejiminde 6/8 = %75; bizim Q8 target + F16 assistant %50–62.
Fark muhtemelen target quantization.)

### Performans (Metal -ngl 99)

E2B çifti (baseline ~50 t/s) — prompt sweep:

| Prompt | baseline | spec en iyi | hızlanma |
|--------|----------|-------------|----------|
| "List the first 50 prime numbers." | 50.1 | **80.6 (n=5)** | **1.61×** |
| "List the first 30 prime numbers." | 50.2 | 74.3 (n=4) | 1.48× |
| "Count from 1 to 100" | 50.6 | 53.1 (n=2) | 1.05× |
| "Photosynthesis simple terms" | 50.7 | 49.3 (n=3) | 0.97× |
| "Eiffel Tower history paragraph" | 50.6 | 46.6 (n=3) | 0.92× |

Bug öncesi: 6.5 t/s (bozuk, acceptance 0). CPU'da spec kaybeder
(compute-bound, batch avantajı yok) — sadece Metal anlamlı.

### Hızlanma neye bağlı

- **Çıktı tahmin edilebilirliği**: şablonlu liste (primes) %78+ acc → 1.6× kazanç;
  yaratıcı paragraf %40-55 acc → marjinal kayıp.
- **Optimum n_max prompt'a göre değişir** — yaratıcı için 2-3, şablonlu liste için 5-6.
  avg_accepted/round değeri en iyi göstergedir; n_max bunu geçince fazla draft boşa overhead.
- Round başına **graph rebuild** overhead'i (`sched_need_reserve=true`, n_kv her round
  büyüdüğü için tensor şekli değişir) tabanı sınırlar. Bunu kaldırmak (sabit-kapasite K/V + mask)
  tüm prompt'larda taban kazancı yükseltebilir.

### E4B denemesi

İlk denemede eldeki E4B target'lar finetune'du (`Gemma-4-E4B-Uncensored-Q6`,
`...OBLITERATED-Q4_K_M`) → assistant ile uyuşmuyor → **acceptance %0**.

Sonra kullanıcı temiz **`gemma-4-E4B-it-Q8_0.gguf`** + **`gemma-4-E4B-it-assistant.F16.gguf`**
çiftini indirdi → **spec decoding gerçekten kazandı.**

#### E4B sonuçları (Metal -ngl 99, -c 2048 -b 512, temp 0)

Not: tüm spec çıktıları baseline ile **byte-byte aynı** (sadece spinner + perf satırı farklı).
Lossless `diff` ile doğrulandı.

##### Prompt: "Count from 1 to 100, one number per line." (n=300)

| n_max | gen t/s | acc% | accepted/round | hızlanma |
|-------|---------|------|----------------|----------|
| baseline | 29.4 | — | — | 1.00× |
| **3** | **49.8** | %84.7 | 2.54 | **1.69×** |
| 4 | 48.1 | %73.3 | 2.93 | 1.64× |
| 5 | 42.9 | %59.7 | 2.99 | 1.46× |
| 6 | 38.5 | %49.8 | 2.99 | 1.31× |
| 8 | 29.8 | %37.3 | 2.99 | 1.01× |
| 10 | 28.9 | %29.9 | 2.99 | 0.98× |

Gözlem: avg_accepted/round n=4'ten sonra **2.99'da doyuyor** — model genelde 2–3 ileri
tahmin ediyor, daha fazla draft sadece boşa overhead. **n_max=3 sweet spot.**

##### Prompt çeşitliliği (n=200, n_max=3)

| Prompt | baseline | spec t/s | acc% | hızlanma |
|--------|----------|----------|------|----------|
| "Eiffel Tower history paragraph" | 29.6 | 30.6 (n=2) | %52 | 1.03× |
| "Water cycle in 5 steps" | 29.6 | 37.0 | (~%55) | 1.25× |
| "How photosynthesis works" | 29.5 | 38.0 | %55 | 1.29× |
| "Count 1 to 50" | 29.7 | 48.3 | %79 | 1.63× |
| "Count 1 to 100" | 29.4 | 49.8 | %85 | **1.69×** |

Çıkarım: hızlanma **prompt tahmin edilebilirliğiyle** doğru orantılı.
Yapısal/şablonlu çıktıda asistant 2-3 token ileriyi rahat tahmin ediyor (%80+ acc),
yaratıcı/uzun anlatımda %40-55 acc → daha mütevazı kazanç.

##### Önemli not: Metal OOM
17 GB RAM'de E4B Q8 (8.2 GB) + assistant + 2 context default `-c 4096` ile **OOM** veriyor.
Çözüm: `-c 2048 -b 512` (veya daha küçük). Daha büyük RAM'li Mac'lerde gerek olmaz.

---

## 7. Kalan iş / sonraki adımlar

1. **Round başına graph rebuild'i kaldır** (en yüksek değerli) — draft K/V'sini sabit kapasiteye
   (n_ctx) pad'le + mask kullan → tensor şekli sabit → rebuild yok. Teorik ~2× kazancı açabilir.
2. **Temiz E4B target** (`google/gemma-4-e4b-it`) edin → büyük target'ta spec asıl kazancı versin.
3. Acceptance'ı artır: daha yüksek precision target (Q8 → F16) acceptance'ı yükseltebilir.
4. Commit öncesi temizlik: kalan cold-path debug (`G4A load:` fprintf, ctor LOG_ERR,
   `G4A_PROBE`/`G4A_NOATTN`/`G4A_NOROPE`/`G4A_SWAP`/`G4A_POS` deney toggle'ları),
   STATS print'i opsiyonel yap.

---

## 8. Değişen dosyalar özeti

| Dosya | Değişiklik |
|-------|-----------|
| `src/models/gemma4.cpp` | K/V tap'lere `ggml_set_output` (ROOT CAUSE + Metal fix) |
| `common/speculative.cpp` | embed √scale default; centroid masked-argmax; cluster map; STATS; debug gate |
| `src/models/gemma4_assistant.cpp` | centroid logits concat; `n_embd_out_impl += n_centroids` |
| `src/llama-model.cpp` | `llama_model_get_centroid_params` accessor |
| `src/llama-ext.h` | accessor decl |
| `src/llama-graph.cpp` | hot-path pos debug print kaldırıldı |
| `src/llama-context.cpp` | hot-path tap-copy debug print kaldırıldı |

Git: henüz commit yok (tüm değişiklikler working tree'de).
