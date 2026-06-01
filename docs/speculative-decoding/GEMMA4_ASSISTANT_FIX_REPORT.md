# Gemma-4 Assistant Speculative Decoding — Bug Hunt & Fix Report

Date: 2026-05-26
Model pair: `gemma-4-E2B-it-UD-Q8_K_XL.gguf` (target) + `gemma-4-E2B-it-assistant.F16.gguf` (draft/MTP head)
Work tree: `/Users/enes/Desktop/all/less-llama-cpp/llama.cpp`

---

## 0. TL;DR

Speculative decoding was completely broken: **0% acceptance**, draft producing garbage tokens (`HttpServlet`, `아멘`, emoji), 6.5 t/s, CPU-only (Metal crashed).

3 bugs found and fixed:

1. **`ggml_set_output()` missing on K/V tap tensors** (ROOT CAUSE) — `src/models/gemma4.cpp`
2. **Target embed scale missing** (×√hidden) — `common/speculative.cpp`
3. **Centroid (masked-embedding) logit head not implemented** — `gemma4_assistant.cpp` + `speculative.cpp`

Result — speedup varies by task type (both models matched, clean pair):

| Task | E2B (~50 t/s baseline) | E4B (~29 t/s baseline) |
|------|----------------------|----------------------|
| List (primes) | **1.61×** (80.6 t/s, n=5) | (similar expected) |
| Counting | 1.05× (n=2) | **1.69×** (49.8 t/s, n=3) |
| Explanatory text | ~0.97× | 1.29× |
| Creative paragraph | ~0.92× | 1.03× |

All lossless. Optimal n_max varies by prompt (2-3 for creative, 5-6 for templated lists).

---

## 1. Starting State (broken)

First run:

```bash
cd /Users/enes/Desktop/all/less-llama-cpp/llama.cpp
./build/bin/llama-cli \
  -m /Users/enes/Desktop/all/llms/gemma-4-E2B-it-UD-Q8_K_XL.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E2B-it-assistant.F16.gguf \
  --spec-type draft-mtp -p "The capital of France is" -n 16 --temp 0 -st -ngl 0
```

Observations:
- Output **lossless** (target verify guarantee) → "The capital ... The"
- But draft proposed garbage on every token: `HttpServlet`, `lename`, `아멘`, `🎂`, `Hiện`...
- Debug line: `TARGET id_last=16499(' Request') logit=32.756 dense_rank=5324`
  → correct token, but rank 5324 in draft's dense lm_head (draft accepts nothing)
- Generation: **6.5 t/s** (0% acceptance, no speedup)
- Metal (`-ngl 99`) crashed (handoff bug #11), so CPU-only

`GEMMA4_ASSISTANT_HANDOFF.md` (previous sessions): pipeline built, compiles, lossless, but 0% acceptance; blockers open (centroid head not implemented, E4B target mismatch, pre/post-norm unverified).

New this session: user downloaded the **matching E2B pair** (E2B target + E2B assistant). Previous sessions had E4B assistant + E2B target geometry mismatch. Now geometry matches → pipeline runs but acceptance still 0.

---

## 2. Build & Run Commands

Build:

```bash
cd /Users/enes/Desktop/all/less-llama-cpp/llama.cpp/build
cmake --build . --target llama-cli -j 8
```

Run (after fixes, recommended):

```bash
cd /Users/enes/Desktop/all/less-llama-cpp/llama.cpp
./build/bin/llama-cli \
  -m  /Users/enes/Desktop/all/llms/gemma-4-E2B-it-UD-Q8_K_XL.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E2B-it-assistant.F16.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 4 \
  -p "The capital of France is" -n 128 --temp 0 -st -ngl 99
```

Notes:
- `-st` = single-turn (otherwise interactive mode spams `> ` to EOF)
- `--spec-draft-n-max N` = draft tokens per round (old `--draft-max` removed)
- Acceptance stats: `G4A STATS:` line printed to stderr (rounds / drafted / accepted)
- For debug detail: `G4A_DEBUG=1` env (off by default)

---

## 3. Bug Hunt Methodology (step by step)

Key principle: **don't guess, measure.** Compare llama.cpp output numerically against authoritative HF (transformers) reference.

Environment (carried over from previous sessions):
- venv: `/tmp/g4a_venv` (torch 2.12 + transformers 5.8.1, `gemma4_assistant` natively supported)
- HF reference directories:
  - `gemma4-assistant-e2b/` (assistant, model-4.safetensors)
  - `gemma4-e2b/` (target, multimodal Gemma4ForConditionalGeneration, model-3.safetensors 10GB)
- Standard name symlinks: `/tmp/g4asst_e2b`, `/tmp/g4tgt_e2b`

### Step 1 — Is draft forward correct? (cosine test)

Dump llama's draft inputs (`G4A_DUMP=1`), feed to HF reference assistant, compare outputs (`/tmp/g4a_ref_e2b.py`).

Result:
```
FEATURE (post_projection): REF |feat|=152.10  LLAMA |feat|=152.14  cosine=1.0000
```
**Cosine 1.0** → llama's draft forward produces the SAME result as reference given the same inputs. Forward math (cross-attention, rope, norm) is **correct**. But reference also can't find the correct token from llama's dump (rank 4096) → bug is not in forward, it's in the **inputs**.

Conclusion: cosine 1.0 only means "same input → same output"; if inputs are WRONG, both give the same wrong answer.

### Step 2 — Exact candidate generator recipe (HF source reading)

`transformers/generation/candidate_generator.py` → `SinglePositionMultiTokenCandidateGenerator`:
```python
last_hidden_state = model_outputs.hidden_states[-1]        # target hidden
last_token_id     = input_ids[:, -1:]                       # last token
position_ids      = [[input_ids.shape[1] - 1]]             # seq_len - 1
inputs_embeds     = cat([embed(last_token_id), last_hidden_state])
```
Is `hidden_states[-1]` pre-norm or post-norm? → empirical test with small random Gemma4TextModel: randomizing norm weights shows `hidden_states[-1] == last_hidden_state` (POST-norm). → llama also taps post-norm, **consistent**.

`store_full_length_kv` (modeling_gemma4.py:1252): stores K/V of last non-shared layer per layer-type. K = k_norm+rope, **V = v_norm (unscaled, unroped)**. llama tap (gemma4.cpp) matches **exactly**.

→ At recipe level everything matches: hidden post-norm, K/V transform/layer, position, token, concat order. Yet acceptance still 0.

### Step 3 — Ground truth: does mechanism work in HF? (load 10GB target)

`/tmp/g4a_groundtruth2.py`: HF target + HF assistant, real pipeline, multi-step.

Normal template (14 tokens):
```
target: "The capital of France is **Paris**."
ASSISTANT acceptance: 8/8  (every token correct)
```
Thinking template (`enable_thinking=True`, 21 tokens — SAME regime llama produces):
```
prompt: [2,105,9731,107,98,...]   (same 21 tokens as llama)
target: '<|channel>','thought','\n','Thinking',' Process',':','\n\n','1'
ASSISTANT acceptance: 6/8
```

**DEFINITIVE ISOLATION:** Mechanism works in HF (6/8 even in thinking regime). llama gets 0/8 in same regime. → **Bug is in llama**, not in model/regime.

### Step 4 — Component-by-component numerical comparison (`/tmp/g4a_cmp.py`)

Compare llama's dump (round-0, n_kv=21, cur_tok=100) against HF-computed values for the same token sequence:

```
EMBED:  cos=1.0000 RAW (ratio 1.0) | HF uses SCALED (ratio 39.188 = √1536)
HIDDEN: cos=0.9997  ✓
k_full: cos=0.9998  ✓
v_full: cos=-0.0243  ✗✗  (anti-correlated — completely wrong)
k_swa:  cos=0.0621   ✗✗
v_swa:  cos=0.4326   ✗
```

→ Two problems clear:
1. **embed scale missing** (ratio 39 = √1536)
2. **v_full / k_swa / v_swa tap garbage** (k_full correct by luck)

### Step 5 — Root cause: tap host-copy

`src/llama-context.cpp` decode tap reads intermediate graph tensors via `ggml_backend_tensor_get_async`. If a tensor is NOT marked with `ggml_set_output()`, ggml-alloc scheduler reuses its buffer for subsequent ops → async read gets **stale/garbage** data.

In `src/models/gemma4.cpp`, the hidden tap called `ggml_set_output` (line 415), but the 4 K/V taps did NOT (only `ggml_build_forward_expand`). k_full survives by luck (allocation order), V_full and SWA K/V overwritten.

**This was also the cause of the Metal crash** (`GGML_ASSERT(buf_dst)` — tensor with no readable Metal buffer).

---

## 4. Changes Made (by file)

### Fix #1 (ROOT CAUSE) — `src/models/gemma4.cpp`

Mark K/V tap tensors as graph outputs:

```cpp
// before: only ggml_build_forward_expand
// after:
if (kv_tap_k_full) { ggml_set_output(kv_tap_k_full); ggml_build_forward_expand(gf, kv_tap_k_full); }
if (kv_tap_v_full) { ggml_set_output(kv_tap_v_full); ggml_build_forward_expand(gf, kv_tap_v_full); }
if (kv_tap_k_swa)  { ggml_set_output(kv_tap_k_swa);  ggml_build_forward_expand(gf, kv_tap_k_swa);  }
if (kv_tap_v_swa)  { ggml_set_output(kv_tap_v_swa);  ggml_build_forward_expand(gf, kv_tap_v_swa);  }
```

Effect: V_full + SWA K/V now read correctly. **Metal crash also fixed** (GPU now works).

### Fix #2 — `common/speculative.cpp` (draft loop)

Apply √(backbone) scale to target embed (candidate generator uses `ScaledWordEmbedding`):

```cpp
// llama_model_get_token_embd returns raw row; Gemma4TextScaledWordEmbedding applies ×√hidden
static const bool g4a_no_escale = getenv("G4A_NO_ESCALE") != nullptr;
if (!g4a_no_escale) {
    const float es = sqrtf((float) n_embd_backbone);   // √1536 ≈ 39.19
    for (auto & x : tmp_embd) { x *= es; }
}
```

Effect: |embd| 1.3 → 51.8 (matches HF).

### Fix #3 — Centroid (masked-embedding) logit head

`src/models/gemma4_assistant.cpp` (build graph):
- `cur` (256-d post-norm hidden) → `centroid_logits = mtp_centroids @ cur` ([n_centroids=2048])
- Centroid logits concatenated with embeddings output, exposed to host:
  `res->t_embd = concat(feature[1536], centroid_logits[2048])`
- `hparams.n_embd_out_impl = n_embd_backbone + n_centroids` (when ordered head present)

`src/models/gemma4_assistant.cpp` (load hparams):
```cpp
hparams.n_embd_out_impl = hparams.n_embd_backbone +
    (hparams.use_ordered_embeddings ? hparams.n_centroids : 0);
```

`common/speculative.cpp`:
- ctor builds `token_to_cluster[canonical_id] = i / vocab_per_centroid` map
  (`token_ordering.view(n_centroids, vocab_per_centroid)` → cluster = flat_index / 128)
- Draft loop uses **masked argmax** instead of dense lm_head argmax: select top-32 centroid clusters, restrict dense logits to those clusters' tokens only, take argmax.
  (Mathematically equivalent to masked_embedding: same dot product per token.)

`src/llama-model.cpp` + `src/llama-ext.h`:
- Added `llama_model_get_centroid_params(model, &n_centroids, &top_k)` accessor.

### Helpers / Cleanup
- Dump gate limited to first call (round 0 only) — for clean reference comparison.
- Hot-path debug prints removed: `llama-graph.cpp` pos set_input fprintf (per step), `llama-context.cpp` tap-copy LLAMA_LOG_ERROR (per ubatch).
- Draft debug blocks gated behind `G4A_DEBUG` env (off by default).
- `G4A STATS:` (rounds/drafted/accepted/accept_rate) printed in destructor.

---

## 5. Verification Scripts (`/tmp`, reproducible)

| Script | What it does |
|--------|-------------|
| `/tmp/g4a_ref_e2b.py` | llama dump → HF assistant, feature cosine + token rank |
| `/tmp/g4a_groundtruth.py` | HF target+assistant, first draft, single step |
| `/tmp/g4a_groundtruth2.py` | HF, multi-step acceptance (normal template) |
| `/tmp/g4a_gt_think.py` | same with `enable_thinking=True` (21-token regime) |
| `/tmp/g4a_postest.py` | which position (L-2/L-1/L) gives correct token |
| `/tmp/g4a_cmp.py` | llama dump vs HF, component-wise cos + magnitude |

To generate dump:
```bash
G4A_DUMP=1 ./build/bin/llama-cli -m <target> -md <draft> --spec-type draft-mtp \
  -p "The capital of France is" -n 4 --temp 0 -st -ngl 0
# writes /tmp/g4a_llama_{concat,kfull,vfull,kswa,vswa,logits,feat}.bin (round 0)
/tmp/g4a_venv/bin/python /tmp/g4a_cmp.py
```

---

## 6. Results

### Acceptance (E2B, Metal, "The capital of France is", n=128)

| n_max | accept_rate | avg accepted/round |
|-------|-------------|--------------------|
| 2 | 62.2% | 1.24 |
| 3 | 50.0% | 1.50 |
| 4 | 45.5% | 1.82 |
| 6 | 38.5% | 2.31 |

(HF reference bf16 in thinking regime: 6/8 = 75%; our Q8 target + F16 assistant: 50-62%. Difference likely target quantization.)

### Performance (Metal -ngl 99)

E2B pair (baseline ~50 t/s) — prompt sweep:

| Prompt | baseline | spec best | speedup |
|--------|----------|-----------|---------|
| "List the first 50 prime numbers." | 50.1 | **80.6 (n=5)** | **1.61×** |
| "List the first 30 prime numbers." | 50.2 | 74.3 (n=4) | 1.48× |
| "Count from 1 to 100" | 50.6 | 53.1 (n=2) | 1.05× |
| "Photosynthesis simple terms" | 50.7 | 49.3 (n=3) | 0.97× |
| "Eiffel Tower history paragraph" | 50.6 | 46.6 (n=3) | 0.92× |

Before fix: 6.5 t/s (broken, 0% acceptance). CPU spec always loses (compute-bound, no batch advantage) — Metal only meaningful.

### What Speedup Depends On

- **Output predictability**: templated list (primes) 78%+ acc → 1.6× gain; creative paragraph 40-55% acc → marginal loss.
- **Optimal n_max varies by prompt** — 2-3 for creative, 5-6 for templated lists. avg_accepted/round is the best indicator; once n_max exceeds it, extra draft is pure overhead.
- Per-round **graph rebuild** overhead (`sched_need_reserve=true`, n_kv grows each round → tensor shape changes) limits the floor. Eliminating this (fixed-capacity K/V + mask) would raise base gains across all prompts.

### E4B Trial

First E4B target attempts were finetunes (`Gemma-4-E4B-Uncensored-Q6`, `...OBLITERATED-Q4_K_M`) → mismatch with assistant → **0% acceptance**.

Then user downloaded clean **`gemma-4-E4B-it-Q8_0.gguf`** + **`gemma-4-E4B-it-assistant.F16.gguf`** pair → **spec decoding actually won.**

#### E4B Results (Metal -ngl 99, -c 2048 -b 512, temp 0)

Note: all spec outputs **byte-for-byte identical** to baseline (lossless). Verified with `diff`.

##### Prompt: "Count from 1 to 100, one number per line." (n=300)

| n_max | gen t/s | acc% | accepted/round | speedup |
|-------|---------|------|----------------|---------|
| baseline | 29.4 | — | — | 1.00× |
| **3** | **49.8** | 84.7% | 2.54 | **1.69×** |
| 4 | 48.1 | 73.3% | 2.93 | 1.64× |
| 5 | 42.9 | 59.7% | 2.99 | 1.46× |
| 6 | 38.5 | 49.8% | 2.99 | 1.31× |
| 8 | 29.8 | 37.3% | 2.99 | 1.01× |
| 10 | 28.9 | 29.9% | 2.99 | 0.98× |

Observation: avg_accepted/round saturates at **2.99 after n=4** — model typically predicts 2-3 ahead, more draft is pure overhead. **n_max=3 is sweet spot.**

##### Prompt diversity (n=200, n_max=3)

| Prompt | baseline | spec t/s | acc% | speedup |
|--------|----------|----------|------|---------|
| "Eiffel Tower history paragraph" | 29.6 | 30.6 (n=2) | 52% | 1.03× |
| "Water cycle in 5 steps" | 29.6 | 37.0 | (~55%) | 1.25× |
| "How photosynthesis works" | 29.5 | 38.0 | 55% | 1.29× |
| "Count 1 to 50" | 29.7 | 48.3 | 79% | 1.63× |
| "Count 1 to 100" | 29.4 | 49.8 | 85% | **1.69×** |

Speedup directly proportional to **prompt predictability**. Structured/templated output: assistant easily predicts 2-3 tokens ahead (80%+ acc). Creative/long narrative: 40-55% acc → more modest gains.

##### Important note: Metal OOM
On 17 GB RAM, E4B Q8 (8.2 GB) + assistant + 2 contexts at default `-c 4096` causes **OOM**. Fix: `-c 2048 -b 512` (or smaller). Not needed on larger RAM Macs.

---

## 7. Remaining Work / Next Steps

1. **Eliminate per-round graph rebuild** (highest value) — pad draft K/V to fixed capacity (n_ctx) + use mask → tensor shape constant → no rebuild. Theoretically unlocks ~2× gain.
2. **Get clean E4B target** (`google/gemma-4-e4b-it`) → large target shows real spec gains.
3. Improve acceptance: higher precision target (Q8 → F16) may increase acceptance.
4. Pre-commit cleanup: remaining cold-path debug (`G4A load:` fprintf, ctor LOG_ERR, `G4A_PROBE`/`G4A_NOATTN`/`G4A_NOROPE`/`G4A_SWAP`/`G4A_POS` experiment toggles), make STATS print optional.

---

## 8. Changed Files Summary

| File | Change |
|------|--------|
| `src/models/gemma4.cpp` | `ggml_set_output` on K/V taps (ROOT CAUSE + Metal fix) |
| `common/speculative.cpp` | embed √scale default; centroid masked-argmax; cluster map; STATS; debug gate |
| `src/models/gemma4_assistant.cpp` | centroid logits concat; `n_embd_out_impl += n_centroids` |
| `src/llama-model.cpp` | `llama_model_get_centroid_params` accessor |
| `src/llama-ext.h` | accessor decl |
| `src/llama-graph.cpp` | hot-path pos debug print removed |
| `src/llama-context.cpp` | hot-path tap-copy debug print removed |

Git: no commit yet (all changes in working tree).
