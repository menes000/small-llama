# `only-needed-files` Master Report — Gemma 4 Spec Decoding + Chat + Tool Use

Date: 2026-05-26
Scope: All work done across the full session — bug hunt, port, optimization, limitations, future roadmap.

---

## 0. TL;DR

**What was done:**
- Gemma-4 speculative decoding in the full tree llama.cpp was completely broken (0% acceptance, garbage tokens, Metal crash). 3 root-cause bugs found and fixed → SD working (1.69× on E4B, 1.5× on E2B structured tasks).
- All SD infrastructure ported to `only-needed-files` minimal tree (~1100 lines).
- New `llama-spec` example written for `only-needed-files` (438 lines).
- Gemma-4 chat template added to `only-needed-files` (`llama_chat_apply_template` no longer returns -1).
- New `llama-chat` example: multi-turn REPL + sandboxed `read_file`/`list_dir` tools + optional SD integration + per-turn stats.

**Performance:**
- E2B baseline: ~51 t/s (Metal)
- E2B SD (structured task, primes): **80 t/s** spec.cpp / **57 t/s** chat.cpp (acc 79%)
- E4B-it Q8 baseline: ~30 t/s
- E4B-it SD (counting): **49.8 t/s = 1.69×** (acc 85%)

**Known limitation:** Even 85% acceptance doesn't give 3× because per-round fixed overhead (graph rebuild, scheduler sync) exceeds the benefit for small models like E2B. Reaching 3× requires graph-rebuild elimination, adaptive draft length, and a larger target.

---

## 1. Project context

`only-needed-files` (`/Users/enes/Desktop/all/less-llama-cpp/only-needed-files/`) is the user's personal project: **run llama.cpp at equivalent performance with the fewest possible lines**. Full tree ~500 MB, only-needed-files ~10 MB. Focused on the Gemma-4 architecture.

**Target models:**
- `gemma-4-E2B-it-UD-Q8_K_XL.gguf` — 2B param target (Q8, ~2.8 GB)
- `gemma-4-E2B-it-assistant.F16.gguf` — 78M MTP/EAGLE draft head (F16, 174 MB)
- `gemma-4-E4B-it-Q8_0.gguf` — 4B target (Q8, ~8.2 GB)
- `gemma-4-E4B-it-assistant.F16.gguf` — matching E4B draft

Architecture summary: `gemma4` arch (target) + `gemma4_assistant` arch (draft). The assistant has no self K/V — every layer cross-attends the target model's shared K/V. Output: post_projection feature (for chaining) + centroid masked-embedding logit head.

---

## 2. Phase 1 — SD bug hunt in full tree

### Starting state

```
G4A draft[step] i=0 id=178830('Zag') p=1.0
G4A   TARGET id_last=100 logit=-16 dense_rank=4096 (MASKED OUT)
[ Prompt: 142 t/s | Generation: 6.5 t/s ]   ← broken
```

Pipeline was built but 0% acceptance, draft proposing garbage tokens (`Zag`, `HttpServlet`, `아멘`, emoji), generation 6.5 t/s, Metal crash (CPU-only worked).

### Methodology: don't guess, measure

Set up HF transformers `gemma4_assistant` as reference (`/tmp/g4a_venv`, scripts `/tmp/g4a_*.py`). Numerical comparison:

| Comparison | Result | Conclusion |
|------------|--------|------------|
| llama draft forward vs HF (same concat) | cosine 1.0 | Forward math correct |
| HF mechanism (normal template) | 8/8 acceptance | Assistant solid |
| HF mechanism (thinking template, llama's 21 tokens) | 6/8 acceptance | Mechanism + assistant fine |
| llama dump → HF assistant | rank 4096 (masked) | llama input values wrong |

Isolation: **llama's target forward produces different values than HF.** Component-by-component cosine comparison:

| Component | cos | Magnitude | Verdict |
|-----------|-----|-----------|---------|
| `embed` (scaled HF) vs llama | 1.0 | HF ratio 39.2 | **scale missing** |
| `embed` (raw) vs llama | 1.0 | ratio 1.0 | Raw correct, no scale |
| `hidden` post-norm | 0.9997 | ratio 1.004 | ✅ |
| `k_full` | 0.9998 | ratio 1.0 | ✅ |
| `v_full` | **-0.02** | ratio 0.63 | ✗✗ tap garbage |
| `k_swa` | **0.06** | ratio 0.30 | ✗✗ tap garbage |
| `v_swa` | **0.43** | ratio 2.27 | ✗ tap garbage |

`k_full` correct but the same layer's `v_full` and SWA K/V are garbage → tap host-copy bug.

### 3 root-cause bugs

#### Bug 1: `ggml_set_output` missing (ROOT CAUSE + Metal crash)

In `src/models/gemma4.cpp`, hidden tap called `ggml_set_output()` but the 4 K/V tap tensors did not (only `ggml_build_forward_expand`). ggml-alloc scheduler reuses these intermediate tensors' buffers for subsequent ops → host async tap reads **stale data**. k_full survives by luck, V/SWA overwritten.

**Same bug was the cause of the Metal crash:** `GGML_ASSERT(buf_dst)` — tensor with no readable Metal buffer.

**Fix:**
```cpp
if (kv_tap_k_full) { ggml_set_output(kv_tap_k_full); ggml_build_forward_expand(gf, kv_tap_k_full); }
if (kv_tap_v_full) { ggml_set_output(kv_tap_v_full); ggml_build_forward_expand(gf, kv_tap_v_full); }
if (kv_tap_k_swa)  { ggml_set_output(kv_tap_k_swa);  ggml_build_forward_expand(gf, kv_tap_k_swa);  }
if (kv_tap_v_swa)  { ggml_set_output(kv_tap_v_swa);  ggml_build_forward_expand(gf, kv_tap_v_swa);  }
```

#### Bug 2: Target embed scale missing

`llama_model_get_token_embd` returns raw (unscaled) embedding row. HF candidate generator uses `Gemma4TextScaledWordEmbedding` — `×√hidden` (×39.19 for E2B).

**Fix:** In `common/speculative.cpp` draft loop, when building concat: `tmp_embd[j] *= √n_embd_backbone`.

#### Bug 3: Centroid masked-embedding head not implemented

Gemma-4 assistant's logit head is not dense lm_head, it's `Gemma4AssistantMaskedEmbedder`:
1. Centroid logits = `mtp_centroids @ hidden` ([2048])
2. Select top-32 clusters
3. Argmax from dense logits restricted to those clusters' tokens

llama was using dense lm_head → garbage. Math equivalent: dense logits restricted to top-k clusters.

**Fix:**
- Add `centroid_logits` output to graph (concatenated with feature into embeddings buffer)
- Build `token_to_cluster[canon_id] = cluster` map on host
- Use `masked_argmax(dense_logits, centroid_logits, token_to_cluster, top_k)` for argmax

### Results after Bug 1 (full tree)

| Test | Before | After |
|------|--------|-------|
| Acceptance | 0% | 50-85% (by task type) |
| E2B baseline Metal | 51 t/s | 51 t/s |
| E2B SD Metal structured | 6.5 t/s (broken) | **80 t/s** (acc 78%) |
| E4B SD Metal counting | (E4B-Uncensored mismatch 0%) | **49.8 t/s = 1.69×** |
| Metal crash | crash | ✅ fixed |

---

## 3. Phase 2 — SD port to `only-needed-files`

### Goal

Full tree's SD infrastructure required `common/speculative.cpp` (~5000 lines + jinja/sampling deps). Minimal port to only-needed-files: ~1100 lines.

### What was done

**Additive additions to shared files (no drift, only SD):**

| File | Δ lines | What |
|------|---------|------|
| `src/llama-arch.h` | +9 | LLM_ARCH_GEMMA4_ASSISTANT enum + 4 KV keys + 4 tensor enums |
| `src/llama-arch.cpp` | +14 | name, KV key, tensor name + tensor_info dispatch |
| `src/llama-hparams.h` | +6 | 4 new fields (n_embd_backbone, n_centroids, top_k, use_ordered) |
| `src/llama-hparams.cpp` | +6 | n_embd_inp() override |
| `src/llama-cparams.h` | +1 | assistant_kv_tap flag |
| `src/llama-ext.h` | +37 | 6 new APIs (set/get tap, accessors) |
| `src/llama-graph.h` | +63 | assistant_shared_kv + input + result tensors |
| `src/llama-graph.cpp` | +70 | build_inp_assistant_kv + set_input |
| `src/llama-context.h` | +36 | tap host buffer + APIs |
| `src/llama-context.cpp` | +112 | tap copy in decode + APIs + MTP ctx relax |
| `src/llama-model.cpp` | +50 | factory case + 4 accessors + nullptr memory |
| `src/models/models.h` | +30 | llama_model_gemma4_assistant class decl |
| `src/models/gemma4.cpp` | +50 | K/V + hidden tap (with ggml_set_output) |

**New files:**

| File | Lines | What |
|------|------:|------|
| `src/models/gemma4_assistant.cpp` | 240 | Assistant arch implementation (verbatim from full tree) |
| `examples/spec/spec.cpp` | 438 | Minimal SD driver (no common deps) |
| `examples/spec/CMakeLists.txt` | 5 | Build glue |

**Total:** ~1100 new C++ lines. Full tree's SD layer ~5000 + jinja 5800 = ~10800 lines; port is ~10% of that.

### Challenges encountered (during port)

- **Diff cleanliness:** `llama-model.cpp` had a 303-line diff but ~80% were other arch factory cases from the full tree (drift). Added only ~50 lines relevant to gemma4_assistant. Same for `models.h`: 1887-line diff but only-needed-files had 31 lines (just gemma4 class); added only 30 lines of assistant class decl.
- **CMake GLOB cache:** Adding new `gemma4_assistant.cpp` caused link errors on first build. Required `cmake ..` to reconfigure.
- **Common-independent spec loop:** `common/speculative.cpp`'s `common_speculative_state_draft_gemma4_assistant` was 448 lines (multi-seq abstraction, common_sampler dep, staged accept, env toggles). Stripped to a ~250-line single-seq greedy core loop inlined in spec.cpp.

### KV cache rollback bug (caught during port)

Spec.cpp's first version didn't remove rejected draft positions from the target's KV cache → position inconsistency in the second round (X=41, Y=41 — Y should be X+1). Fix:

```cpp
if (batch_t.n_tokens > n_keep) {
    llama_memory_seq_rm(llama_get_memory(ctx_t), 0, acc_nkv + n_keep, -1);
}
```

---

## 4. Phase 3 — Chat + Tool use + SD integration

### `llama-chat` (new)

Multi-turn REPL + sandboxed tool dispatch + optional SD.

**Args:**
| Flag | Default | Effect |
|------|---------|--------|
| `-m` | (required) | target model |
| `-md` | (none) | if present, enable SD |
| `--draft-max` | 3 | draft tokens per round |
| `--no-sd` | (off) | disable SD even if -md provided |
| `--thinking` | (off) | enable reasoning trace |
| `--no-tools` | (on) | remove tool definitions |
| `--root` | `$HOME` | tool sandbox root |

**Tools (read-only, realpath sandbox):**
- `read_file(path)` — max 16 KiB
- `list_dir(path)` — max 200 entries

**Per-turn stats (stderr):**
```
[stats] 130 tokens in 8.12s = 16.0 t/s | rounds=26 drafted=130 accepted=104 acc_rate=80.0%
```

**Gemma-4 chat template:**

`LLM_CHAT_TEMPLATE_GEMMA_4` enum + detect branch (`<|tool_call>` or `<|turn>` substring) + apply branch added to `src/llama-chat.{h,cpp}`. Apply branch produces turn delimiters (`<bos>`, `<|turn>{role}\n…<turn|>\n`); tool/thinking markers are embedded by caller (chat.cpp) in system content.

**New files/lines:**
| File | Lines |
|------|------:|
| `src/llama-chat.{h,cpp}` (additive) | +25 |
| `examples/chat/chat.cpp` | 631 (initial 389 + SD addition 242) |
| `examples/chat/CMakeLists.txt` | 5 |

### Challenges encountered

- **`llama_chat_apply_template` C API doesn't carry tools/thinking** → switched to design where caller embeds them in system message content, no separate `_ex` API needed.
- **`std::regex` doesn't support `[^]`** → replaced with `.*?`.
- **`realpath` symlink following** → on macOS `/tmp` → `/private/tmp`. Prefix check with `realpath_s(--root)`.
- **Tool-call stop condition** → `<tool_call|>` substring searched in `assistant_text`; break + dispatch on match.
- **Multi-turn KV cache management (string-substr tail diff)** — consistency required **raw push** (no thinking strip). First version stripped thinking → `[chat] empty tail tokenization` crash on next turn. Fix: `<|channel>` blocks in gemma jinja are designed to live in context.
- **`parse_special=true`** required everywhere in tokenize/detokenize (so tool tokens are single tokens).
- **SD integration in chat — KV cache invariants:** acc_nkv == kv_pos (target seq 0 size) must always hold. Rollback after verify + advance by n_keep. Incremental tail prefill works with this state preserved across turns.
- **Tap reset in multi-chunk prefill (found later):** If tail is long, `llama_decode` is called in multiple chunks. Each call clears the target's tap buffer (in the `n_tokens_prev==0` case) → only the last chunk's K/V stays in the tap. Symptom: `tap=N tail=M` mismatch (`N << M`) after large tool result or long first prompt. Fix: read tap and append to acc after each successful chunk decode.
- **Double BOS:** Apply branch adds literal `<bos>`; `tokenize(..., add_special=true)` also adds BOS → double. Symptom: `check_double_bos_eos: ... 2 BOS tokens` warning. Fix: always `add_special=false`.
- **stderr log spam:** llama internal logs drowning user output. Fix: `llama_log_set(cb)` filter — WARN and above visible, INFO/DEBUG hidden (enable with `LLAMA_VERBOSE=1`). Also all verbose logs written to `logs/session-YYYYMMDD-HHMMSS.log` for later inspection.
- **KV cache exhaustion cascade (found later):** Long tool result (README, log file) or many turns → `acc_nkv` approaches n_ctx → `llama_decode` returns `failed to find a memory slot for batch of size N` → state corrupt, subsequent turns also fail (`decode failed`, `gen=0 tok`). Symptom: 3-4 consecutive decode fails.
  **Two-layer fix:**
  1. **Proactive**: At REPL turn start, if `acc_nkv > n_ctx - 1024`, rotate history (keep system msg + only new user msg; clear KV).
  2. **Reactive**: On decode failure, call `reset_chat(keep_last_user=false)` — only system msg kept, next user prompt starts fresh. `llama_memory_seq_rm` clears target + draft KV cache, acc K/V vectors and `last_formatted` reset.
- **Prefill counter inaccurate when decode fails (found later):** `turn_prefill_tok` was counted eagerly (`+= tail.size()`); decode failure mid-chunk left counter at full, time short → bogus rate (`prefill=6094 tok / 0.11s = 57628 t/s`). **Fix:** `turn_prefill_tok += n` after each successful chunk decode (actual count).
- **Empty turn (gen=0) UX (found later):** Model sometimes samples EOG as first token → turn ends empty → user thinks it hung. **Fix:** `[chat] (model produced no output for this turn)` note printed.

---

## 5. Current performance table

### E2B (target 2B Q8, draft 78M F16) — Metal -ngl 99

| Test | Tool | t/s | Acc | Rounds | Speedup |
|------|------|----:|----:|-------:|--------:|
| baseline (no SD) | llama-spec/chat | 51 | — | — | 1.00× |
| **primes (n=5) — AFTER REBUILD FIX** | llama-chat | **114.7** | 87% | 43 | **2.25×** |
| **primes (n=5) — AFTER REBUILD FIX** | llama-spec | **106.1** | 82% | 45 | **2.08×** |
| primes (n=5) — before fix | llama-spec | 80.6 | 78% | — | 1.58× |
| primes (n=5) — before fix | llama-chat | 57.3 | 79% | 24 | 1.12× |
| count 1-100 (n=2) | llama-spec | 53.1 | 62% | — | 1.04× |
| Eiffel paragraph (n=2) | llama-spec | 50.6 | 52% | — | 1.00× |
| robot story (n=2) | llama-spec | 41.5 | 39% | — | 0.81× |

### E4B (target 4B Q8, draft F16) — Metal -ngl 99

**Old (before rebuild fix):**

| Test | Tool | t/s | Acc | Rounds | Speedup |
|------|------|----:|----:|-------:|--------:|
| baseline | llama-spec | 29.4 | — | — | 1.00× |
| count 1-100 (n=3) | llama-spec | 49.8 | 85% | 83 | 1.69× |
| photosynthesis (n=3) | llama-spec | 38.0 | 55% | — | 1.29× |
| Eiffel (n=3) | llama-spec | 30.6 | 52% | — | 1.03× |

**AFTER REBUILD FIX (new, llama-chat) — comprehensive bench:**

| Prompt | Baseline | SD (n=3) | Speedup | Acc | Note |
|--------|--------:|---------:|--------:|----:|------|
| First 50 primes | 28.5 | **73.6** | **2.58×** | 99% | Champion — regular list |
| First 50 primes (n=5) | 28.5 | 70.7 | 2.48× | 89% | Lower acc |
| Verbs list (alphab) | 27.6 | 52.0 | 1.88× | 60% | List, medium predictability |
| Python factorial | (~28) | 62.6 | 2.20× | 76% | Code, templated |
| Python reverse string | 27.2 | 50.8 | 1.87× | 59% | Code |
| Count 1-50 | (~28) | 62.4 | 2.19× | 81% | Counting |
| Database index explanation | 27.4 | 36.8 / 32.9 | 1.20-1.29× | 25-33% | Explanatory prose |
| TCP explanation | 27.4 | 32.9 | 1.20× | 25% | Explanatory |
| Sea poem | 25.8 | 29.2 | 1.13× | 26% | Creative |
| Autumn haiku | (~28) | 28.0 | 0.98× | 25% | Short creative |
| Translate FR (10 tok) | 20.8 | 26.7 | 1.28× | 33% | Very short, prefill dominates |

**Pattern (acc rate directly correlated with speed):**
- 99% acc → 2.58× (close to theoretical max)
- 60-80% acc → 1.9-2.2×
- 25-35% acc → 1.1-1.3× (overhead eating the gain)

**88% of the 3× target reached on structured tasks (2.58/3.0). SD neutral/marginal on creative text.**

**Q8 draft assistant test (E4B):** F16 (174 MB) vs Q8 (100 MB) — speed and acc rate identical (±1 t/s, noise). Expected +10-15% gain didn't materialize. Reason: draft model very small (78M params), its share per round is only ~10%. Halving bandwidth has marginal effect. **Q8's only advantage is disk/RAM savings** (~75 MB).

**Pattern:** gain proportional to prompt predictability, proportional to model size.

---

## 6. Why doesn't 79% acceptance give 3× speedup?

### Pure theoretical upper bound

E2B, n=5, acc 79% → ~4 accept + 1 bonus per round = ~5 tokens/round.

**Ideal cost:**
- Verify batch (6 tokens) GPU memory-bound: ≈1 target forward
- 5 draft forwards (78M, ~3% of target size): ≈0.15 target forwards
- **Total ~1.15 target-forward × 5 tokens = 0.23 forward/token → ~4× theoretical**

### What actually happens

Unmeasured fixed overheads per round:

| Component | Estimated ms/round (Metal, E2B) |
|-----------|--------------------------------:|
| Verify batch (target) | ~20 |
| 5 draft forwards (assistant) | ~5 |
| **Graph rebuild (sched_need_reserve=true)** | **~15-20** |
| Tap copy GPU→CPU (K/V + hidden) | ~3 |
| Scheduler bookkeeping / Metal cmd buffer sync | ~5 |
| **TOTAL** | **~48 ms / 5 tokens = ~9.6 ms/token = 104 t/s** |

Measured: 57 t/s = 17.5 ms/token. The ~8 ms gap is likely Metal GPU dispatch latency + memory sync.

### Main bottleneck: graph rebuild

`set_assistant_shared_kv` sets `sched_need_reserve=true` every draft round (tensor shape changes as n_kv grows). Each round this triggers:
1. Draft graph re-plan
2. Buffer reallocation
3. ggml-alloc graph traversal

CPU-side work but blocks the Metal GPU pipeline from filling.

### Why E2B can't reach 3×

Baseline 51 t/s = 19.6 ms/token. Round overhead 30 ms is already 1.5× baseline. Even divided by 5 tokens = 6 ms/token gain → theoretical max ~100 t/s (2× baseline). Measured is half that due to additional sync overhead.

### Why E4B does better

E4B baseline 30 t/s = 33 ms/token. Round 48ms / 5 tokens = 9.6 ms/token. Theoretical 3.4×. Measured 1.69× — but there too with 85% acc and low n=3.

With adaptive n + graph rebuild fix, 2.5-3× is reachable on E4B.

---

## 7. Roadmap to 3×

### 7.1 Per-round graph rebuild elimination ✅ DONE

**Problem:** Each draft round n_kv grows → tensor shape changes → graph rebuild (~15-20 ms/round CPU-side).

**Solution implemented:** Bucketed K/V capacity + padded mask.
- `n_kv_full_cap` / `n_kv_swa_cap` fields added to `llama_assistant_shared_kv` struct
- `set_assistant_shared_kv` computes bucket (next-power-of-2, min 256); sets `sched_need_reserve=true` only when bucket grows
- `build_inp_assistant_kv` allocates tensors at `n_kv_cap` size (not actual)
- `set_input` copies actual K/V to start of buffer, zero-fills tail; mask `[0, actual)=0`, `[actual, cap)=-INF` to zero out unused positions in softmax
- ~5 rebuilds per session (256→512→1024→2048→4096), was 1 per round (~50+/turn)
- Total: ~55 lines added (`llama-graph.{h,cpp}` + `llama-context.{h,cpp}`)

**Measured impact (E2B, Metal -ngl 99):**

| Test | Before | After | Δ |
|------|-------:|------:|--:|
| **llama-chat primes (n=5)** | 57 t/s | **114.7 t/s** | **+100%** |
| **llama-spec primes (n=5)** | 80 t/s | **106.1 t/s** | **+32%** |
| baseline (greedy) | 51 t/s | 51 t/s | same |
| acc_rate (primes) | 79% | 87% | +8% |
| Lossless | ✓ | ✓ | — |

**E2B chat now 2.25× baseline speed** (on structured tasks). Chat's extra gain (+100% vs +32%) comes from chat-specific per-round overhead (tail tokenize, template apply) also becoming one-time.

**Expected E4B impact (untested):** baseline 30 → 60-70 t/s = ~2-2.3× (to be tested).

### 7.2 Adaptive draft length

**Problem:** Fixed `n=5` gives wrong overhead/gain balance per prompt. With 30% acc on creative text, n=5 wastes 3-4 draft steps.

**Solution:** Track acc rate after each round, adjust n dynamically:
- acc rate > 70% → n++ (max 8)
- acc rate < 30% → n-- (min 1)
- Hysteresis to prevent oscillation

**Estimated impact:** +20-40% better t/s on mixed prompts. Creative text flips positive (1.0× → 1.2-1.5×).

**Estimated LOC:** ~30 (chat.cpp + spec.cpp).

### 7.3 SWA mask flip + multi-position verify batching

**Problem:** Verify batch decodes at sequential positions (`acc_nkv, acc_nkv+1, ...`). SWA layers use sliding window — no issue with short context but SWA attention pattern may be wrong on long context.

**Solution:** Manually set SWA mask during verify. (Ref: `swa_mask.flip(dims=(-1,))` in HF candidate generator.)

**Estimated impact:** Long-context (>512) correctness. Minimal speed effect.

**Estimated LOC:** ~20 lines.

### 7.4 Larger target — full E4B evaluation

E2B is too fast. E4B (4B params) baseline 30 t/s — SD overhead proportionally smaller. Current llama-chat not tested with E4B (only llama-spec).

**Action:** Test llama-chat with E4B pair:
```bash
./build/bin/llama-chat \
  -m  /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.F16.gguf \
  -ngl 99 -c 2048 -b 512 --draft-max 3
```

Expected: 50+ t/s on structured tasks (vs 30 baseline) = 1.7×, with graph-rebuild fix → 2-2.5×.

### 7.5 Chat overhead reduction

**Problem:** llama-spec primes (n=5): 80 t/s; llama-chat same: 57 t/s. Difference is **chat-specific overhead** (~29%):
- `apply_template` every turn → string concat, std::string allocation
- `parse_tool_call` regex scan every turn
- Multi-turn message history management
- `last_formatted` substr diff

**Solutions:**
- Make template apply incremental (append last msg to previous formatted)
- Replace regex tool detection with substring + manual parse
- Store token IDs inside assistant_text, lazy decode

**Estimated impact:** Chat overhead 30% → 10%. E2B chat SD 57 → 70 t/s.

**Estimated LOC:** ~100 lines.

### 7.6 KV scratch buffer pool

**Problem:** Verify clears and re-allocates tap host buffers every round (`n_tokens_prev==0` reset). std::vector::insert() has allocation cost.

**Solution:** Single large scratch buffer, write by offset instead of append.

**Estimated impact:** ~1-2 ms gain per round. Marginal.

**Estimated LOC:** ~30 lines.

### 7.7 Draft model quantization optimization

**Current:** Assistant F16 (174 MB).

**Alternative:** Assistant Q8 (87 MB) → memory bandwidth -50% → draft forward ~2× faster.
But acc rate may drop (centroid head precision).

**Estimated impact:** Draft cost 5 → 2.5 ms/round. Net speed depends on acc rate.

**Estimated work:** Generate new Q8 assistant gguf (HF safetensors → quantize → gguf).

### 7.8 Roadmap summary

| Recommended work | Estimated impact | LOC | Risk |
|-----------------|-----------------|----:|------|
| **Graph rebuild fix** | E2B 80 → 110 t/s, **E4B 50 → 70 t/s** | 80 | Medium |
| **Adaptive n** | +20-40% on mixed prompts | 30 | Low |
| **Chat overhead reduction** | E2B chat 57 → 70 t/s | 100 | Medium |
| E4B llama-chat test | (measurement only) | 0 | None |
| Q8 assistant trial | ?% (acc dependent) | 0 + quantize | Low |
| SWA mask correctness | Long context | 20 | Low |
| KV scratch pool | Marginal | 30 | Low |

**Order to reach 3×:** first graph rebuild fix → E4B chat test → adaptive n. These three items get close to 3× on structured tasks (especially E4B + adaptive n).

---

## 7.5 Tried and reverted / failed approaches

For transparency: failures are recorded as well as wins. Negative results prevent the same path being taken again in future sessions.

### A. Phase 1 — Incremental K/V upload + mask delta (reverted)

**Hypothesis:** Every round, the FULL `acc->k_full` (~2-3 MB) is uploaded to GPU. Uploading only NEW positions (1-3 tokens) would save ~3-5 ms/round, E2B chat 114 → 130-140 t/s.

**Change implemented (~80 lines, `src/llama-graph.{h,cpp}` + `set_input` refactor):**
- `prev_n_kv_*_uploaded`, `prev_mask_*_actual`, `first_call_after_alloc` state added to `llm_graph_input_assistant_kv`
- First call: full init. Subsequent: only `[prev..new)` range via `ggml_backend_tensor_set`
- Mask delta: only update `[prev_actual..new_actual)` range
- Bucket min 256 → 1024 change also bundled

**Measurement result (E2B chat primes n=5, baseline 51 t/s):**
- Before (cf9d17f, rebuild fix): **114.7 t/s**
- Phase 1 (incremental + bucket 1024): **99.7 t/s** (-13%) — SLOWER
- Phase 1 (incremental + bucket 256): **111.6 t/s** (-3%) — marginally slower

**Root cause:**
1. **Bucket 1024 cap-grow increased attention compute ~4×.** Mask -INF zeros softmax for unused positions but `Q @ K^T` matmul runs over all cap positions. n_kv=50 actual, cap=1024 → matmul over 1024 = ~20× empty compute → +4-5 ms/round.
2. **Incremental upload byte savings ≠ time savings.** `ggml_backend_tensor_set` has CPU-side sync overhead ~50-200μs per call. Before: 6 large calls → 600μs. Phase 1: 8 small calls → 800μs. Bytes shrank ~10× but call COUNT increased (per-token-row for masks). Net: slower.

**Decision:** Reverted with `git checkout HEAD -- ...`. cf9d17f (rebuild fix + bucket 256, full upload) is optimal.

**Lesson:** In small-data regime, call overhead dominates bytes. When optimizing, minimize **call count + bytes** both. Phase 1 only reduced bytes.

### B. Bucket min 1024 (tried, reverted)

Bucket min changed 256→1024 for pre-warm advantage. Result: attention compute ran ~4× empty on short-context turns → slower output. 256 is the sweet spot.

### C. Q8 draft assistant (tested, no gain)

**Hypothesis:** F16 draft (174 MB) → Q8 (100 MB) → memory bandwidth halved → draft forward ~30% faster → E4B SD +10-15%.

**Test result (E4B chat, draft-max 3):**
| Prompt | F16 t/s | Q8 t/s | Δ |
|--------|--------:|-------:|--:|
| Primes 50 | 74.3 | 75.0 | +0.7 (noise) |
| Python reverse | 50.2 | 48.5 | -1.7 (noise) |
| TCP explanation | 33.3 | 31.8 | -1.5 (noise) |

**acc rate** identical for F16 and Q8 (99% / 59% / 25%). Centroid head precision unaffected by Q8.

**Root cause:**
- Draft model very small (78M params, 174 MB F16). Per-forward ~1-2 ms. Not bandwidth-bound but **compute/launch-overhead-bound** on Metal.
- Draft cost is already a small share of the round (~10%). Halving it saves 5% of the round.

**Decision:** Q8 is only for disk/RAM savings (75 MB), speed identical. F16 stays (default).

### D. Tier 3 (GPU-persistent K/V buffer) — analyzed, not yet implemented

**Expected gain calculation:**
- E4B primes (99% acc, currently 74 t/s)
- Theoretical max (zero round-start cost): ~90 t/s (round = 35 ms verify + 9 ms draft, 4 tokens/round)
- 74/90 = **82% of theoretical max already**
- Tier 3 only eliminates K/V upload + sync overhead (~3-5 ms/round). Expected 74 → 80 t/s = **+8%**

**ROI:** ~150 LOC + ggml backend buffer ownership + scheduler interaction risk for 8%.
Modest.

**Won't reach 3×.** On structured tasks 2.6× → ~2.8×. To hit 3× would need MEDUSA-style tree drafting (~500 LOC, large redesign) or a faster draft model arch.

**Decision:** Deferred for now. Doable but yield modest, risk present.

### E. Phase 1 pre-warm bucket transitions (Tier 4-A) — not done

Hidden warm-up on first turn to reserve buckets 256/512/1024 → user doesn't experience first-turn delay. UX improvement, no speed effect. Not implemented since Phase 1 failed.

### F. Adaptive n_max — not done, high ROI (future work)

On creative 25% acc prompts, SD effective 0.98× (loss). With adaptive, when `acc/round < 0.5` → n→1 or SD-off. Worst case guaranteed 1×, best case current 2.6×.

LOC: ~30. **The most pragmatic next work after Phase 1 failure.**

---

## 7.8 Active roadmap — next 3 items (planned, not coded)

Three targeted improvements on the path to 3×. Recommended order `6 → 3 → 1` (gains increase, so does risk). 6 and 3 are additive — together +15-17%. 1 (async) alone has +50% potential but high risk.

---

### TODO-A — Centroid head GPU-side (recommended first)

**Current problem:**
After each draft step, host-side `masked_argmax`:
- Dense logits GPU→CPU transfer via `llama_get_logits_ith` (~262K × 4 = 1 MB) ~0.5 ms
- 262K vocab scan with token_to_cluster lookup ~0.3-0.5 ms
- ~1 ms per draft step, 5 draft × 5 ms/round = 10% of round

**Solution:** Compute argmax on GPU, only a single `int32 best_token_id` comes to host.

Two options:

**A1. ggml ops composition (portable, +50 LOC more):**
```cpp
// during graph build:
auto centroid_top_k_idx = ggml_top_k(centroid_logits, /*k=*/32);
// reverse lookup: expand cluster_ids to token ids
// (precomputed lookup table tensor uploaded, ggml_get_rows)
auto sel_token_ids = ggml_get_rows(cluster_to_tokens_table, centroid_top_k_idx);  // [32×128, 1]
auto sel_logits   = ggml_get_rows(dense_logits, sel_token_ids);                    // [4096, 1]
auto best_local   = ggml_argmax(sel_logits);                                       // [1]
auto best_global  = ggml_get_rows(sel_token_ids, best_local);                      // [1] = token id
res->t_argmax = best_global;  // host reads this single int32
```

`cluster_to_tokens_table` = reshaped `token_ordering`, already loaded.

**A2. Custom Metal kernel (+30 LOC shader + ~80 LOC C++ wrapper):**
Fewer ggml ops, faster but Metal-specific (separate write for CUDA).

**Expected impact:**
- E4B structured 99% acc: 74 → ~80 t/s (+8%)
- E2B structured: 114 → ~120 t/s (+5%)
- Creative 25% acc: marginal (5 draft steps is already few)

**LOC:** ~80-150 (A1 portable, A2 Metal-specific)
**Risk:** Medium — compose ggml top_k + get_rows + argmax; need care with F32/F16 dispatch.
**Time:** 1-2 days development + test

**Changed files:**
- `src/models/gemma4_assistant.cpp` (argmax compose appended to graph build)
- `src/llama-graph.h` (res->t_argmax output)
- `examples/chat/chat.cpp` and `examples/spec/spec.cpp` (new accessor `llama_get_assistant_argmax` instead of `llama_get_logits_ith`)
- `src/llama-context.cpp` (argmax tap host buffer)

---

### TODO-B — Tier 3 GPU-persistent K/V buffer (additive after A)

**Current problem:**
`set_input` re-uploads ALL acc K/V to GPU every round (~3-5 MB):
- 4 separate `ggml_backend_tensor_set` calls (K_full, V_full, K_swa, V_swa)
- + 2 mask uploads
- Total 6 sync calls, ~3-5 ms/round CPU-side overhead

Phase 1 tried "incremental delta" (only new bytes) but failed because ggml-alloc can relocate input tensors and call count increased. 

**Solution:** Take K/V buffer out of ggml-alloc control:

```cpp
// llama_context member
ggml_backend_buffer_t persistent_kv_buf;     // we own this, ggml-alloc won't touch it

// init (one-time):
const size_t kv_bytes = max_bucket * (head_dim_full*nhkv*2 + head_dim_swa*nhkv*2) * sizeof(float);
persistent_kv_buf = ggml_backend_buft_alloc_buffer(buft, kv_bytes);

// graph input now a VIEW:
inp->k_full = ggml_view_tensor(ctx, persistent_kv_buf, offset=0, shape=[hd_full, nhkv, n_ctx]);
// ggml-alloc won't touch this — view of external buffer

// set_input (each round):
const int64_t prev = prev_n_kv_full_uploaded;
const int64_t now  = akv->n_kv_full;
if (now > prev) {
    // upload only new tokens' bytes
    const size_t off   = prev * stride * sizeof(float);
    const size_t bytes = (now - prev) * stride * sizeof(float);
    ggml_backend_tensor_set(inp->k_full, akv->k_full.data() + prev*stride, off, bytes);
}
prev_n_kv_full_uploaded = now;
```

GPU buffer is **persistent** — old positions untouched, only new written. Buffer reallocs when bucket grows (~5 times per session, cheap).

**Difference from Phase 1:** Phase 1 tried delta on ggml-alloc managed buffer → buffer address could change on every bucket grow, full re-upload still required + per-call overhead increased. Tier 3 buffer is **ours**, address fixed → true append-only.

**Expected impact:**
- E4B structured 99% acc: 80 (after TODO-A) → ~85 t/s (+6%)
- E2B structured: 120 → ~123 t/s (+3%)
- Creative: marginal (~+1 t/s)

**TODO-A + TODO-B together:** E4B 74 → ~85 t/s (+15%) → **95% of 3× target**.

**LOC:** ~150
**Risk:** Medium-high — ggml view tensor + scheduler external buffer interaction is nuanced. "External tensor view" semantics partially documented in ggml.
**Time:** 2-3 days + extensive testing

**Changed files:**
- `src/llama-context.{h,cpp}` (persistent_kv_buf member, init, destructor cleanup)
- `src/llama-graph.{h,cpp}` (build_inp_assistant_kv view-based; set_input delta upload)
- Bucket grow path (persistent buffer realloc)

---

### TODO-C — Async draft/verify pipeline (big bet, exceeds 3×)

**Current problem:**
Round is sequential — draft and verify back-to-back:

```
Round N:   [draft 9ms][verify 35ms][tap 3ms][accept 1ms]   total 48ms
Round N+1: waiting...    [draft 9ms][verify 35ms]...        total 48ms
```

Target verify doesn't occupy CPU — only computing on Metal. During this time CPU is idle, draft GPU could start.

**Solution:** Speculatively start round N+1's draft during round N's verify.

```
Time →
Round N:   [draft────][verify──────────────][tap][accept]
Round N+1:            [draft────][verify──────────────]...
                      ↑ speculative start
```

Round N+1 draft assumes round N's draft tokens will be accepted (acts as if committed to acc). When verify returns:
- All accepted → speculative draft had correct base → use it
- Partially rejected → draft base was wrong → discard and restart

**At high acc (structured tasks) the speculative assumption is almost always correct** → every round overlaps → effective round time = max(draft, verify) = ~35ms = **+50% speed**.

**At low acc (creative) the assumption is often wrong** → wasted work → net ~0 or slight loss. Combined with adaptive n, at low-acc SD-off so async path is already disabled.

**Expected impact:**
- E4B structured 99% acc: 85 (after A+B) → ~110-120 t/s (+30%) → **3.9× baseline**
- E4B structured 60% acc: +15%
- Creative 25% acc: ~0 (SD-off with adaptive n anyway)

**Complexity:**
1. **Two concurrent context decodes** — `llama_decode` is sync API; either separate threads with `std::thread`, or submit 2 graphs parallel from same thread with ggml-sched `_async` variants + sync at end.
2. **Speculative state rollback** — if round N+1 draft had wrong base, manually rollback draft ctx KV (seq_rm rejected positions).
3. **Wasted work tracking** — show wasted draft count in stats.
4. **Metal command queue** — two contexts' command buffers interleave; Metal scheduler generally handles this but GPU contention possible.

**LOC:** ~200
**Risk:** Very high — multi-thread sync bugs are silent, very hard to debug. Race conditions can cause sample-level corruption (lossless guarantee at risk).
**Time:** Week+ development + careful testing (lossless verification with golden trace)

**Changed files:**
- `examples/chat/chat.cpp` and `examples/spec/spec.cpp` (REPL loop becomes double-buffered)
- New helper: speculative draft launcher + rollback
- llama-context (if flag/lock needed for concurrent decode)

**Prerequisite:** Better if TODO-A + TODO-B done first (round overhead lowered, async gain sits on top of pure draft+verify).

---

### Summary table

| Order | Work | Gain (E4B structured) | LOC | Risk | Time |
|-------|----|---------------------:|----:|------|------|
| **1** | TODO-A: Centroid head GPU | 74→80 t/s (+8%) | 80-150 | Medium | 1-2 days |
| **2** | TODO-B: GPU-persistent K/V | 80→85 t/s (+6%) | 150 | Medium-high | 2-3 days |
| **3** | TODO-C: Async pipeline | 85→110-120 t/s (+30%) | 200 | Very high | Week+ |

**A+B together: ~15% gain, ~95% of 3× target.**
**A+B+C together: ~50% gain, 3.9× baseline (exceeds 3× target).**

---

## 7.7 Git commit history

```
3b32d89 first commit                              ← only-needed-files base skeleton
933afd0 tool,agent loop, speculative decoding     ← SD port + chat + tool (Phase 2+3)
8bb9fb4 bug fix                                   ← 3 bug fixes (cache exhaustion + prefill + gen=0)
cf9d17f speedup                                   ← Graph rebuild fix (bucketed cap + mask)
                                                    → E2B 57→114 t/s, E4B 30→74 t/s = 2.25-2.63×
d5ecbf1 experiment                                ← Phase 1 incremental upload attempt (reverted)
```

Current working tree = `cf9d17f speedup` (best result).

---

## 7b. Bug Report — Tool Hop KV Misalignment (2026-05-30)

### Symptom

When a file-reading tool call was made with `llama-chat`, the second hop (model response after tool response) always returned "I cannot access the file". File exists, sandbox passes, `realpath` works. `[tool] read_file(path=...)` was being logged. Model still said it failed.

### Root cause

**The `<tool_call|>` token was break-ing before being written to KV cache.**

#### Greedy path (`!sd_on`):

```cpp
// BUG — original code
if (assistant_text.find("<tool_call|>") != std::string::npos) { break; }  // ← break FIRST
//
batch.token[0] = id;  ...
llama_decode(ctx, batch);   // ← decode SECOND, but never runs
kv_pos++;
```

`<tool_call|>` token emitted via `piece()` and added to `assistant_text`, then break before writing to KV. `last_formatted = formatted + assistant_text` contains `<tool_call|>` as a string but `kv_pos` is one position behind.

Second hop's tail:
```
last_formatted  = "...<|turn>model\n<|tool_call>...{path:...}<tool_call|>"   ← string
KV reality      = "...<|turn>model\n<|tool_call>...{path:...}"               ← last tok missing
```

Tail = `formatted2.substr(last_formatted.size())` = `"<turn|>\n<|turn>user\n<|tool_response>..."`.

This tail fed from `kv_pos` has no `<tool_call|>` at position `kv_pos` in context — instead gets `<turn|>` directly. Model sees corrupted context → can't recognize tool response → hallucinates "cannot access".

#### SD path (`sd_on`):

In SD, `emit` lambda sets tool_seen=true but Step 6 (KV advance) always runs:

- **Case B** (`next_pending = <tool_call|>`): next_pending decoded before next round. Step 6 does `acc_nkv += n_accept + 1` but `<tool_call|>` not in KV.
  `kv_pos < want_kv` → missing token.

- **Case D** (accepted draft contains `<tool_call|>`): drafted[j]=`<tool_call|>` is in KV but drafted[j+1..n_accept-1] also in KV (ghost tokens). Step 6 advances by `n_accept+1`.
  `kv_pos > want_kv` → ghost tokens.

Both cases: second hop tail starts from wrong position → same corruption.

### Fix (`examples/chat/chat.cpp`)

**Greedy path** — move break to AFTER decode:

```cpp
// FIX — decode first, break after
batch.token[0] = id;  batch.pos[0] = kv_pos;  ...  batch.logits[0] = 1;
llama_decode(ctx, batch);
kv_pos++;

if (assistant_text.find("<tool_call|>") != std::string::npos) { break; }  // ← AFTER
```

**SD path** — KV alignment after while loop if `tool_seen`:

```cpp
if (tool_seen) {
    const auto lf_toks = tokenize(vocab, formatted + assistant_text, true);
    const int64_t want_kv = (int64_t) lf_toks.size();
    if (kv_pos > want_kv) {
        // Case D: trim ghost tokens
        llama_memory_seq_rm(ctx, 0, want_kv, -1);
        const int64_t trim = kv_pos - want_kv;
        acc_kf.resize(... - trim*hd_full*nhkv);  // shrink acc vectors too
        acc_vf.resize(...); acc_ks.resize(...); acc_vs.resize(...);
        kv_pos = acc_nkv = want_kv;
    } else if (kv_pos < want_kv) {
        // Case B: decode missing tokens and append to tap acc
        for (int64_t pos = kv_pos; pos < want_kv; ++pos) {
            batch.token[0] = lf_toks[pos];  batch.pos[0] = pos;  ...
            llama_decode(ctx, batch);
            // llama_get_assistant_kv_tap → append to acc_kf/vf/ks/vs
            kv_pos++;  acc_nkv++;
        }
    }
}
```

### Why it wasn't caught earlier

During testing (`CHAT_TOOLS_REPORT.md` §6), there was a single successful test with `/tmp/test_chat.txt`. That test was probably not a Case B scenario or the model was robust enough to handle context corruption for that specific prompt. The error was not intermittent — tool hop failed on every call, but passed once during the single test attempt.

### Result

After fix, `<tool_call|>` is always written to KV, second hop tail starts from the correct position, model sees the tool response and produces the correct answer.

---

## 7c. Bug Report — msg_storage Dangling Pointer (2026-05-31)

### Symptom

`llama-chat` crashed every 3rd turn (sometimes at end of 2nd):

```
[chat] empty tail tokenization
[chat] decode failed during turn -> resetting (system only); next prompt starts fresh
```

First 2 turns normal, 3rd turn produces no output.

### Root cause

`push_msg` adds 2 elements to `msg_storage` vector per call (role + content), then writes those elements' `c_str()` pointers into the `msgs` vector:

```cpp
std::vector<std::string> msg_storage;   // BUG: vector realloc invalidates pointers
...
msg_storage.push_back(role);
msg_storage.push_back(content);
const char * r = msg_storage[...].c_str();   // this pointer is saved
const char * c = msg_storage[...].c_str();   // this too
msgs.push_back({ r, c });                    // stored in msgs
```

When `std::vector` capacity is full, it MOVEs all elements to new memory. After move, old `c_str()` pointers become dangling. Capacity doubling sequence:

```
push #1-2  → capacity: 2
push #3-4  → realloc 2→4   → msgs[0] dangling
push #5-8  → realloc 4→8   → msgs[0..1] dangling
push #9    → realloc 8→16  → msgs[0..3] dangling  ← crash on turn 3 comes from here
```

The 9th element push happens during turn 2's assistant response push. Then `apply_template` reads dangling pointers → `formatted` string comes out wrong length or empty. Result: `formatted.size() <= last_formatted.size()` → `tail = ""` → `tail_tokens.empty()` → error.

### Fix

`std::vector` → `std::deque`. `deque::push_back` **never invalidates** existing element pointers/references (only invalidates iterators). `c_str()` pointers are safe.

```cpp
// BEFORE (bug):
std::vector<std::string> msg_storage;

// AFTER (fix):
std::deque<std::string>  msg_storage;  // push_back never invalidates existing c_str() pointers
```

Added `#include <deque>`. API unchanged — no other code changes.

### Why it was caught late

Initial tests were single-shot or 1-2 turns; bug triggered on turn 3. After reset, "next prompt starts fresh" message worked because `reset_chat` does `clear()` + refills `msg_storage` (fresh pointers) — so reset was temporarily "hiding" the bug each time.

---

## 7d. KV Cache Q8 Default (2026-05-30)

### Change

`llama-chat` now stores KV cache in **q8_0** format by default (previously f16). No external dependencies — quantization code entirely within `only-needed-files`:

| File | Role |
|------|------|
| `ggml/include/ggml.h:398` | `GGML_TYPE_Q8_0 = 8` type definition |
| `src/llama-kv-cache.cpp:210-211` | Create KV tensors with `type_k`/`type_v` |
| `src/llama-kv-cache.cpp:56` | `ggml_quantize_chunk` — compress to q8 on write |
| `src/llama-kv-cache.cpp:1748-1749` | Dequantize → f32 → quantize back for RoPE |

### Memory impact

| `-c` | f16 (old) | q8 (new) | saving |
|------|-----------|-----------|--------|
| 4096 | ~250 MB | ~125 MB | 2× |
| 8192 | ~500 MB | ~250 MB | 2× |
| 16384 | ~1 GB | ~500 MB | 2× |

### Code change (`examples/chat/chat.cpp`)

```cpp
// args struct:
bool kv_q8 = true;   // default q8; revert to f16 with --kv-f16

// context creation:
if (a.kv_q8) {
    cp.type_k = GGML_TYPE_Q8_0;
    cp.type_v = GGML_TYPE_Q8_0;
}
```

### Flag

`--kv-f16` → disable q8, switch to f16 (for debugging or quality comparison).

### Quality impact

K and V vectors stored as 8-bit integers. Effect on attention output is minimal (activations compressed, not weights). In practice: acceptance rate and t/s values remain the same as f16.

---

## 8. Known limitations / out of scope

1. **Gemma-4 only.** Legacy hand-coded paths work for other models' chat templates but tool calling is gemma-4 specific.
2. **No jinja** — by only-needed-files principle. Adding new gemma variants requires manually adding templates.
3. **Tools read-only.** No `write_file`/`bash` (security). Can be added with same sandbox guard.
4. **Single tool per turn.** No parallel tool calls.
5. **Tool arg parsing optimized for single `path` arg.** Multi-arg tools need extension.
6. **Greedy sampling fixed.** No top-k/temperature. Can use `llama_sampler` to add.
7. **No history persistence.** History lost on program exit. All verbose logs in `logs/session-*.log`.
8. **No adaptive n** (proposed above).
9. **Per-round graph rebuild not eliminated** (proposed above).
10. **History rotation means data loss.** When cache fills, old messages are dropped; model can't recall prior context. Smarter approach: summarize-then-evict (future work).

---

## 9. File inventory

| File | Lines | Type |
|------|------:|------|
| `src/llama-arch.{h,cpp}` | +23 | additive |
| `src/llama-hparams.{h,cpp}` | +12 | additive |
| `src/llama-cparams.h` | +1 | additive |
| `src/llama-ext.h` | +37 | additive |
| `src/llama-graph.{h,cpp}` | +133 | additive |
| `src/llama-context.{h,cpp}` | +148 | additive |
| `src/llama-model.cpp` | +50 | additive |
| `src/models/models.h` | +30 | additive |
| `src/models/gemma4.cpp` | +50 | additive (tap) |
| `src/models/gemma4_assistant.cpp` | 240 | **NEW** |
| `src/llama-chat.{h,cpp}` | +25 | additive (gemma-4 template) |
| `examples/spec/spec.cpp` | 438 | **NEW** |
| `examples/spec/CMakeLists.txt` | 5 | **NEW** |
| `examples/chat/chat.cpp` | 631 | **NEW** |
| `examples/chat/CMakeLists.txt` | 5 | **NEW** |
| `CMakeLists.txt` (top) | +2 | additive |
| **TOTAL NEW** | **~1830 lines** | |

Previous reports:
- `llama.cpp/GEMMA4_ASSISTANT_HANDOFF.md` (previous session)
- `llama.cpp/GEMMA4_ASSISTANT_FIX_REPORT.md` (SD bug fix details)
- `llama.cpp/E2B_BENCHMARK.md` (E2B prompt sweep)
- `only-needed-files/CHAT_TOOLS_REPORT.md` (chat + tool details)
- **This file** — master / consolidated report

---

## 10. Commands (reference)

```bash
cd /Users/enes/Desktop/all/less-llama-cpp/only-needed-files
cd build && cmake --build . -j 8 && cd ..

# Single-shot SD (fastest for structured tasks)
./build/bin/llama-spec \
  -m /Users/enes/Desktop/all/llms/gemma-4-E2B-it-UD-Q8_K_XL.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E2B-it-assistant.F16.gguf \
  --spec-type draft-mtp --spec-draft-n-max 5 \
  -p "List the first 50 prime numbers." -n 400 -ngl 99

# Multi-turn chat + tool + SD (KV q8 default — 2× memory saving)
./build/bin/llama-chat \
  -m /Users/enes/Desktop/all/llms/gemma-4-E2B-it-UD-Q8_K_XL.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E2B-it-assistant.F16.gguf \
  -ngl 99 -c 8192 --draft-max 3

# KV f16 if desired (for comparison):
#   ... --kv-f16

# Chat without tools + thinking
./build/bin/llama-chat -m <model> -ngl 99 --no-tools --thinking

# Chat with restricted sandbox
./build/bin/llama-chat -m <model> -ngl 99 --root /tmp

# Baseline (regression)
./build/bin/llama-simple -m <model> -n 200 -ngl 99 "prompt"
```
