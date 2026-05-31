# Gemma 4 Assistant — speculative decoding in llama.cpp · FULL HANDOFF

Last updated: 2026-05-26. This is the complete, self-contained record. Read after `/clear`
to resume with zero context loss. **Read SESSION 2 first — it supersedes parts of the original
(below) where they conflict.**

---

# ===== SESSION 2 UPDATE (2026-05-26) =====

## GOAL (unchanged)
Speed up Gemma-4-E4B via **speculative decoding** with Google's `gemma4_assistant` (78M
EAGLE/MTP draft head). Lossless (verify guarantees output identity). Currently builds, runs,
LOSSLESS — but **draft acceptance still = 0** (no speedup yet).

## BIG WIN THIS SESSION: found the dominant forward bug (×16 embed scale)
`Gemma4TextModel` applies the `sqrt(hidden)` embedding scale **only when it builds embeddings
from `input_ids`**. The assistant feeds `inputs_embeds` **directly** (the pre_projection
output) → `embed_tokens` is bypassed → **NO scaling**. Our draft graph multiplied by
`sqrt(256)=16` anyway. Because RMSNorm is scale-invariant, the ×16 only inflated the residual
stream so the raw input dominated every layer's output → confident garbage.
**Fix:** removed the `ggml_scale` in `src/models/gemma4_assistant.cpp`. Result: draft-forward
cosine vs the reference jumped **0.49 → 0.94** (remaining gap = Q8_0/Q6 quantization).

## HOW WE VERIFIED (reproducible harness — use this to continue)
We now run the **authoritative reference** and compare numerically:
- venv: `/tmp/g4a_venv` (torch 2.12 + transformers 5.8.1; transformers has `gemma4_assistant`
  AND `gemma4` native — no download needed for the modeling code).
- Reference HF dir (assistant): `/Users/enes/Desktop/all/less-llama-cpp/gemma4-assistant/`
  (config.json, modeling_gemma4_assistant.py, configuration_gemma4_assistant.py, README.md,
  **model-2.safetensors** = full assistant weights, model.safetensors.index.json, tokenizer).
- Original TARGET HF dir: `/Users/enes/Desktop/all/less-llama-cpp/gemma4/` — **BUT THIS IS E2B**
  (text hidden=1536, 35 layers, kv_shared=20). The assistant is **E4B** (backbone 2560).
  **MISMATCH — cannot pair** (pre_projection needs 2*2560=5120; E2B gives 2*1536=3072).
  Files have non-standard suffixes (config-2.json, model-3.safetensors). Need original
  **google/gemma-4-e4b-it** instead.
- llama.cpp side dumps the exact draft inputs/outputs to `/tmp/g4a_llama_*.bin` via env
  `G4A_DUMP=1` (concat, kfull/vfull/kswa/vswa, logits, feat). Layout of K/V = `[tok][head][dim]`.
- Python compare scripts (in `/tmp`, recreate if gone): `g4a_ref.py` (feed llama dumps to
  reference, compare feature cosine + top tokens), `g4a_probe.py` (per-layer), `g4a_remap.py`
  (canonical-vs-ordered table check).
- Run llama dump:
  `G4A_DUMP=1 ./build/bin/llama-cli -m <E4B target gguf> -md <assistant gguf> --spec-type
   draft-mtp -p "The capital of France is" -n 4 --temp 0 -st -ngl 0`
  then `/tmp/g4a_venv/bin/python /tmp/g4a_ref.py`.

## BUGS SOLVED THIS SESSION (all in the work tree, all compile)
1. **Incomplete context (acc had 4 of 20 tokens).** Root cause: SWA checkpoint logic splits
   prefill (server-context.cpp:2821 → 16+4); AND the hidden tap was emitted from the
   `inp_out_ids`-pruned tensor, so `n_outputs=0` bulk-prefill produced an empty hidden →
   `process()` rejected the whole chunk ("kv tap mismatch"). **Fix:** in `src/models/gemma4.cpp`
   defer the `inp_out_ids` prune until AFTER the post-norm hidden tap (skip prune when
   `assistant_kv_tap`, then prune just before lm_head). Now `acc_nkv=20` (full prompt).
2. **token_ordering tensor never loaded.** `LLM_TENSOR_MTP_TOKEN_ORDERING` was mapped to
   `GGML_OP_NONE`, which the loader skips as "unused" (llama-model-loader.cpp:1102 returns null).
   **Fix:** `src/llama-arch.cpp` → map it to `GGML_OP_GET_ROWS` (ggml supports i32 source). Now
   loads (n_ord=262144). NOTE: needed only for the centroid head (clustering), see learning #3.
3. **EAGLE shift missing.** Draft used `pending_tok` (last prompt token). Correct EAGLE input is
   `(embed(t_N), hidden_{N-1})` where `t_N` = the just-sampled token = `dp.id_last`. **Fix:**
   `common/speculative.cpp` → `cur_tok = dp.id_last`.
4. **×16 embed scale (THE big one).** See "BIG WIN" above. Fix in `gemma4_assistant.cpp`.
5. **Q position wrong.** Candidate generator pins `position_ids = seq_len - 1`, constant across
   the draft round. We used `n_kv_full` (=seq_len). **Fix:** `pos = acc.n_kv_full - 1`.
6. **Wrong output remap.** We mapped `id = token_ordering[argmax]`. But the embed/lm_head table
   is **CANONICAL** (verified: `cos(mine,ref_dense_canonical)=0.895` > `cos(as-ordered)=0.763`).
   token_ordering is ONLY a cluster map. **Fix:** `id = ord_id` (no remap).

## KEY LEARNINGS FROM THE REFERENCE (correct the session-1 assumptions)
- **inputs_embeds is NOT scaled** (modeling_gemma4.py: scale only on the input_ids path).
- **shared_kv_states used AS-IS** — no re-rope, no re-norm (modeling_gemma4.py:1234-1238). The
  target's K/V are already roped+k-normed. Our tap (post-rope, post-k_norm) matches.
- **Attention scaling = 1.0** (`self.scaling = 1.0`), Q gets q_norm then rope. Matches us.
- **Logit head = centroid `masked_embedding`** (modeling_gemma4_assistant.py:42-87): project to
  2048 centroids, take top-32, compute logits ONLY for those clusters' tokens (32*128=4096),
  **mask the rest**; output is in **canonical** space (scatter to canonical ids). Confirmed
  dense-lm_head top tokens ≠ centroid top tokens → the centroid masking is load-bearing, not an
  optimization. WE STILL USE DENSE lm_head (TODO).
- **token_ordering** = `view(2048,128)` cluster→canonical map (for the centroid head). It does
  NOT reorder the embedding table. Table is canonical.
- **Candidate generator** = `SinglePositionMultiTokenCandidateGenerator`
  (transformers/generation/candidate_generator.py:1230+):
  - `last_hidden_state = model_outputs.hidden_states[-1]` (target's last-layer hidden, sliced to
    the last validated token). **OPEN: is hidden_states[-1] pre- or post-final-norm?** We tap
    POST-norm. If the assistant wants pre-norm, that's a remaining bug. VERIFY in python.
  - `last_token_embedding = target_model_input_embeddings(last_token_id)`, `last_token_id =
    input_ids[:,-1:]` (the just-sampled token). raw embed, no scale.
  - `inputs_embeds = cat([embed, hidden])`, `position_ids = [[seq_len-1]]` (constant).
  - `shared_kv_states = model_outputs.shared_kv_states` — the TARGET emits this via
    `store_full_length_kv` (gemma4 modeling:1252-1253): the **last K/V-computing layer of each
    type** stores its (roped, k-normed) K/V. Our tap = last has_kv full/swa layer. Should match —
    VERIFY the exact layer index.
  - chain: `last_token_id = argmax(logits)`, `last_hidden_state = outputs.last_hidden_state`
    (= assistant post_projection output, 2560-d). Matches our chaining.

## WHAT WE COULD NOT DO / REMAINING WORK (the blockers)
1. **End-to-end acceptance still 0.** Two causes still open:
   a. **No matching E4B target.** Uncensored-Q6 is a finetune (the f32 reference fed its K/V also
      fails — 'Paris' masked at rank 4096). The "original" downloaded is **E2B (wrong size)**.
      → NEED **google/gemma-4-e4b-it** (f16 gguf for llama.cpp target; and/or HF safetensors to
      run the full python pipeline target→K/V→assistant and confirm it predicts the right token).
   b. **Centroid head not implemented** (still dense lm_head). Confirmed needed.
2. **Verify the assistant "hidden" input = pre- or post-final-norm** (hidden_states[-1]). We tap
   post-norm. Quick to check in python with any gemma4_text model (output_hidden_states=True,
   compare hidden_states[-1] vs last_hidden_state).
3. **Verify K/V tap layer** matches `store_full_length_kv` (last has_kv full + last has_kv swa).
4. **Metal crash on the hidden tap** (original #11) — still CPU-only (`-ngl 0`).
5. **Cleanup before commit:** lots of debug scaffolding is still in the tree (see below).

## DEBUG SCAFFOLDING CURRENTLY IN THE TREE (remove before any commit)
- `common/speculative.cpp`: `LOG_ERR` blocks in `process()`/`draft()` (G4A ...), top-8 candidate
  dump, TARGET-rank dump, `G4A_DUMP` file dumps, env overrides `G4A_POS`, `G4A_ESCALE`,
  `G4A_SWAP`; the `dbg` counter increments.
- `src/models/gemma4_assistant.cpp`: `G4A load:` fprintf, env toggles `G4A_ASCALE`, `G4A_NOATTN`,
  `G4A_NOROPE`, `G4A_PROBE` + `dbg_probe` vector + the probe fprintf.
- `src/llama-context.cpp`: `G4A tap-copy:` `LLAMA_LOG_ERROR` in the decode tap section.
- `src/llama-graph.cpp`: `G4A pos set_input:` fprintf.
The REAL fixes to KEEP: gemma4.cpp deferred prune; gemma4_assistant.cpp removed ×16; llama-arch.cpp
GET_ROWS; speculative.cpp `cur_tok=dp.id_last`, `pos=n_kv_full-1`, `id=ord_id`.

## IMMEDIATE NEXT STEPS (in order)
1. Get **google/gemma-4-e4b-it** (E4B). Run full python pipeline (target.generate with
   assistant_model, hook assistant.forward) → confirm the assistant proposes the correct tokens.
   This isolates target-mismatch from any residual bug.
2. Verify pre/post-norm of the assistant hidden input; fix the tap if needed.
3. Implement the **centroid head** in the draft graph (or host-side in `draft()`): centroids
   mat-vec → top-32 → gather cluster token ids via token_ordering → masked logits → argmax
   (canonical). Validate numerically vs the reference (`out.logits` from masked_embedding).
4. Convert the E4B target to gguf, run the §2 acceptance command, expect `#acc tokens > 0`.

# ===== END SESSION 2 UPDATE — original session-1 handoff follows =====

---

## SUMMARY (read first)

**Goal:** speed up the big Gemma-4-E4B model via **speculative decoding** using Google's official
`gemma4_assistant` (79M, EAGLE/MTP-style "draft" head). The small model proposes many tokens, the
big model verifies them in parallel → lossless speedup.

**Current status:**
- The whole pipeline is built, compiles, runs end-to-end, and is **LOSSLESS** (output is
  token-identical to running without the draft). The target's verify step guarantees the output
  no matter how bad the draft is.
- **THE ONLY PROBLEM: draft acceptance = 0%.** The draft produces garbage tokens → no speedup.

**What the problem was (root cause, found this session):** the assistant's own forward produces
garbage because (1) the target hidden fed to the first draft was ZERO (fixed), and (2) **the real
root cause:** the assistant only sees the **last ~4 tokens** of the prompt as context — the server
"bulk-prefills" most of the prompt without it reaching my accumulation — plus the Q position
alignment is wrong. So the assistant attends to incomplete, wrongly-positioned context → a
confidently wrong token.

**What's solved:** the assistant architecture, model loading, draft graph, cross-attention K/V
transport, the spec loop, and losslessness all work. The `hidden=0` bug, several
segfaults/asserts, graph reuse, flash mask, etc. are fixed (detailed below).

**Next plan (the blocker):** give the assistant the target's **FULL** K/V (all prompt positions)
and align Q to absolute positions. Two routes: (A) read the target KV cache directly, (B)
accumulate the tap persistently in `llama_context::decode`. Then check: first draft token ==
target's first generated token.

---

## 1. GOAL & BACKGROUND

**Speculative decoding:** a small "draft" model proposes several tokens; the big "target" model
verifies them all in one forward pass. Accepted tokens are emitted; the first rejected one is
resampled from the target. Output distribution is provably identical to plain target decoding
(**lossless**), but fewer target forward passes → speedup (if acceptance is good).

**`gemma4_assistant`:** Google's purpose-built draft head for Gemma 4. Unlike a classic draft
(a smaller standalone LM), it is an **EAGLE/MTP-style head**: a 4-layer transformer with **no
K/V weights of its own** — every layer cross-attends to the **target model's** already-computed
K/V. It takes as input the concatenation of the target's token embedding + hidden state, and
chains its own output feature for multi-token drafting.

**Why this is hard in llama.cpp:** classic spec decoding assumes the draft is independent.
Here the draft must read the target's K/V **across two model contexts** — a mechanism that did
not exist in llama.cpp and had to be built.

---

## 2. SETUP

### Files / paths
- Target: `/Users/enes/Desktop/all/llms/Gemma-4-E4B-Uncensored-Q6.gguf` (arch `gemma4`, 42 layers, n_embd=2560)
- Draft:  `/Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf` (arch `gemma4_assistant`, 4 layers, n_embd=256, 78.78M)
- **Work tree (edit + build HERE):** `/Users/enes/Desktop/all/less-llama-cpp/llama.cpp`
- Trimmed tree (NOT used): `/Users/enes/Desktop/all/less-llama-cpp/only-needed-files` (no `common/`, no build)
- Build dir: `/Users/enes/Desktop/all/less-llama-cpp/llama.cpp/build`
- Plan: `/Users/enes/.claude/plans/continue-sleepy-kite.md`
- The transformers reference the user supplied: `modeling_gemma4_assistant.py`,
  `configuration_gemma4_assistant.py` (the **candidate generator** is NOT in these — that gap
  is why exact input assembly/positions are partly inferred).

### Build + run
```
cd /Users/enes/Desktop/all/less-llama-cpp/llama.cpp/build && cmake --build . --target llama-cli -j 8
cd /Users/enes/Desktop/all/less-llama-cpp/llama.cpp
# NOTE: Metal currently crashes on the hidden tap → use CPU (-ngl 0) for now.
./build/bin/llama-cli -m /Users/enes/Desktop/all/llms/Gemma-4-E4B-Uncensored-Q6.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf \
  --spec-type draft-mtp -p "Count from 1 to 10:" -n 24 --temp 0 -st -ngl 0 -v </dev/null 2>&1 \
  | grep "statistics draft"
```
- `-st` = single turn (else interactive mode spams `> ` to EOF → 100s of MB).
- Acceptance = `#acc tokens` in `statistics draft-mtp`. Currently 0.
- Lossless check: run with vs without `-md`; greedy output must be token-identical (it is).
- `-no-cnv` unsupported by llama-cli; `llama-completion` doesn't accept `-md`.

---

## 3. THE MODEL (from transformers v5.9.0 reference)

`Gemma4AssistantForCausalLM.forward`:
1. `inputs_embeds = pre_projection(concat[embedding, hidden_states])`; `pre_projection: Linear(2*backbone=5120 → 256)`.
   - `embedding` = target `embed_tokens(token)` (raw, 2560-d).
   - `hidden_states` = target `last_hidden_state` (**POST final norm**, 2560-d). Both for the **last seen token**.
2. `self.model` = 4-layer `Gemma4TextModel`, `num_kv_shared_layers = num_hidden_layers` → every
   layer cross-attends the target's shared K/V (`shared_kv_states = {"full_attention":(K,V),
   "sliding_attention":(K,V)}`, each = the target's **last layer of that type**).
   `Gemma4TextModel` **scales inputs_embeds by sqrt(hidden)≈16**. **constant position_ids**.
   bidirectional masks (full+SWA, kv-axis flips); q_len==1 = full attention.
3. `last_hidden_state` (post-norm) → `post_projection: Linear(256→2560)` = the **feature**,
   chained as the next step's `hidden_states`.
4. logits: `masked_embedding(last_hidden, lm_head.weight)` (centroid sparse head) else `lm_head`.
   `use_ordered_embeddings=True` → the embedding/lm_head table is **reordered** (cluster-grouped);
   `token_ordering[ordered_pos] = canonical_token_id`. lm_head tied to `model.embed_tokens` (256-d).

### Draft gguf facts (verified by dump)
- 4 layers, n_embd=256, n_head=4, n_head_kv=2, n_ff=2048, vocab=262144, rms_eps=1e-6.
- **Per-layer-type head dims:** sliding (blk 0,1,2) head_dim=256; full (blk 3) head_dim=512.
- SWA pattern `[T,T,T,F]`. rope: full base 1e6 dim 512; swa base 1e4 dim 256.
- `mtp.` tensors: `pre_projection`[5120→256], `post_projection`[256→2560], `centroids`[256→2048],
  `token_ordering` i32[262144]. **No `attn_k`/`attn_v`.**
- KV: `gemma4_assistant.{n_centroids=2048,centroid_top_k=32,n_embd_backbone=2560,use_ordered_embeddings=true}`.
- Target KV geometry matches exactly (2 kv heads, head_dim 512 full / 256 swa, backbone 2560, vocab 262144).

---

## 4. IMPLEMENTATION JOURNEY (phase by phase)

**Phase 0 — gguf inspection.** Dumped both models' KV/tensors; locked names, shapes, dims;
verified target↔draft geometry match.

**Phase A — arch plumbing.** Registered `LLM_ARCH_GEMMA4_ASSISTANT`, the 4 custom KV keys, the
4 `mtp.*` tensor enums/names, hparams fields. (The fork uses a flat global tensor-name map, so
no per-arch tensor map needed.)

**Phase B — model load.** New `llama_model_gemma4_assistant` class; `load_arch_hparams` /
`load_arch_tensors`; factory + NEOX rope. Verified: model loads, all 51 tensors map, hparams
correct (78.78M params). Build green.

**Phase C — draft graph + cross-KV carrier.** New `build_arch_graph`: pre_projection → 4 Q-only
layers cross-attending the shared K/V via `build_attn_mha` → norms/FFN → output_norm →
post_projection feature → lm_head. New `llama_assistant_shared_kv` carrier +
`llm_graph_input_assistant_kv` graph input (4 K/V + 2 masks), threaded through
params/context (mirrors T5's `llama_cross`). `n_embd_inp()` returns 2*backbone;
`create_memory`=nullptr (no self KV cache); head dims from hparams.

**Phase D — target K/V tap + spec loop.** Tap the target gemma4 graph's last full/swa layer
K/V (gated flag) → result tensors → context host buffers → `llama_get_assistant_kv_tap`.
New `common_speculative_state_draft_gemma4_assistant` (selected when draft arch ==
`gemma4_assistant`): accumulates target K/V across decodes, feeds `concat[embed,hidden]`,
constant positions, chains the post_projection feature. Helpers `llama_model_get_token_embd`,
`_get_output_norm`, `_get_token_ordering`.

**Phase E — verify.** End-to-end runs, output **lossless**. Acceptance = 0 (open).

---

## 5. ALL BUGS FOUND → ROOT CAUSE → FIX

1. **`unknown model architecture 'gemma4_assistant'`** → added the arch end-to-end (Phase A/B).
2. **Tap copy placed in `llama_context::encode` (dead path).** decode never calls encode → tap
   always 0. **Fix:** moved to the `decode` per-ubatch output extraction.
3. **MTP context type returned null → segfault.** `llama_init_from_model` returned nullptr for
   `LLAMA_CONTEXT_TYPE_MTP` when `nextn_predict_layers==0` (true here) → null ctx_dft → crash.
   **Fix:** relaxed the check to allow `LLM_ARCH_GEMMA4_ASSISTANT`.
4. **`begin()` wiped the prefill K/V.** `begin()` fires AFTER prompt prefill; it cleared `acc`
   → draft saw n_kv=1 → garbage. **Fix:** `begin()` is a no-op.
5. **Graph reuse baked a stale dummy n_kv=1.** Draft graph reserved once with n_kv=1 and reused;
   `set_input` copied only 1 position. **Fix:** `set_assistant_shared_kv` sets
   `sched_need_reserve=true`; confirmed graph rebuilds with real n_kv.
6. **flash-attn needs an F16 mask.** `ggml_flash_attn_ext` aborted on F32 mask. **Fix:** `_cnv`
   F16-cast masks in `build_inp_assistant_kv`.
7. **Head dims from an empty carrier at reserve → `mul_mat` shape assert.** **Fix:** take head
   dims from hparams, only n_kv from the carrier.
8. **Draft input width.** Input is a 5120-wide concat, not a token. **Fix:** `n_embd_inp()`
   returns 2*backbone; feed via `batch.embd`; allocate a dummy `tokens` input so `set_input`
   never derefs null.
9. **Feature read-back dimension.** post_projection feature is 2560 but `n_embd_out` was 256.
   **Fix:** set `n_embd_out_impl = n_embd_backbone`.
10. **`|hid|=0` — target hidden was zero (THIS SESSION).** Enabling `embeddings_pre_norm` after
    context creation does NOT allocate the `embd_pre_norm` buffer → `get_embeddings_pre_norm_ith`
    returns zeros. **Fix:** tap the target's **post-norm hidden** (`res->t_embd`) through the same
    host-buffer tap as K/V (`t_kv_tap_hidden`); spec loop reads it directly. Confirmed `|hid|≈112`.
11. **Metal crash on the hidden tap (THIS SESSION, OPEN).** `ggml-metal GGML_ASSERT(buf_dst)`
    in `ggml_metal_cpy_tensor_async` reading `t_kv_tap_hidden` (a `ggml_scale`+`set_output` node
    with no readable Metal buffer). **Workaround:** CPU (`-ngl 0`). **TODO:** make Metal-readable.

### Fixes applied from the reference (correct, kept; did NOT alone fix acceptance)
- **sqrt(hidden) input scaling** after pre_projection.
- **post-norm target hidden** (was pre-norm).
- **`token_ordering` remap** `real_id = token_ordering[argmax]` (ordered→canonical). Unverified
  (blocked by the context bug below); kept.
- Tried & reverted: concat order swap (doc order `[embedding,hidden]` is correct); reading
  post-norm via `llama_get_embeddings` (broken prefill indexing).

---

## 6. THE ROOT CAUSE (current blocker) — incomplete prompt context

After fixing `|hid|=0`, inputs are sane (`|embd|≈1, |hid|≈112, |kfull|≈100`) but the draft's
top token is still **confidently garbage** (`p=1.0`). Magnitude dumps revealed:

- For a ~10-token prompt, `process()` only sees the **last ~4 tokens** (`G4A proc n_tok=4,
  acc 0→4`). The server **bulk-prefills most of the prompt via `llama_decode(ctx_tgt)` without
  it reaching my accumulation** (`common_speculative_process` covers only the tail batch). So
  the draft cross-attends to the prompt **TAIL only** → wrong/insufficient context.
- **Position mismatch:** Q is roped at `pos = n_kv_full` (acc-relative, e.g. 4) while the tapped
  K were roped at their **real** prompt positions (e.g. 6-9). Relative distances are wrong →
  garbage attention. (This compounds because `acc` starts mid-prompt.)

The target's own KV cache is complete (it correctly says "Paris"); only my host accumulation is
partial. So the host-accumulation-via-`process()` design is fundamentally incomplete.

---

## 7. THE PLAN FORWARD (do next — this is the blocker)

Capture the target's **FULL** K/V (all prompt positions), and align Q positions to **absolute**
positions. Two routes:

- **(A) Read the target's KV cache directly** at draft time. It always has the full, correct
  context and handles rollback itself. Hard part: iSWA cache internals (`v_trans`, typed, two
  sub-caches; `get_k(ctx,il,n_kv,sinfo)`), extracting the last full + last swa layer K/V into
  `[head_dim, n_head_kv, n_kv]`. **Cleanest/correct.**
- **(B) Persistent tap accumulation in `llama_context::decode`** (the tap fires for bulk prefill
  too): accumulate into a positional buffer 0..N; trim on rollback in `accept()`. Avoids cache
  internals; needs careful rollback bookkeeping.

Then set Q `position_ids` to the **absolute** position (= acc index, once acc starts at 0).
Verification after the fix:
1. First draft token (after prefill) SHOULD equal the target's first generated token.
2. `#acc tokens` > 0; tokens/s higher than baseline (run without `-md` to compare).
3. If still wrong with full+aligned context → THEN suspect cross-attention math, the constant
   position value, or `token_ordering` direction; dump per-layer `kqv_out` / `result_norm`.

Separately: fix the **Metal hidden-tap buffer** (#11) so the fast path works (CPU works now).

---

## 8. FILE-BY-FILE CHANGES (all in `llama.cpp/`, all compile)

- `src/llama-arch.{h,cpp}` — arch enum+string, 4 KV keys, 4 `mtp.*` tensor enums/names/infos.
- `src/llama-hparams.h` — fields; `src/llama-hparams.cpp` — `n_embd_inp()`=2*backbone.
- `src/llama-cparams.h` — `assistant_kv_tap` flag (init false in `llama-context.cpp`).
- `src/models/models.h` — `llama_model_gemma4_assistant` (+ `graph`).
- `src/models/gemma4_assistant.cpp` (NEW) — load + graph (pre_proj → sqrt scale → 4 cross-attn
  layers → norms/ffn → output_norm → post_proj feature → lm_head); sets `n_embd_out_impl=backbone`.
- `src/models/gemma4.cpp` — target tap: last full/swa K/V + post-norm hidden (`t_kv_tap_hidden`),
  gated by `cparams.assistant_kv_tap`.
- `src/llama-model.cpp` — factory case; NEOX rope; `create_memory`=nullptr for the arch;
  `llama_model_get_token_embd`, `_get_output_norm`, `_get_token_ordering`.
- `src/llama-graph.{h,cpp}` — `llama_assistant_shared_kv`, `llm_graph_input_assistant_kv`,
  `build_inp_assistant_kv`, result tensors `t_kv_tap_{k,v}_{full,swa}` + `t_kv_tap_hidden`,
  threading through params/context.
- `src/llama-context.{h,cpp}` — `assistant_kv` member; tap host buffers (`kv_tap_*` incl.
  `kv_tap_hidden`) + per-ubatch copy in `decode`; `set_assistant_shared_kv` (sets
  `sched_need_reserve`), `set_assistant_kv_tap`, `get_assistant_kv_tap(...&hidden...)`;
  MTP ctx-type relax.
- `src/llama-ext.h` — API decls.
- `common/speculative.cpp` — `common_speculative_state_draft_gemma4_assistant` + registration.

---

## 9. SPEC LOOP INTERNALS (`common_speculative_state_draft_gemma4_assistant`)

- **ctor:** `llama_set_embeddings_pre_norm(ctx_tgt,true,false)` (forces all-token outputs so the
  hidden tap has every token), `llama_set_assistant_kv_tap(ctx_tgt,true)`,
  `llama_set_embeddings(ctx_dft,true)` (read the feature); loads `tgt_output_norm_w` (now unused)
  and `tok_ordering`.
- **process():** reads tap (K/V + post-norm hidden) via `llama_get_assistant_kv_tap`; prefill
  (`!in_generation`) commits all to `acc`; generation stages for `accept()`.
- **accept(n):** commits n+1 staged rows; updates `pending_{feat,tok,pos}`.
- **draft():** sets `assistant_kv` from `acc`; builds `concat[embed(tok), feat]` into `batch.embd`;
  constant `pos=n_kv_full`; decodes ctx_dft 1 tok/step; remaps `id=token_ordering[argmax]`;
  chains `feat = post_projection` via `llama_get_embeddings_ith(ctx_dft)`.
- **begin():** no-op (fires after prefill).

### Debug scaffolding (currently removed; re-add to iterate)
Counter-gated `LOG_ERR` in `process()` (`n_tok`, tap, `acc0`, `|hid|`) and `draft()` (`|embd|`,
`|hid|`, `|kfull|`, `|feat_out|`, top token+piece). Helper `dbg_l2`. Build `llama-cli`, run the
CPU command in §2, grep `G4A`.

---

## 10. STATUS
Pipeline complete, builds clean, **lossless**. Acceptance 0 (blocked on §6/§7). Metal crashes on
the hidden tap (#11) → use CPU. **No git commit** — all working-tree edits in `llama.cpp/`.
