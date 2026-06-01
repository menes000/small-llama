# SSD (Saguaro) Implementation Report

Paper: **Speculative Speculative Decoding** — arXiv 2603.03251v3 (Kumar/Dao/May).

File: `examples/spec/spec.cpp` (438 → 649 lines).

## 1. What Was Done

### 1.1 CLI Flags
`struct args` + `parse_args` extended:
- `--ssd-fan-out B` — total alternative budget (default `1` = SSD off).
- `--ssd-r R` — geometric fan-out exponent (default `1.0`, Theorem 12).
- `--ssd-sampling-C C` — Saguaro sampling downweight (default `1.0` = passive; warns in greedy mode).

### 1.2 Helper Functions
- `masked_topf(...)` — generalization of `masked_argmax` to top-F. Ranks tokens inside top-K centroid clusters by dense_logits, returns F. `out_tokens[0]` = old argmax.
- `masked_argmax(...)` is now a wrapper around `masked_topf(..., F=1)`.
- `geometric_fanout(B, K, a_p, r)` — Theorem 12 formula:
  ```
  F_k = round(F_0 · a_p^(k/(1+r)))    each F_k >= 1, sum(F_k) <= B
  ```
  `a_p` (running accept rate) is `0.8` for the first round, then `total_accepted / total_drafted`.

### 1.3 Speculation Cache
```cpp
struct cache_entry {
    llama_token alt_bonus;          // "if bonus were this..."
    llama_token next_tok;            // pre-speculated next draft token
    std::vector<float> next_feat;    // ready-chained feature for round T+1
};
std::vector<std::vector<cache_entry>> spec_cache;   // spec_cache[k] = alts at position k
```

Reset at start of each round. Cache hit at `start_k` skips the first draft step.

### 1.4 Draft Loop Restructure
Old (lines 309–337): single pass, per-step inline decode + masked_argmax.

New: **two phases** — to fix the pollution bug (see below).
1. **Greedy phase**: K-step pure greedy chain. Per step, `(in_tok, in_feat, alts)` is saved.
2. **SSD post-pass**: after greedy chain completes, saved `in_*` replay alt decodes. Written to cache.
3. **Restore**: after post-pass, last greedy step re-decoded → K/V[draft_pos] matches vanilla path.

### 1.5 Post-verify Cache Lookup
After verify (`n_accept`, `next_pending` known):
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
Hit → next round's k=0 draft step skipped (1 decode saved).

### 1.6 Stats Line
Additional `[spec]` output:
```
SSD: hits=N misses=N hit_rate=N.N% extra_draft_decodes=N steps_saved=N net_extra=N
```

## 2. Problems Encountered + Solutions

### 2.1 BUG #1: Inline alt decode → accept rate dropped
**Symptom**: With B=10 on primes prompt, accept rate 82.5% → 69.6%. Output byte-identical (lossless) but draft chain broken — subsequent rounds degraded.

**Hypothesis**: Doing alt decode IMMEDIATELY after each greedy step corrupts ctx_d's K/V[draft_pos]. Next greedy step reads corrupted K/V.

**Fix attempt #1 — Post-pass**:
Alt decodes moved to AFTER the full greedy chain. `step_state{in_tok, in_feat, alts}` saved per step, replayed in a separate loop.

**Result**: ❌ Accept rate still 69.6%. Problem elsewhere.

### 2.2 BUG #2: Alt decode corrupts K/V → affects NEXT ROUND
**Hypothesis**: After post-pass, K/V[draft_pos] = last alt's K/V. Next round's draft self-attention reads this corrupted position.

**Fix attempt #2 — Restore decode**:
After post-pass, re-decode the last greedy step (in_tok[K-1], in_feat[K-1]). K/V[draft_pos] returns to vanilla path value.

**Result (partial success)**:
- B=7, 8 → accept rate ✓ preserved (82.5%)
- B=10+ → accept rate ✗ still drops (69.6%)

**Interpretation**: Restore fixes K/V at a single position but multiple alt decodes (when F_k >= 3, 2+ alts per pos) may corrupt non-K/V internal state in ctx_d (hidden counter, n_outputs, etc.). Debugging MTP context internal state beyond this version's scope.

### 2.3 BUG #3 (false positive): LSP "headers not found"
**Symptom**: clang LSP reported `'llama.h' file not found` + std type errors after each edit.
**Diagnosis**: LSP include path not configured. cmake build succeeded → diagnostics not real. Ignored.

### 2.4 Design decision: Saguaro sampling σ_F,C scaffold-only
In greedy decode mode, top-F downweight shifts argmax → breaks losslessness (doesn't mismatch target greedy but changes draft output). Flag added, code path missing, warning emitted if greedy + C<1. Will activate when temperature sampling added.

### 2.5 Design decision: threading skip
Paper assumes `T_p < 1` — draft must be on a separate device to hide inside target verify wall-time. Single Mac Metal = shared GPU queue → `std::thread` overlap gives no gain. Documented, skipped.

## 3. Validation Results

Setup: `gemma-4-E2B-it-UD-Q8_K_XL` (target) + `gemma-4-E2B-it-assistant.F16` (draft), `--draft-max 5`, Mac Metal.

### 3.1 Determinism check
Two consecutive runs with same flags → identical stats (rounds, accept_rate, hits) ✓

### 3.2 Losslessness check
`diff baseline.out ssd_on.out` for all B values → 0 difference ✓

### 3.3 Fan-out sweep — primes prompt
| B | t/s | rounds | accept% | hits | extra | saved |
|---|---|---|---|---|---|---|
| 1 (off) | 110.7 | 24 | 82.5 | 0 | 0 | 0 |
| 6 | 105.4 | 24 | 82.5 | 0 (path entered) | 0 | 0 |
| 7 | 97.0 | 24 | 82.5 | 1 | 56 | 1 |
| 8 | 93.1 | 24 | 82.5 | 1 | 95 | 1 |
| 10 | 80.0 | 27 | 69.6 ⚠️ | 7 | 148 | 7 |
| 15 | 67.5 | 28 | 67.1 ⚠️ | 10 | 274 | 10 |

### 3.4 Prompt type comparison (B=10)
| Prompt | t/s base → SSD | accept% base → SSD | hit% |
|---|---|---|---|
| "first 30 primes" | 110.7 → 80.0 | 82.5 → 69.6 ⚠️ | 41 |
| "count 1 to 50" | 90.9 → 76.0 | 66.1 → 66.1 ✓ | 25 |
| "write short story dragon" | 40.3 → 34.7 | 15.7 → 15.7 ✓ | 23 |

**Hit rate aligns with paper Fig.3/4 pattern**: structured > creative.

## 4. Conclusion

**Works correctly**:
- Algorithm faithful to paper (cache, geometric fan-out, hit lookup, stats).
- Lossless (greedy target output byte-identical).
- Stats accurate (p_hit, extra_decodes, steps_saved measured correctly).

**No gain on single GPU**:
- net_extra = extra_decodes - steps_saved → always positive (>> 0).
- Paper's 30%-5× speed requires **SEPARATE DEVICE** assumption.
- Intentionally "pragmatic" choice, expected result.

**Known limitation**:
- High B + some prompts (e.g. primes B=10) still drop accept rate. MTP context K/V external state not fully restorable. Future iteration: inspect `src/models/gemma4.cpp` MTP path.

**Future work** (not in this version):
- Temperature sampling + Saguaro σ_F,C activation.
- Threading: if draft and target on separate backends (Metal GPU + CPU NEON), partial overlap possible.
- Fallback speculator (paper §4.3) — second fast draft model required.
