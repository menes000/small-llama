# llama-spec Usage Guide

`examples/spec/spec.cpp` — Gemma4 speculative decoding (SD) + optional Saguaro SSD (paper arXiv 2603.03251v3).

## Defaults (No Flags Given)

```
-ngl 99           target Metal GPU (full)
-ngl-draft 0      draft CPU NEON         ← new default
--ssd-fan-out 1   SSD OFF (pure SD)      ← default
--ssd-async       off                    ← default
```

→ **Default mode = SD + CPU draft + Metal target.** Fastest for E4B Q8 pair.

## Standard Commands

### 1. Default (fastest E4B Q8)
```bash
./build/bin/llama-spec \
  -m  /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf \
  -p "list the first 40 prime numbers" -n 150 --draft-max 5
```
No SSD, pure SD. Draft CPU, target Metal. **38 t/s** (primes, average).

### 2. SSD enabled (paper algorithm)
```bash
./build/bin/llama-spec \
  -m  /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf \
  -p "list the first 40 prime numbers" -n 150 --draft-max 5 \
  --ssd-fan-out 7 --ssd-async
```
SSD speculation cache + async overlap active. **~35 t/s** (8% slower than default — post-pass overhead not fully hidden).

### 3. All SSD options
```bash
--ssd-fan-out B      budget (default 1 = off; active when >K, typical 7-15)
--ssd-r R            geometric fan-out exponent (default 1.0)
--ssd-sampling-C C   Saguaro sampling (default 1.0; warns in greedy mode)
--ssd-async          parallel SSD post-pass with target verify (only useful on different backends)
```

## When to Use Each Mode

| Goal | Extra flags | Expected |
|---|---|---|
| **Max speed** | (none) | SD CPU draft, E4B avg 38 t/s |
| Show paper SSD | `--ssd-fan-out 7 --ssd-async` | SSD active, lossless, ~35 t/s |
| Legacy default (SD Metal) | `-ngl-draft 99` | Draft on GPU, E4B 34 t/s |
| SSD without overlap | `--ssd-fan-out 7` | Serial SSD, for measurement |
| Measure hit rate | `--ssd-fan-out 10 --ssd-async` | Higher hit but slower |

## Reading Output

### Banner
```
[spec] backends: target ngl=99 (GPU)   draft ngl=0 (CPU)
[spec] prompt tokens=35 n_predict=150 n_draft_max=5 n_centroids=2048 top_k=32 backbone=2560
[spec] SSD: fan_out_budget=1 r=1.00 sampling_C=1.00 -> OFF (budget <= K, no alternatives)
```

### Stats
```
[spec] emitted=150 in 2.76s = 54.3 t/s | rounds=41 drafted=205 accepted=108 accept_rate=52.7% acc_per_round=2.63
[spec] SSD: hits=0 misses=0 hit_rate=0.0% extra_draft_decodes=0 steps_saved=0 net_extra=0
[spec] timing: verify=64ms post_pass=0ms overlap_region=64ms hidden=-0.0%   async=off draft_backend=CPU
```

Field descriptions:
- `t/s` — emitted tokens / wall time
- `accept_rate` — acceptance rate (draft quality indicator)
- `acc_per_round` — accepted tokens per round
- `hits/misses/hit_rate` — SSD cache hit rate
- `extra_draft_decodes` — extra draft compute done by SSD
- `steps_saved` — draft steps skipped via cache hit
- `verify` — total target Metal verify wall-time
- `post_pass` — total SSD alt-decode wall-time
- `hidden%` — percentage of total time hidden by async overlap

## Lossless Guarantee

Every mode produces identical output (greedy target is deterministic). The following diff must always be empty:

```bash
./build/bin/llama-spec -m T -md D -p "X" -n 100                              > /tmp/sd.out
./build/bin/llama-spec -m T -md D -p "X" -n 100 --ssd-fan-out 7 --ssd-async  > /tmp/ssd.out
diff /tmp/sd.out /tmp/ssd.out  # must be empty
```

## Important Notes

1. **E4B target + E4B assistant** pair — assistant and target MUST match. Mismatched pair gives 0% acceptance.
2. **E2B pair**: SD Metal (both GPU) is still fastest. CPU draft gain only applies to LARGE targets (E4B+).
3. **SSD net gain on Mac M2 Pro**: none — paper's speed claim requires separate GPUs. SSD async is only marginally better than SD Metal (~+3%), slower than SD CPU.

## One-Click Comparison

```bash
TGT=/Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf
DRF=/Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf
P="list the first 40 prime numbers"

echo "=== Default (SD CPU draft) ==="
./build/bin/llama-spec -m $TGT -md $DRF -p "$P" -n 150 --draft-max 5 2>&1 \
  | grep -E "^\[spec\] (emitted|timing)" | head -2

echo "=== SSD enabled ==="
./build/bin/llama-spec -m $TGT -md $DRF -p "$P" -n 150 --draft-max 5 \
  --ssd-fan-out 7 --ssd-async 2>&1 \
  | grep -E "^\[spec\] (emitted|SSD: hits|timing)" | head -3
```
