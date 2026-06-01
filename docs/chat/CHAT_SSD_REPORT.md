# Chat: New Architecture (SSD + CPU draft + async overlap)

`examples/chat/chat.cpp` 858 → 1019 lines. All SSD infrastructure from spec.cpp ported:
- CPU/GPU split per-model (`-ngl-draft`)
- Geometric fan-out cache (`--ssd-fan-out`, `--ssd-r`)
- Saguaro sampling scaffold (`--ssd-sampling-C`)
- `std::async` verify/post-pass overlap (`--ssd-async`)
- Cache hit lookup → seed for next round
- Per-turn stats (hits, hidden%, etc.)

## Defaults

```
-ngl 99           target Metal GPU
-ngl-draft 0      draft CPU NEON          ← new default
--draft-max 3     chat default (spec uses 5)
--ssd-fan-out 1   SSD OFF                 ← default
--ssd-async       off                     ← default
```

→ Default = **SD + draft CPU + target Metal** (fastest for E4B target).

## Test Results (E4B Q8 pair, "list first 30 primes")

### `--draft-max 3` (chat default)
| Mode | t/s | hidden% | hit% |
|---|---|---|---|
| SD Metal (both GPU) | 65.2 | — | — |
| **SD CPU** | **72.8** | — | — |
| SSD serial CPU | 64.5 | 0% | 0 |
| **SSD async CPU** | **72.5** | **18.9%** | 0 |

→ SSD async = SD CPU (overhead fully hidden).
→ hit=0 because K=3 + acc %98.9 → bonus at position K, cache covers up to K-1.

### `--draft-max 5` (better for SSD)
| Mode | t/s | hidden% | timing |
|---|---|---|---|
| SD CPU | 69.8 | — | — |
| **SSD async CPU** | **69.8** | **29.3%** | verify=48ms post=79ms overlap=89ms |

→ Exactly equal. SSD adds zero cost.
→ Hidden 29.3% — better than spec.cpp's 20-22%.

### Output Lossless ✓
4 modes × 1 prompt = 4 runs. All `*.out` byte-identical.

## New Stats Line

When SSD active (B>K), per-turn:

```
[stats] gen=119 tok / 1.71s = 69.8 t/s | prefill=31 tok / 0.23s = 135 t/s | rounds=22 ...
[stats] SSD: hits=0 misses=5 hit=0.0% extra=41 saved=0 | verify=48ms post=79ms overlap=89ms hidden=29.3% async=on draft=CPU
```

Fields:
- `hits/misses/hit%` — speculation cache hit rate
- `extra` — extra draft decodes done by SSD (post-pass + restore)
- `saved` — draft steps skipped via cache hit
- `verify/post/overlap` — wall-times (ms)
- `hidden%` — percentage of total time hidden by async overlap
- `async` — whether `--ssd-async` is active
- `draft` — draft model backend (CPU/GPU)

## Standard Commands

### Default (fastest E4B):
```bash
./build/bin/llama-chat \
  -m  /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf \
  --no-tools
```

### SSD enabled (paper algorithm, async overlap):
```bash
./build/bin/llama-chat \
  -m  /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf \
  --no-tools --draft-max 5 --ssd-fan-out 7 --ssd-async
```

### Legacy (both GPU):
```bash
./build/bin/llama-chat -m TARGET -md DRAFT --no-tools -ngl-draft 99
```

## Important Notes

1. **Per-turn reset**: `ssd_have_seed` and `spec_cache` are cleared at the start of each new user turn. Prefill resets pending_feat, making previous round seeds invalid.

2. **`reset_chat()` clears SSD state**: On KV overflow or decode error, SSD state is also reset (for consistency).

3. **Tool calls don't affect SSD**: Tool hops within the same turn continue with new run_inference calls. SSD seed persists across rounds within a turn.

4. **Hit rate depends on K**: Chat default `--draft-max 3` is small → bonus usually falls at position K=3 → cache miss. Use `--draft-max 5+` for higher hit rate.

5. **Lossless guarantee**: Greedy target is deterministic. Even with SSD active, diff output is always empty.

## Implementation Details

Changed sections in `examples/chat/chat.cpp`:
- `struct args` — new fields (~line 41)
- `usage()` — new flag list (~line 57)
- `parse_args()` — new parser cases (~line 85)
- `masked_topf` + `geometric_fanout` helpers (~line 250)
- `mp_tgt`/`mp_drf` split (~line 340)
- SSD state + reset_chat hook (~line 497, 540)
- Draft loop 2-phase + post-pass + async dispatch (~line 673)
- Per-turn SSD stats (~line 1080)
- `<future>` + `<cmath>` includes (~line 24)

No header/API changes. Only `chat.cpp` and the previously modified `spec.cpp`. `llama-ext.h`, `llama.h`, CMakeLists unchanged.
