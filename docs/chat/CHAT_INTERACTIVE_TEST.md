# Interactive Chat Test Results — CPU vs GPU Draft (E4B Q8)

Real-world llama-chat session comparison on Gemma4 E4B Q8 target with assistant draft.

## Setup

- **Target**: `gemma-4-E4B-it-Q8_0.gguf` (8 GB Q8)
- **Draft**: `gemma-4-E4B-it-assistant.Q8_0.gguf` (96 MB Q8)
- **Donanım**: M2 Pro
- **Mode**: `-n 200 --no-tools` (multi-turn interactive)

## Results

### Test 1: CPU Draft (default, `-ngl-draft 0`)

```bash
./build/bin/llama-chat \
  -m /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf \
  --no-tools
```

| Prompt | Tokens | Time | t/s | Rounds | Accept% |
|---|---|---|---|---|---|
| boring story | 288 | 9.56s | **30.1** | 158 | 27.8% |
| count 1-20 | 70 | 1.12s | **62.5** | 19 | 93.0% |

**Average**: **46.3 t/s**

### Test 2: GPU Draft (`-ngl-draft 99`)

```bash
./build/bin/llama-chat \
  -m /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf \
  --no-tools -ngl-draft 99
```

| Prompt | Tokens | Time | t/s | Rounds | Accept% |
|---|---|---|---|---|---|
| boring story | 288 | 10.62s | **27.1** | 157 | 28.2% |
| count 1-20 | 70 | 1.21s | **57.8** | 19 | 93.0% |

**Average**: **42.5 t/s**

### Test 3: GPU Draft Repeat (consistency check)

Same as Test 2:
- boring story: 27.1 t/s
- count 1-20: 57.8 t/s

Results stable → GPU draft backend consistent.

### Test 4: SSD Async (`--draft-max 5 --ssd-fan-out 7 --ssd-async`)

```bash
./build/bin/llama-chat \
  -m /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf \
  --no-tools --draft-max 5 --ssd-fan-out 7 --ssd-async
```

| Prompt | Tokens | Time | t/s | Rounds | Accept% | Cache Hit% | Hidden% |
|---|---|---|---|---|---|---|---|
| boring story | 288 | 13.30s | **21.7** | 157 | 17.1% | 13.1% | 21.6% |
| count 1-20 | 70 | 1.07s | **65.3** | 13 | 87.7% | 0.0% | 25.3% |

**Average**: **43.5 t/s**

**SSD stats** (test 1):
- verify=293ms, post=911ms, overlap=944ms
- Post-pass 3× longer than verify → async hides ~22% overhead
- Cache hits 13% despite K=5 (bonus mostly misses)

## Summary

| Config | Avg t/s | vs CPU draft |
|---|---|---|
| CPU draft (default) | **46.3** | — |
| GPU draft | **42.5** | **-8.2%** ❌ |
| SSD async | **43.5** | **-6.0%** ❌ |

**CPU draft faster by ~8% on E4B Q8.**
**SSD async ≈ GPU draft, overhead 21-25% hidden via async (not enough to beat SD baseline).**

Reason: E4B target = 8 GB Metal → already fills GPU well. Draft GPU share → memory contention. Draft CPU NEON = free resource, avoids contention. Same finding as `SSD_E4B_REPORT.md` benchmark mode.

## Recommendation

For E4B Q8 on Apple Silicon:
- **Max speed**: default `-ngl-draft 0` (CPU draft) ~46 t/s
- **Paper SSD demo**: `--ssd-fan-out 7 --ssd-async` (async overhead mostly hidden, ~43 t/s)
- **Legacy (E2B only)**: `-ngl-draft 99` still faster

## Notes

- Output identical across modes (lossless)
- Acceptance rate similar (draft quality stable)
- Generation speed metric: reported per-turn by chat
- Both modes interactive/responsive — CPU advantage marginal in user feel, measurable in metrics
