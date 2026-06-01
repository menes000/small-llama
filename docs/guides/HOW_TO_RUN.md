# How to Run — Quick Reference

Two binaries:
- `llama-spec` — single prompt, batch execution (benchmark, test)
- `llama-chat` — interactive REPL (multi-turn chat, tools)

Both share the same SSD/CPU-draft infrastructure.

---

## Model Paths

```
/Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf              # target
/Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf    # draft
```

E2B pair also exists, but CPU draft performs better with large targets — use E4B.

---

## Build (if needed)

```bash
cd /Users/enes/Desktop/all/less-llama-cpp/only-needed-files
cmake --build build --target llama-spec llama-chat
```

---

## llama-chat (interactive)

### Fastest Mode (Default — SD + CPU draft)

```bash
./build/bin/llama-chat \
  -m  /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf \
  --no-tools
```

**~73 t/s** on primes prompt with E4B Q8 pair. No extra flags needed.

### SSD Enabled (paper algorithm, async overlap)

```bash
./build/bin/llama-chat \
  -m  /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf \
  --no-tools --draft-max 5 --ssd-fan-out 7 --ssd-async
```

**~70 t/s** + lossless. SSD overhead fully hidden via async overlap (hidden=29%).

### Tools Enabled (read_file, list_dir)

```bash
./build/bin/llama-chat \
  -m  /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf \
  --root /Users/enes/Desktop/all --yolo
```

`--yolo` = auto-confirm every tool call. Remove `--yolo` for interactive confirmation.

### Legacy (both GPU — SD Metal)

```bash
./build/bin/llama-chat -m TARGET -md DRAFT --no-tools -ngl-draft 99
```

Slowest mode for E4B. Still fastest for E2B pair.

---

## llama-spec (benchmark)

### Default (SD CPU draft)

```bash
./build/bin/llama-spec \
  -m  /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf \
  -p "list the first 40 prime numbers" -n 150 --draft-max 5
```

### SSD Enabled

```bash
./build/bin/llama-spec \
  -m  /Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf \
  -md /Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf \
  -p "list the first 40 prime numbers" -n 150 --draft-max 5 \
  --ssd-fan-out 7 --ssd-async
```

---

## Flag Reference

### Backend split (same for both binaries)
| Flag | Default | Description |
|---|---|---|
| `-ngl N` | 99 | Target GPU layers (99 = full Metal) |
| `-ngl-draft N` | **0** | Draft GPU layers (0 = CPU NEON) |

### SSD (paper 2603.03251v3)
| Flag | Default | Description |
|---|---|---|
| `--ssd-fan-out B` | 1 | Total fan-out budget (B≤K = off; B>K = active) |
| `--ssd-r R` | 1.0 | Geometric fan-out exponent (Thm 12) |
| `--ssd-sampling-C C` | 1.0 | Saguaro sampling downweight (no effect in greedy mode) |
| `--ssd-async` | off | Parallel SSD post-pass with target verify |

### General
| Flag | Default | Description |
|---|---|---|
| `--draft-max N` | spec=5, chat=3 | Draft tokens per round (K) |
| `-n N` | spec=128, chat=1024 | Max generation tokens |
| `-c N` | 4096 | Context size |
| `-p "..."` | "hello" (spec) | Single prompt (spec only) |

### Chat-only
| Flag | Default | Description |
|---|---|---|
| `--no-tools` | off | Disable tools |
| `--no-sd` | off | Disable SD (even if -md provided) |
| `--thinking` | off | Thinking mode |
| `--kv-f16` | off | K/V cache f16 (default q8) |
| `--root DIR` | $HOME | Tool sandbox root |
| `--yolo` | off | Skip tool confirmations |

---

## What Each Mode Gains

| Scenario | Extra flags | Expected gain |
|---|---|---|
| **Speed only** | (none) | SD CPU 12% faster than Metal on E4B |
| Paper SSD demo | `--ssd-fan-out 7 --ssd-async` | Equal to SD CPU, lossless |
| Measure hit rate | `--ssd-fan-out 10 --ssd-async` | Hit 20-40%, t/s slightly lower |
| Legacy (both GPU) | `-ngl-draft 99` | Slow on E4B, fast on E2B |

---

## Reading the Output

### Spec stats:
```
[spec] emitted=150 in 1.36s = 110.2 t/s | rounds=30 drafted=150 accepted=119 accept_rate=79.3% acc_per_round=3.97
[spec] SSD: hits=0 misses=19 hit_rate=0.0% extra_draft_decodes=77 steps_saved=0 net_extra=77
[spec] timing: verify=42ms post_pass=256ms overlap_region=261ms hidden=12.5%   async=on draft_backend=CPU
```

### Chat per-turn stats:
```
[stats] gen=119 tok / 1.71s = 69.8 t/s | prefill=31 tok / 0.23s = 135 t/s | rounds=22 drafted=110 accepted=99 acc_rate=90.0% | total=1.92s
[stats] SSD: hits=0 misses=5 hit=0.0% extra=41 saved=0 | verify=48ms post=79ms overlap=89ms hidden=29.3% async=on draft=CPU
```

### Key metrics
- `t/s` → main speed (higher = better)
- `acc_rate` → draft quality (higher = better)
- `hidden%` → how much async overlap hid (higher = better)
- `net_extra` → SSD net cost (lower/negative = better)

---

## Lossless Verification

Every mode produces identical output (greedy target is deterministic):

```bash
TGT=/Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf
DRF=/Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf
P="list 30 primes"

./build/bin/llama-spec -m $TGT -md $DRF -p "$P" -n 100                              > /tmp/sd.out
./build/bin/llama-spec -m $TGT -md $DRF -p "$P" -n 100 --ssd-fan-out 7 --ssd-async > /tmp/ssd.out
diff /tmp/sd.out /tmp/ssd.out  # must be empty
```

---

## Quick One-Click Comparison

```bash
TGT=/Users/enes/Desktop/all/llms/gemma-4-E4B-it-Q8_0.gguf
DRF=/Users/enes/Desktop/all/llms/gemma-4-E4B-it-assistant.Q8_0.gguf
P="list the first 40 prime numbers"

echo "=== SD CPU draft (default, fastest) ==="
./build/bin/llama-spec -m $TGT -md $DRF -p "$P" -n 150 --draft-max 5 2>&1 \
  | grep -E "^\[spec\] (emitted|timing)"

echo "=== SSD async ==="
./build/bin/llama-spec -m $TGT -md $DRF -p "$P" -n 150 --draft-max 5 \
  --ssd-fan-out 7 --ssd-async 2>&1 \
  | grep -E "^\[spec\] (emitted|SSD: hits|timing)"

echo "=== Chat default ==="
printf "list the first 30 primes\n" | ./build/bin/llama-chat \
  -m $TGT -md $DRF --no-tools -n 200 2>&1 \
  | grep -E "^\[stats\]"
```

---

## Related Docs

- `SSD_REPORT.md` — Algorithm detail + bug fix history
- `SSD_COMPARISON.md` — E2B SD vs SSD sweep
- `SSD_ASYNC_REPORT.md` — E2B async overlap analysis
- `SSD_E4B_REPORT.md` — E4B Q8 pair comparison + CPU draft discovery
- `CHAT_SSD_REPORT.md` — Chat port details
- `USAGE.md` — spec.cpp-focused reference
