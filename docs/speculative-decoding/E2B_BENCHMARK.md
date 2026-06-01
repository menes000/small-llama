# E2B Speculative Decoding Benchmark

Date: 2026-05-26
Model pair:
- target: `/Users/enes/Desktop/all/llms/gemma-4-E2B-it-UD-Q8_K_XL.gguf`
- draft:  `/Users/enes/Desktop/all/llms/gemma-4-E2B-it-assistant.F16.gguf`

Environment: Metal `-ngl 99`, `--temp 0` (greedy), `-n 300`, `-st`.
Command template (baseline):
```bash
./build/bin/llama-cli -m <target> -p "<prompt>" -n 300 --temp 0 -st -ngl 99
```
Spec:
```bash
./build/bin/llama-cli -m <target> -md <draft> \
  --spec-type draft-mtp --spec-draft-n-max <N> \
  -p "<prompt>" -n 300 --temp 0 -st -ngl 99
```

All spec outputs **byte-for-byte identical** to baseline (lossless guarantee).

---

## Results Table

| # | Prompt | baseline (t/s) | spec n=3 (t/s · acc% · acc/round) | spec n=5 (t/s · acc% · acc/round) | Best | Speedup |
|---|--------|---------------:|----------------------------------:|----------------------------------:|:----:|--------:|
| 1 | "hello" | 50.9 | 53.2 · 59.0% · 1.77 | **58.1** · 50.8% · 2.54 | n=5 | **1.14×** |
| 2 | "What is 2+2? Answer in one word." | 50.6 | **52.6** · 56.5% · 1.70 | 51.0 · 42.0% · 2.10 | n=3 | 1.04× |
| 3 | "Translate to French: The quick brown fox jumps over the lazy dog." | 50.5 | 48.9 · 50.4% · 1.51 | 45.5 · 35.4% · 1.77 | baseline | 0.97× |
| 4 | "List the first 50 prime numbers." | 50.0 | 68.2 · 80.8% · 2.43 | **74.8** · 72.0% · 3.60 | n=5 | **1.50×** |
| 5 | "Count from 1 to 100, one number per line." | 50.2 | **52.6** · 56.5% · 1.69 | 50.0 · 41.0% · 2.05 | n=3 | 1.05× |
| 6 | "Explain how a transistor works in simple terms." | 50.5 | 44.2 · 42.2% · 1.27 | 40.3 · 30.3% · 1.51 | baseline | 0.88× |
| 7 | "Write a Python function to compute the factorial of n." | 50.9 | **53.7** · 59.5% · 1.79 | 51.6 · 46.4% · 2.32 | n=3 | 1.05× |
| 8 | "Write a short story about a robot learning to paint." | 50.4 | 41.5 · 39.4% · 1.18 | 35.9 · 27.8% · 1.39 | baseline | 0.82× |

Baseline average: **50.4 t/s** (nearly constant across prompts).

---

## Summary by Type

| Type | Example | Speedup | Note |
|------|---------|---------|------|
| **Structured/templated list** | primes | **1.50×** | highest acceptance (80%+), n=5 ideal |
| Very short / Q&A | hello, 2+2 | 1.04–1.14× | thinking template produces structured output |
| Sequential counting | 1→100 | 1.05× | lower than expected (separate argmax per number) |
| Code generation | python factorial | 1.05× | structured but with explanation → medium acc |
| Short translation | French | 0.97× | near breakeven |
| Explanatory text | transistor | 0.88× | creative → loss |
| Creative prose | robot story | 0.82× | most diverse → largest loss |

---

## Observations

### 1. avg_accepted/round can't exceed n_max
Even at n=5, most prompts saturate at 2-3 → higher n_max wastes overhead. Only structured tasks (primes 3.60) rewarded by larger n_max.

### 2. acceptance ≠ speedup — **acc/round** is the real metric
n=5 acc% always lower than n=3 (math: more tokens → more rejections), but the actual metric is `accepted/round`. n=5 primes gives 3.60 → large gain.

### 3. n_max sweet-spot heuristic
- avg_accepted/round approaching n_max → **increase n_max** (more to gain)
- avg_accepted/round ≤ n_max/2 → **decrease n_max** (wasted draft)
- avg_accepted/round < 1.3 → spec **likely losing**, use baseline

### 4. Thinking template effect
Gemma-4-it gguf chat template enables thinking mode (`<|channel>thought`, "Thinking Process:"). Even short prompts generate long thinking blocks — these structured sections boost acceptance (hello 59%, 2+2 57%). Creative final answer has low acceptance.

### 5. Loss patterns (why losses happen)
- "Write a short story" 39% acc, 1.18 acc/round, 0.82× speed
- Even at 1 token/round average, per-round fixed overhead (graph rebuild + N draft + verify + tap copy + cluster mask) costs more than 1 baseline token → net loss
- This happens **while preserving losslessness** — output correct, just slower

---

## Recommended Usage

| Task | Command |
|------|---------|
| Structured list / table / format | `--spec-draft-n-max 5` or `6` |
| Short Q&A / code / translation | `--spec-draft-n-max 3` |
| Long creative text | baseline (spec loses) or `--spec-draft-n-max 2` |

Check acceptance on first run (`G4A STATS:` in stderr), tune n_max accordingly.

---

## Raw Commands (to reproduce)

```bash
cd /Users/enes/Desktop/all/less-llama-cpp/llama.cpp
M=/Users/enes/Desktop/all/llms/gemma-4-E2B-it-UD-Q8_K_XL.gguf
D=/Users/enes/Desktop/all/llms/gemma-4-E2B-it-assistant.F16.gguf

# baseline
./build/bin/llama-cli -m $M -p "<prompt>" -n 300 --temp 0 -st -ngl 99

# spec
./build/bin/llama-cli -m $M -md $D \
  --spec-type draft-mtp --spec-draft-n-max <N> \
  -p "<prompt>" -n 300 --temp 0 -st -ngl 99
```
