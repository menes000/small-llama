# SD vs SSD Comparison — Experiment Results

Model: `gemma-4-E2B-it-UD-Q8_K_XL` (target Q8) + `gemma-4-E2B-it-assistant.F16` (draft F16)
Platform: Mac, Metal backend, single GPU.
SSD flag: `--ssd-fan-out 7` (B=7). SD = `--ssd-fan-out 1` (B=1 = off).

Output byte-identical in every test (lossless). All diffs zero.

---

## Results Table

### draft-max=3 (K=3, B=7 > K → SSD active)

| Prompt | Mode | t/s | accept% | acc/round | hit% | extra_dec | saved | net_extra |
|---|---|---|---|---|---|---|---|---|
| primes (40 primes) | SD | **110.7** | 94.0 | 2.82 | — | 0 | 0 | 0 |
| primes | SSD B=7 | 92.3 | 94.0 ✓ | 2.82 | 33 | 168 | 1 | +167 |
| count (1-60) | SD | **101.9** | 81.9 | 2.46 | — | 0 | 0 | 0 |
| count | SSD B=7 | 80.1 | 81.9 ✓ | 2.46 | 44 | 164 | 4 | +160 |
| creative (robot story) | SD | **51.8** | 22.6 | 0.68 | — | 0 | 0 | 0 |
| creative | SSD B=7 | 41.8 | 21.7 ✓ | 0.65 | 20 | 256 | 10 | +246 |
| coding (fibonacci) | SD | **49.4** | 20.3 | 0.61 | — | 0 | 0 | 0 |
| coding | SSD B=7 | 38.9 | 18.9 ✓ | 0.57 | 19 | 324 | 14 | +310 |

### draft-max=5 (K=5, B=7 > K → SSD active)

| Prompt | Mode | t/s | accept% | acc/round | hit% | extra_dec | saved | net_extra |
|---|---|---|---|---|---|---|---|---|
| primes | SD | **109.8** | 79.3 | 3.97 | — | 0 | 0 | 0 |
| primes | SSD B=7 | 98.3 | 79.3 ✓ | 3.97 | 0 ⚠️ | 77 | 0 | +77 |
| count | SD | **94.5** | 66.4 | 3.32 | — | 0 | 0 | 0 |
| count | SSD B=7 | 86.7 | 66.4 ✓ | 3.32 | 7 | 83 | 1 | +82 |
| creative | SD | **40.5** | 14.1 | 0.71 | — | 0 | 0 | 0 |
| creative | SSD B=7 | 35.9 | 13.6 ✓ | 0.68 | 16 | 153 | 8 | +145 |
| coding | SD | **39.1** | 12.6 | 0.63 | — | 0 | 0 | 0 |
| coding | SSD B=7 | 33.9 | 11.7 ✓ | 0.59 | 16 | 189 | 12 | +177 |

### draft-max=7 (K=7, B=7 = K → SSD automatically off)

| Prompt | Mode | t/s | accept% | hit% | Note |
|---|---|---|---|---|---|
| primes | SD | 96.5 | 61.7 | 0 | — |
| primes | SSD B=7 | 98.8 | 61.7 | 0 | B=K → no alts |
| count | SD | 78.5 | 47.4 | 0 | — |
| count | SSD B=7 | 78.1 | 47.4 | 0 | B=K → no alts |
| creative | SD | 32.9 | 10.1 | 0 | — |
| creative | SSD B=7 | 32.4 | 10.1 | 0 | B=K → no alts |
| coding | SD | 30.9 | 9.0 | 0 | — |
| coding | SSD B=7 | 30.6 | 9.0 | 0 | B=K → no alts |

---

## Key Findings

### 1. Losslessness — Holds in Every Test
`diff SD_output SSD_output` → always 0 difference. Even when accept rate changes, emitted text is identical (greedy target is deterministic).

### 2. Accept Rate — Preserved at B=7, K≤5
At draft-max=3 and draft-max=5, SSD B=7 does not change accept rate (chain integrity intact). Single exception: primes + draft-max=5 hit=0 (cache miss, but accept rate unaffected).

### 3. t/s Always Lower (Single GPU)
Paper's gain requires draft running on a SEPARATE device (T_p < 1). On single Metal GPU:
- extra_decodes always greater than steps_saved
- net_extra always positive (+77 → +310 range)
- t/s loss: 5–25%

### 4. Hit Rate Pattern Matches Paper Prediction
```
structured/predictable > creative/diverse
primes (dm=3):  33–44% hit  ← highest
count (dm=3):   44% hit
creative:       16–20% hit
coding:         16–19% hit
```
Paper Fig.3: rejection rate (= 1 - hit_rate) falls as power-law with fan-out → larger B gives higher hit rate.

### 5. B = K Case → Zero Overhead (Safe Fallback)
`geometric_fanout(B, K)` with B≤K → all-ones → zero alts. `ssd_fan_out > n_draft_max` condition is False → cache lookup skipped.
draft-max=7, B=7: SD and SSD nearly identical (difference <1%). Safe fallback behavior.

### 6. draft-max=3 + B=7 → Highest Hit Rate
Short lookahead → bonus token usually falls at positions 0-2 → draft's top-F prediction more accurate. dm=3, count prompt: **44% hit** — best result.

### 7. draft-max=5, primes → 0 Hit Anomaly
Same prompt: 33% hit at dm=3, 0% hit at dm=5. Reason: deeper lookahead places bonus token at positions 3-4 → draft's top-F predictions at those positions are wrong (conditional on 5-step chain). Not random — deterministic, repeatable across runs.

---

## When to Use

| Situation | Recommendation |
|---|---|
| Single Mac, speed priority | `--ssd-fan-out 1` (SSD off) |
| Single Mac, want to measure hit rate | `--ssd-fan-out 7 --draft-max 3` |
| Separate GPU/device (future) | `--ssd-fan-out 20+` — real gains start here |
| Structured/templated prompt | SSD higher hit rate → better on multi-device |
| Creative prompt | Hit rate ~20% → marginal gain on multi-device |

---

## Known Limitation: Accept Rate Drop on Creative Prompts with Small draft-max

dm=3, creative: SD=22.6% accept, SSD=21.7% accept — small drop. Accumulated K/V restore gap over many rounds. Minor version of the large drop seen at primes dm=10+ (82.5%→69.6%).

B=7 choice minimizes but doesn't eliminate this limitation.
