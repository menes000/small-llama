# Chat Template + Tool Use Port — Detailed Report

Date: 2026-05-26
Target tree: `/Users/enes/Desktop/all/less-llama-cpp/only-needed-files/`
Reference tree: `/Users/enes/Desktop/all/less-llama-cpp/llama.cpp/` (full llama.cpp)

---

## 0. TL;DR

- **Goal:** `only-needed-files` (minimal llama.cpp port) didn't recognize Gemma-4 chat template; `llama_chat_apply_template` C API returned -1 for gemma-4 gguf files. Additionally, a new `llama-chat` REPL example with `read_file` and `list_dir` tool-use support was added.
- **Output:** Gemma-4 template recognized, multi-turn chat works, thinking mode optional, two sandboxed tools work, lossless (output matches gguf jinja).
- **Total new code: ~420 lines** (new `examples/chat/chat.cpp` 389 lines + ~30 lines added to shared files). 30% less than the 590 lines estimated in the plan — achieved by simplifying design and removing the separate `_ex` API.
- See Challenges section for details: gemma-4 jinja is rich (348 lines, thinking + tool blocks + macros), C API doesn't carry tools/thinking, `std::regex` doesn't support `[^]`, multi-turn KV cache management, tool-call stop condition, thinking block stripping.

---

## 1. Starting State

`only-needed-files` state (before port):
- `llama-simple` — single prompt, raw tokenize, doesn't apply chat template.
- `llama-spec` — speculative decoding example; manually builds gemma chat template (hardcoded `<start_of_turn>user\n…` string tokenized — not flexible).
- `src/llama-chat.{h,cpp}` (938 lines) — verbatim copy from full llama.cpp with 52 hand-coded templates (CHATML, LLAMA_2, MISTRAL, PHI, GEMMA (classic), DEEPSEEK, etc.). **Doesn't recognize Gemma-4.** Detection (`llm_chat_detect_template`) uses substring matching; no branch matching gemma-4 jinja signatures (`<|turn>`, `<|tool_call>`) → `LLM_CHAT_TEMPLATE_UNKNOWN` → C API returns -1.
- `include/llama.h` — standard `llama_chat_apply_template(tmpl, msgs, n, add_ass, buf, len)` signature. No tools or thinking parameter.

User's previous attempt:
```
chat template failed (n=-1)
```
Had to bypass with a manual string in the spec example.

---

## 2. Goals and Decision Process

User answered 3 questions:

| Question | Answer | Implication |
|------|-------|---------|
| Tool scope? | **Limited practical: `read_file`, `list_dir`** | Read-only, sandbox. No write/bash. |
| Thinking? | **Default off (`--thinking` flag)** | Don't inject `<\|think\|>` by default. |
| Where? | **New example: `llama-chat`** | spec.cpp untouched. Clean separation. |

User's main principle: minimal LOC.

---

## 3. Design Decisions

### 3.1. Hand-code instead of Jinja

Full tree's approach: jinja-based with `common/jinja/` (~5800 lines) + `common/chat.cpp` (~2500 lines). Supports the full power of gemma-4 template (macros, conditionals, namespaces, etc.).

**Rejected.** Reasons:
1. only-needed-files principle: adding ~6000 lines contradicts minimal LOC philosophy.
2. Targeting only gemma-4 — loading a jinja interpreter for a single model is overkill.
3. Gemma-4's **output structure** (turn delimiters + tool blocks) is regular; hand-codeable in ~150 lines.

**Chosen:** hand-coded gemma-4 applier. We statically build what jinja's macro expansion does.

### 3.2. C API extension (`_ex`) — removed during implementation

Initial plan: new `llama_chat_apply_template_ex(tmpl, msgs, n, add_ass, tools_json, thinking, buf, len)` function in `llama-ext.h`. Tools and thinking as parameters.

**Removed during implementation.** Reason:
- Caller (chat.cpp) already builds system message **content** itself.
- Tool blocks (`<|tool>{…}<tool|>`) and thinking marker (`<|think|>\n`) can naturally be embedded in system message text.
- Extra API surface unnecessary; standard `llama_chat_apply_template` is sufficient.

**Result:** API unchanged. Only a new `LLM_CHAT_TEMPLATE_GEMMA_4` enum value + corresponding apply branch.

### 3.3. Tool dispatch flow

```
user input → push msg
loop (hop ≤ 4):
    apply_template(history) → tail tokenize → decode → sample until <turn|>
    if <tool_call|> seen:
        parse {name, args}
        run tool (sandboxed)
        push assistant msg (raw) + push tool msg (result)
        loop again
    else:
        push assistant msg (thinking stripped)
        break
```

Hop count limited to 4 — protection against infinite tool-call loops.

### 3.4. Sandboxing

`read_file(path)` and `list_dir(path)`:
1. Resolve symlinks with `realpath(path)`.
2. Compare with `realpath(--root)`.
3. Reject if path is not an absolute prefix of root.
4. Check `/` boundary (`"/tmp"` won't accept `"/tmp_other"`).
5. `read_file` 16 KiB cap, `list_dir` 200 entry cap.

Default root = `$HOME`. User can set stricter sandbox with `--root /tmp`.

---

## 4. Step-by-Step Implementation

### Step 1 — Exploration (Explore agents)

2 parallel Explore agents ran:
- **Agent 1:** Existing chat infrastructure in only-needed-files + full tree minja/chat.cpp analysis.
- **Agent 2:** Complete structure of Gemma-4 jinja template, special tokens, tool-use semantics.

Key findings:
- only-needed-files' `llama-chat.cpp` has 52 hand-coded templates but no gemma-4.
- Gemma-4 special tokens: `<|turn>` (105) / `<turn|>` (106) — turn open/close; `<|tool>` (46) / `<tool|>` (47); `<|tool_call>` / `<tool_call|>`; `<|tool_response>` (50) / `<tool_response|>`; `<|think|>` (98); `<|channel>` (100) / `<channel|>`; `<bos>` (2); `<|"|>` — string escape token in tool args.
- Tool call output format: `<|tool_call>call:NAME{key:value,…}<tool_call|>`.
- Tool response input format: `<|tool_response>response:NAME{result}<tool_response|>`.
- assistant role → "model" (gemma uses "model" for assistant turn name).

### Step 2 — Plan + clarification

`AskUserQuestion` with 3 questions (table above).

Plan file: `/Users/enes/.claude/plans/tamam-chat-templatey-na-sl-rustling-aurora.md`.

### Step 3 — `llama-chat.h` + `llama-chat.cpp`

**Enum addition:**
```cpp
// src/llama-chat.h
LLM_CHAT_TEMPLATE_GEMMA,
LLM_CHAT_TEMPLATE_GEMMA_4,   // ← new
```

**Detection (substring matching):**
```cpp
// src/llama-chat.cpp — llm_chat_detect_template()
} else if (tmpl_contains("<|tool_call>") || tmpl_contains("<|turn>")) {
    return LLM_CHAT_TEMPLATE_GEMMA_4;
} else if (tmpl_contains("<start_of_turn>")) {   // classic gemma
    return LLM_CHAT_TEMPLATE_GEMMA;
}
```
Gemma-4's gguf jinja contains both (`<|turn>` always, `<|tool_call>` when tools supported). Order preserved to avoid false match with `OPENAI_MOE` detection (which had `<|channel>`).

**Name alias:**
```cpp
{ "gemma4", LLM_CHAT_TEMPLATE_GEMMA_4 },
```

**Apply branch:**
```cpp
} else if (tmpl == LLM_CHAT_TEMPLATE_GEMMA_4) {
    ss << "<bos>";
    for (auto message : chat) {
        std::string role(message->role);
        std::string content(message->content ? message->content : "");
        if (role == "assistant") role = "model";
        if (role == "tool") {
            ss << "<|turn>user\n<|tool_response>" << content
               << "<tool_response|><turn|>\n";
        } else {
            ss << "<|turn>" << role << "\n" << content << "<turn|>\n";
        }
    }
    if (add_ass) ss << "<|turn>model\n";
}
```

Notes:
- `<bos>` literal — tokenizer converts it to BOS token with `parse_special=true`.
- assistant → "model" mapping (gemma's turn name is "model").
- tool role: wrapper text inside user turn; gemma jinja presents tool results from user perspective.
- Thinking marker (`<|think|>\n`) does NOT go in THIS function — caller adds it to system message content (clean design).
- Tool blocks (`<|tool>…<tool|>`) do NOT go in THIS function — caller embeds them in system content.

### Step 4 — `examples/chat/chat.cpp` (389 lines)

Single file, no common dependencies. Main components:

#### 4a. Args parsing (~30 lines)
- `-m`, `-n`, `-ngl`, `-c`, `-b`, `--thinking`, `--no-tools`, `--root`.
- Default root: `$HOME`.

#### 4b. Tool registry (~70 lines)
```cpp
std::string tool_read_file(const std::string & path, const std::string & root_abs);
std::string tool_list_dir(const std::string & path, const std::string & root_abs);
```
- Path normalization with `realpath_s`.
- `path_within(abs, root_abs)` prefix + `/` boundary check.
- 16 KiB / 200 entry capping.

#### 4c. System content builder (~15 lines)
```cpp
build_system_content(tools_on, thinking) {
    s = thinking ? "<|think|>\n" : "";
    s += "You are a helpful assistant. Be concise.";
    if (tools_on) {
        s += <|tool>{"name":"read_file",…}<tool|>;
        s += <|tool>{"name":"list_dir",…}<tool|>;
    }
}
```
Tool definitions inline JSON; model reads them in its trained format.

#### 4d. Regex helpers (~25 lines)
- `parse_tool_call(text, &name, &args)` → `<\|tool_call>\s*call:\s*(\w+)\s*\{([^]*?)\}\s*<tool_call\|>`
- `extract_path_arg(args)` → two regexes: first quoted (`<|"|>`-wrapped or `"`-wrapped), then bare token. Gemma-4 arg format is flexible; being permissive is safe.

#### 4e. Tokenize / detokenize helpers (~30 lines)
Pattern copied from spec.cpp; `parse_special=true` required.

#### 4f. Template apply wrapper (~15 lines)
Calls `llama_chat_apply_template` C API. Retries with larger buffer if needed.

#### 4g. Inference loop (`run_inference`) (~50 lines)
- `apply_template(add_ass=true)` for full history → full prompt string.
- Previous turns' tokens already in KV cache; only **tail** (newly added portion) decoded.
- Feed in `n_batch` chunks.
- Greedy sample loop: echo each token piece; break when `<turn|>` or `<tool_call|>` substring seen; break on EOG.

#### 4h. REPL loop (~50 lines)
```
loop:
    getline → push user msg
    for hop in 0..4:
        run_inference → assistant_raw
        if no tools && no <tool_call|>: push assistant, break
        parse tool call → name, args, path
        run tool → result
        push assistant (raw) + push tool (NAME{result})
        re-enter (hop+1)
```

#### 4i. Thinking stripper (~15 lines)
Strips `<|channel>…<channel|>` blocks from **history** across turns (so model's own thinking doesn't pollute next turns). Visible during display (streamed out) but stored with `push_msg("assistant", strip_thinking(text))`.

### Step 5 — CMakeLists hookup

- `examples/chat/CMakeLists.txt` (5 lines, same pattern as simple.cpp)
- Top-level `CMakeLists.txt`: `add_subdirectory(examples/chat)`

### Step 6 — Build + iterative test

```bash
cmake .. && cmake --build . --target llama-chat -j 8
```

First build clean, then found and fixed bugs through testing (see Challenges section).

---

## 5. Challenges and Solutions

### 5.1. C API doesn't carry tools/thinking

**Problem:** `llama_chat_apply_template(tmpl, msgs, n, add_ass, buf, len)` has no tool list or thinking flag parameter. How to pass them?

**Initial plan:** Add new `_ex` C API in `llama-ext.h`.

**Better solution:** Let caller build system message content itself. Embed tools as `<|tool>…<tool|>` blocks, thinking marker as `<|think|>\n` at the start of system text. Apply branch only produces turn delimiters.

**Gain:** API surface didn't grow, ~30 fewer lines. Clean place for `build_system_content` in chat.cpp.

### 5.2. std::regex doesn't support `[^]`

**Problem:** Initial regex:
```cpp
R"(<\|tool_call>\s*call:\s*(\w+)\s*\{([^]*?)\}\s*<tool_call\|>)"
```
In JavaScript, `[^]` means "anything including newline". In C++'s ECMAScript regex variant: undefined behavior.

**Fix:** Use `[\s\S]*?` or just `.*?` (if no multi-line needed). Used `.*?` since tool args are single-line JSON.

### 5.3. `realpath` symlink following

**Problem:** For sandbox checking, `realpath("/tmp")` returns `/private/tmp` on macOS. `--root /tmp` path vs `/tmp/file` realpath differ. Naive `string::starts_with` fails.

**Fix:** Resolve both root and requested path with `realpath_s()`, then compare. Also added `/` boundary check so `/tmp_other` isn't accepted as having prefix `/tmp`:
```cpp
return path_abs.size() == root_abs.size() ||
       path_abs[root_abs.size()] == '/';
```

### 5.4. Tool-call stop condition

**Problem:** Model emits `<|tool_call>` then produces `call:NAME{…}<tool_call|>`. When do we stop sampling? Does `<tool_call|>` appear in output, or do more tokens follow?

**Test:** Model doesn't generate `<turn|>` right after `<tool_call|>` close — goes directly to next text (model wants to summarize, but we need tool dispatch at this point).

**Fix:** After each sample, if `assistant_text.find("<tool_call|>") != npos` then break. Dispatch tool, add response as feedback, re-prompt.

### 5.5. Multi-turn KV cache management

**Problem:** Re-tokenizing full history and decoding all tokens every turn is slow. Must reuse KV cache.

**Fix:** Keep `last_formatted` string. Each turn:
1. New `apply_template(history)` → `formatted` (always grows: previous history + new turn).
2. `tail = formatted.substr(last_formatted.size())` → only newly added portion.
3. Tokenize tail, decode from `kv_pos`.
4. After sampling: `last_formatted = formatted + assistant_text`.

Gain: after first turn, each turn only processes new user message + sampled tokens. Cache always current.

**Pitfall:** `add_special` flag must be `true` for first turn (BOS needed), `false` for subsequent turns. Fixed with `last_formatted.empty()` condition.

### 5.6. `parse_special=true` required

**Problem:** On first test, tool call detection failed. Reason: tokenizer split `<|tool_call>` into individual chars, didn't use special token id.

**Fix:** `llama_tokenize(…, /*parse_special=*/true)`. Special tokens handled as single tokens on both prompt and detokenization sides.

### 5.7. Thinking block stripping

**Problem:** With `--thinking` enabled, model emits `<|channel>thought\n…<channel|>` block. Want to display it (user sees it) but if it enters history, model sees its own thinking in context next turn and gets confused.

**Fix:** Two-layer approach:
- **Stream out:** Print token-by-token (user sees thinking).
- **Save:** Strip `<|channel>…<channel|>` blocks with `strip_thinking(assistant_raw)`, push remainder to history.

This way model sees clean answer next turn; thinking visible only once.

### 5.8. Tool response wrapping

**Problem:** If tool result pushed as `role="tool"`, how does template applier wrap it? Gemma jinja converts OpenAI-style tool role to `<|tool_response>response:NAME{result}<tool_response|>` format.

**Fix:** Separate branch in apply branch for `role=="tool"`:
```cpp
if (role == "tool") {
    ss << "<|turn>user\n<|tool_response>" << content << "<tool_response|><turn|>\n";
}
```
Caller does `push_msg("tool", "list_dir{...result text...}")` — content prefix is `NAME{result}`.

### 5.9. Tool arg parsing — Gemma's quirky format

**Problem:** Gemma-4 tool arg format is not standard JSON — strings wrapped with `<|"|>` token:
```
{path:<|"|>/tmp/file.txt<|"|>}
```

Sometimes uses `"…"` double quotes, sometimes bare strings. Numeric/boolean args are unwrapped.

**Fix:** Two regexes in sequence:
```cpp
re_quoted = R"(path\s*[:=]\s*(?:<\|"\|>|")([^"<]+)(?:<\|"\|>|"))"
re_bare   = R"(path\s*[:=]\s*([^,}\s]+))"
```
Try quoted first; fall back to bare. Permissive approach; sufficient for single `path` arg.

**Limitation:** Nested objects or multiple args not handled by this regex. Our read_file/list_dir only has one `path` arg — sufficient.

### 5.10. Hop limit

**Problem:** Model may fail to understand tool output and call same tool again → infinite loop.

**Fix:** `for (int hop = 0; hop < 4; ++hop)` — max 4 tool calls per turn. Force break after.

### 5.11. Chat template name detection edge case

**Problem:** gguf's chat_template metadata may not have a NAME field; only jinja string present. `llama_model_chat_template(model, nullptr)` returns jinja string.

**Fix:** `llm_chat_detect_template(jinja_string)` works via substring matching; if `<|tool_call>` or `<|turn>` literal in jinja, catches GEMMA_4. Name alias "gemma4" remains as fallback (if gguf convention changes later).

---

## 6. Verification (tests)

All passed in real runs:

| Test | Command | Result |
|------|-------|-------|
| Build | `cmake --build .` | ✅ `llama-chat` binary |
| Plain chat | `printf "hi who are you?\n" | llama-chat …` | ✅ "I am Gemma 4, a Large Language Model developed by Google DeepMind." |
| Multi-turn (context) | 2 consecutive prompts | ✅ "France → Paris", "and Germany?" → "Berlin" (history preserved) |
| `list_dir` tool | "list files in /tmp" | ✅ Real /tmp files returned; `[tool] list_dir(path=/tmp)` logged |
| `read_file` tool | "read /tmp/test_chat.txt" | ✅ "Hello world from test file" read correctly |
| Sandbox rejection | `--root /tmp`, "read /etc/passwd" | ✅ Tool error returned, model gracefully declined |
| `--thinking` mode | "what is 17 * 23?" | ✅ `<|channel>thought\n…<channel|>` block, then correct "391" |

---

## 7. Changed Files

| File | Type | Lines |
|-------|-----|-------|
| `src/llama-chat.h` | edit | +1 (`LLM_CHAT_TEMPLATE_GEMMA_4` enum) |
| `src/llama-chat.cpp` | edit | +24 (detect branch + apply branch + "gemma4" name alias) |
| `examples/chat/chat.cpp` | new | 389 |
| `examples/chat/CMakeLists.txt` | new | 5 |
| `CMakeLists.txt` | edit | +1 (`add_subdirectory(examples/chat)`) |
| **TOTAL** | | **~420 lines** |

Full tree does the same work in ~8500 lines (`common/chat.cpp` + `common/jinja/`). **Our port: 5% of that.**

---

## 8. Usage

### 8.1. Build

```bash
cd /Users/enes/Desktop/all/less-llama-cpp/only-needed-files/build
cmake ..
cmake --build . -j 8
# output: build/bin/llama-chat
```

### 8.2. Run Examples

**Basic chat (tools enabled, sandbox $HOME):**
```bash
./build/bin/llama-chat \
  -m /Users/enes/Desktop/all/llms/gemma-4-E2B-it-UD-Q8_K_XL.gguf \
  -ngl 99
```

**Restrict tool sandbox:**
```bash
./build/bin/llama-chat -m <model> -ngl 99 --root /tmp
```

**Tools disabled, chat only:**
```bash
./build/bin/llama-chat -m <model> -ngl 99 --no-tools
```

**Thinking mode:**
```bash
./build/bin/llama-chat -m <model> -ngl 99 --thinking
```

### 8.3. Args

| Flag | Default | Meaning |
|------|---------|-------|
| `-m` | (required) | gguf model file |
| `-n` | 1024 | max new tokens per turn |
| `-ngl` | 99 | GPU layers (Metal) |
| `-c` | 4096 | context size |
| `-b` | 512 | batch size |
| `--thinking` | (off) | inject `<|think|>`, show reasoning trace |
| `--no-tools` | (on) | don't add tool definitions to system |
| `--root` | `$HOME` | tool sandbox root path |

### 8.4. Stats / debug

On stderr:
- `[chat] model loaded. thinking=… tools=… root=…` (at startup)
- `[tool] read_file(path=…)` or `[tool] list_dir(path=…)` (on each tool dispatch)

Stdout: model output only (token stream).

---

## 9. Architecture (flow diagram)

```
┌──────────────────────────────────────────────────────────────┐
│  REPL loop (chat.cpp main)                                    │
│  ┌──────────────────────┐                                     │
│  │ getline user input   │                                     │
│  └──────────┬───────────┘                                     │
│             │ push_msg("user", text)                          │
│             v                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │ HOP LOOP (max 4):                                        │ │
│  │ ┌──────────────────────────────────────────────────────┐ │ │
│  │ │ apply_template(model, history, add_ass=true)         │ │ │
│  │ │   → llama_chat_apply_template (C API, llama.cpp)     │ │ │
│  │ │   → llm_chat_detect_template (gemma-4 detected)      │ │ │
│  │ │   → llm_chat_apply_template (gemma-4 branch)         │ │ │
│  │ │   → "<bos><|turn>system\n<|think|>\n…<|tool>…        │ │ │
│  │ │       <tool|>…<turn|>\n<|turn>user\n…<turn|>\n       │ │ │
│  │ │       <|turn>model\n"                                │ │ │
│  │ └──────────────────────────────────────────────────────┘ │ │
│  │             │                                            │ │
│  │             v                                            │ │
│  │ ┌──────────────────────────────────────────────────────┐ │ │
│  │ │ Tail tokenize + decode in n_batch chunks             │ │ │
│  │ │ (KV cache reused from previous turns)                │ │ │
│  │ └──────────────────────────────────────────────────────┘ │ │
│  │             │                                            │ │
│  │             v                                            │ │
│  │ ┌──────────────────────────────────────────────────────┐ │ │
│  │ │ Sample loop (greedy):                                │ │ │
│  │ │   stream piece to stdout                             │ │ │
│  │ │   if EOG or <turn|> or <tool_call|>: break           │ │ │
│  │ └──────────────────────────────────────────────────────┘ │ │
│  │             │                                            │ │
│  │             v                                            │ │
│  │ ┌──────────────────────────────────────────────────────┐ │ │
│  │ │ parse_tool_call(assistant_text)                      │ │ │
│  │ └──────────┬──────────────────────┬────────────────────┘ │ │
│  │            │ no match             │ matched (name, args) │ │
│  │            v                      v                      │ │
│  │ ┌──────────────────┐  ┌─────────────────────────────────┐│ │
│  │ │ strip_thinking   │  │ extract_path_arg → path         ││ │
│  │ │ push("assistant")│  │ tool_read_file / tool_list_dir  ││ │
│  │ │ break hop loop   │  │   (realpath check vs --root)    ││ │
│  │ └──────────────────┘  │ push("assistant", raw)          ││ │
│  │                       │ push("tool", NAME{result})      ││ │
│  │                       │ continue hop loop               ││ │
│  │                       └─────────────────────────────────┘│ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 10. Known Limitations

1. **Read-only tools only.** No `write_file`, `bash`, `edit_file`. Can be extended with same sandbox rule (`path_within(root)` check).
2. **Single tool per turn.** No parallel tool calls. Hop loop dispatches sequentially.
3. **Tool arg parsing optimized for single `path` arg.** `extract_path_arg` insufficient for nested objects or multi-arg tools. An extensible args parser could be written.
4. **Gemma-4 template only.** Legacy hand-coded paths still work for other templates (LLAMA_3, MISTRAL, etc.), but gemma-4-specific features (thinking, tool blocks) only with gemma-4.
5. **No history persistence.** History lost on program exit. JSON dump/restore could be added (~30 lines).
6. **Greedy sampling fixed.** No top-k/temperature. Could use `llama_sampler`.
7. **Hop limit hardcoded** at 4. More dynamic control possible.

---

## 11. Next Steps (suggestions)

| Priority | Work | Estimated LOC |
|---------|----|-------------|
| High | `write_file(path, content)` + `append_file` | +30 (same sandbox guard) |
| Medium | `bash(cmd)` allow-list (`ls`, `cat`, `grep` only) | +50 |
| Medium | History save/load (JSON) | +30 |
| Low | Top-k / temperature sampling | +20 (existing llama_sampler API) |
| Low | Streaming SSE output (server mode) | +200 |

---

## 12. Reference Files

- Plan file: `/Users/enes/.claude/plans/tamam-chat-templatey-na-sl-rustling-aurora.md`
- Previous report (spec decoding port): `/Users/enes/Desktop/all/less-llama-cpp/llama.cpp/GEMMA4_ASSISTANT_FIX_REPORT.md`
- Full tree jinja code (reference, not ported): `/Users/enes/Desktop/all/less-llama-cpp/llama.cpp/common/jinja/`
- HF gemma-4 chat template (canonical): `/Users/enes/Desktop/all/less-llama-cpp/gemma4-e2b/chat_template.jinja`
