// Minimal multi-turn chat REPL for Gemma 4, with two read-only sandboxed tools.
//
// What this does:
//   - Loads a single gemma-4-it gguf and runs a stdin/stdout REPL.
//   - Applies the gemma-4 chat template (via the C API extended in src/llama-chat.cpp).
//   - On each user turn, builds the full templated prompt, decodes incrementally on the
//     existing context (KV cache reused across turns), and samples greedily until the
//     model emits the close-turn special token (<turn|>, id 106) or another EOG.
//   - Scans the assistant's emitted text for a tool call block:
//         <|tool_call>call:NAME{args}<tool_call|>
//     If found, runs NAME against the local tool registry (`read_file`, `list_dir`),
//     appends a {role:"tool", content:"NAME{result}"} message to the history, and
//     re-prompts the model. Otherwise the assistant response is committed and we loop.
//
// Tools are sandboxed against --root (default $HOME): resolved with realpath() and
// rejected if the resolved path is not a prefix of realpath(root). read_file caps at
// 16 KiB, list_dir caps at 200 entries.

#include "llama.h"
#include "../../src/llama-ext.h"

#include <algorithm>
#include <cerrno>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <regex>
#include <string>
#include <vector>
#include <fstream>
#include <deque>
#include <sstream>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>

struct args {
    std::string model;
    std::string model_draft;        // -md (optional; turns on SD if set)
    int  n_predict    = 1024;
    int  n_gpu_layers = 99;
    int  n_ctx        = 4096;
    int  n_batch      = 512;
    int  n_draft_max  = 3;
    bool thinking     = false;
    bool no_tools     = false;
    bool no_sd        = false;      // force SD off even if -md given
    bool yolo         = false;      // skip per-tool approval (auto-allow all)
    bool kv_q8        = true;       // K/V cache stored as q8_0 by default; --kv-f16 reverts
    std::string root  = "";
};

static void usage(const char * a0) {
    fprintf(stderr,
        "\nusage:\n  %s -m <model.gguf> [-md <draft.gguf>] [-n N] [--draft-max N]\n"
        "              [-ngl N] [-c N] [-b N] [--thinking] [--no-tools] [--no-sd]\n"
        "              [--kv-f16] [--root <dir>]\n\n", a0);
}

static bool parse_args(int argc, char ** argv, args & a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&](const char * what) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", what); return nullptr; }
            return argv[++i];
        };
        if      (s == "-m")           { auto v = next("-m");           if (!v) return false; a.model        = v; }
        else if (s == "-md")          { auto v = next("-md");          if (!v) return false; a.model_draft  = v; }
        else if (s == "-n")           { auto v = next("-n");           if (!v) return false; a.n_predict    = atoi(v); }
        else if (s == "-ngl")         { auto v = next("-ngl");         if (!v) return false; a.n_gpu_layers = atoi(v); }
        else if (s == "-c")           { auto v = next("-c");           if (!v) return false; a.n_ctx        = atoi(v); }
        else if (s == "-b")           { auto v = next("-b");           if (!v) return false; a.n_batch      = atoi(v); }
        else if (s == "--draft-max")  { auto v = next("--draft-max");  if (!v) return false; a.n_draft_max  = atoi(v); }
        else if (s == "--thinking")   { a.thinking = true; }
        else if (s == "--no-tools")   { a.no_tools = true; }
        else if (s == "--no-sd")      { a.no_sd    = true; }
        else if (s == "--yolo")       { a.yolo     = true; }
        else if (s == "--kv-f16")     { a.kv_q8   = false; }
        else if (s == "--root")       { auto v = next("--root");       if (!v) return false; a.root         = v; }
        else if (s == "-h" || s == "--help") { usage(argv[0]); return false; }
        else { fprintf(stderr, "unknown arg: %s\n", s.c_str()); usage(argv[0]); return false; }
    }
    if (a.model.empty()) { usage(argv[0]); return false; }
    if (a.root.empty()) {
        const char * h = getenv("HOME");
        a.root = h ? h : "/";
    }
    return true;
}

// ---- sandbox + tools ------------------------------------------------------

static std::string realpath_s(const std::string & p) {
    char buf[PATH_MAX];
    if (realpath(p.c_str(), buf)) { return std::string(buf); }
    return {};
}

static bool path_within(const std::string & path_abs, const std::string & root_abs) {
    if (path_abs.empty() || root_abs.empty()) { return false; }
    if (path_abs.size() < root_abs.size()) { return false; }
    if (path_abs.compare(0, root_abs.size(), root_abs) != 0) { return false; }
    // require exact match or '/' boundary so root="/tmp" does not accept "/tmp_other"
    return path_abs.size() == root_abs.size() || path_abs[root_abs.size()] == '/';
}

// Resolve `path` to an absolute canonical path. If it's relative (or absolute-but-missing)
// and direct realpath fails, retry joined against `root_abs`. Returns empty on failure.
static std::string resolve_with_root(const std::string & path, const std::string & root_abs) {
    std::string abs = realpath_s(path);
    if (!abs.empty()) { return abs; }
    // strip a leading '/' so the join is well-formed
    std::string rel = path;
    while (!rel.empty() && rel.front() == '/') { rel.erase(0, 1); }
    if (rel.empty()) { return {}; }
    abs = realpath_s(root_abs + "/" + rel);
    return abs;
}

static std::string tool_read_file(const std::string & path, const std::string & root_abs) {
    const std::string abs = resolve_with_root(path, root_abs);
    if (abs.empty()) { return "error: cannot resolve path (tried '" + path + "' and '" + root_abs + "/' prefix)"; }
    if (!path_within(abs, root_abs)) { return "error: path outside allowed root (resolved to '" + abs + "')"; }
    std::ifstream f(abs, std::ios::binary);
    if (!f) { return "error: cannot open file"; }
    constexpr size_t cap = 16 * 1024;
    std::string s(cap + 1, '\0');
    f.read(&s[0], cap + 1);
    const std::streamsize got = f.gcount();
    if (got <= 0) { return ""; }
    const bool truncated = (size_t) got > cap;
    s.resize(truncated ? cap : (size_t) got);
    if (truncated) { s += "\n...[truncated]"; }
    return s;
}

static std::string tool_list_dir(const std::string & path, const std::string & root_abs) {
    const std::string abs = resolve_with_root(path, root_abs);
    if (abs.empty()) { return "error: cannot resolve path (tried '" + path + "' and '" + root_abs + "/' prefix)"; }
    if (!path_within(abs, root_abs)) { return "error: path outside allowed root (resolved to '" + abs + "')"; }
    DIR * d = opendir(abs.c_str());
    if (!d) { return "error: cannot open directory"; }
    std::string out;
    int count = 0;
    const int cap = 200;
    while (dirent * e = readdir(d)) {
        if (!std::strcmp(e->d_name, ".") || !std::strcmp(e->d_name, "..")) { continue; }
        if (count++ >= cap) { out += "...[truncated]\n"; break; }
        out += e->d_name;
        // mark directories with trailing '/'
        struct stat st;
        std::string child = abs + "/" + e->d_name;
        if (!stat(child.c_str(), &st) && S_ISDIR(st.st_mode)) { out += "/"; }
        out += "\n";
    }
    closedir(d);
    if (out.empty()) { return "(empty directory)"; }
    return out;
}

// Returns the system content with all available tool blocks pre-formatted as
// <|tool>{"name":..., "description":..., "parameters":{...}}<tool|> blocks. The
// gemma-4 template applier inserts this content verbatim inside the system turn.
static std::string build_system_content(bool tools_on, bool thinking) {
    std::string s;
    if (thinking) { s += "<|think|>\n"; }
    s += "You are a helpful assistant. Be concise.";
    if (tools_on) {
        s += "\n";
        s += "<|tool>{\"name\":\"read_file\",\"description\":\"Read a small text file. Returns its content (capped at 16 KiB).\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\",\"description\":\"absolute or relative file path\"}},\"required\":[\"path\"]}}<tool|>";
        s += "<|tool>{\"name\":\"list_dir\",\"description\":\"List entries in a directory (capped at 200).\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\",\"description\":\"absolute or relative directory path\"}},\"required\":[\"path\"]}}<tool|>";
    }
    return s;
}

// Extract `path` argument from a model-generated tool-call args blob like
//   path:<|"|>/etc/hosts<|"|>     or     path:"/etc/hosts"     or     {"path":"/etc/hosts"}
// We are deliberately permissive: scan for path:..., extract the first quoted-or-bare string.
static std::string extract_path_arg(const std::string & args) {
    static const std::regex re_quoted(R"(path\s*[:=]\s*(?:<\|"\|>|")([^"<]+)(?:<\|"\|>|"))");
    static const std::regex re_bare  (R"(path\s*[:=]\s*([^,}\s]+))");
    std::smatch m;
    if (std::regex_search(args, m, re_quoted)) { return m[1].str(); }
    if (std::regex_search(args, m, re_bare))   { return m[1].str(); }
    return {};
}

// Ask the user to approve a tool call.
//   1 = allow once (default on Enter; explicit 'y')
//   2 = allow always for the rest of the session (sets `always`)
//   0 = deny (explicit 'n', EOF)
// Stdin not a tty (pipe/script) → auto-allow (1). Current tools are read-only with sandbox
// guard, so non-interactive auto-approve is safe; if write tools are added later this should
// be reconsidered.
static int prompt_tool_approval(const std::string & name, const std::string & path, bool & always) {
    if (always) { return 1; }
    if (!isatty(fileno(stdin))) {
        fprintf(stderr, "[tool] auto-allowed (non-interactive stdin): %s(path=%s)\n",
                name.c_str(), path.c_str());
        return 1;
    }
    fprintf(stderr, "\n[tool] %s(path=%s)\nallow? [Y/n/a]: ", name.c_str(), path.c_str());
    fflush(stderr);
    std::string line;
    if (!std::getline(std::cin, line)) { return 0; }
    const auto first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) { return 1; }            // empty (Enter) → allow
    const char c = (char) std::tolower((unsigned char) line[first]);
    if (c == 'n') { return 0; }
    if (c == 'a') { always = true; return 2; }
    return 1;                                                // 'y' or anything else → allow
}

// Parse the first tool-call block in `text`. Returns true if found.
static bool parse_tool_call(const std::string & text, std::string & name, std::string & args) {
    static const std::regex re(R"(<\|tool_call>\s*call:\s*(\w+)\s*\{([^]*?)\}\s*<tool_call\|>)");
    std::smatch m;
    if (!std::regex_search(text, m, re)) { return false; }
    name = m[1].str();
    args = m[2].str();
    return true;
}

// ---- tokenize / decode helpers -------------------------------------------

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    int32_t n = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, add_special, /*parse_special=*/true);
    std::vector<llama_token> out(n);
    int32_t r = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), (int32_t) out.size(), add_special, true);
    if (r < 0) { out.clear(); }
    out.resize(std::max(0, r));
    return out;
}

static std::string piece(const llama_vocab * vocab, llama_token id) {
    char buf[256];
    int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, /*special=*/true);
    if (n <= 0) { return {}; }
    return std::string(buf, n);
}

// Masked argmax for the gemma-4 assistant centroid head: pick top-k centroid clusters
// from `centroid_logits`, restrict the dense logits to those clusters' tokens, argmax.
// Equivalent to the reference Gemma4AssistantMaskedEmbedder (same per-token dot product).
static llama_token masked_argmax(
        const float * dense_logits, int n_vocab,
        const float * centroid_logits, int n_centroids, int top_k,
        const std::vector<int32_t> & token_to_cluster) {
    std::vector<int> idx(n_centroids);
    std::iota(idx.begin(), idx.end(), 0);
    const int k = std::min(top_k, n_centroids);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
            [&](int a, int b) { return centroid_logits[a] > centroid_logits[b]; });
    std::vector<uint8_t> sel(n_centroids, 0);
    for (int i = 0; i < k; ++i) { sel[idx[i]] = 1; }
    llama_token best = -1;
    float bv = -INFINITY;
    for (int v = 0; v < n_vocab; ++v) {
        if (!sel[token_to_cluster[v]]) { continue; }
        if (dense_logits[v] > bv) { bv = dense_logits[v]; best = v; }
    }
    return best;
}

static llama_token greedy(const float * logits, int n_vocab) {
    llama_token best = 0;
    float bv = logits[0];
    for (int v = 1; v < n_vocab; ++v) {
        if (logits[v] > bv) { bv = logits[v]; best = v; }
    }
    return best;
}

// Apply the gemma-4 chat template via the C API. Returns the formatted string.
static std::string apply_template(const llama_model * model, const std::vector<llama_chat_message> & msgs, bool add_ass) {
    const char * tmpl = llama_model_chat_template(model, nullptr);
    if (!tmpl) { tmpl = "gemma4"; }
    std::vector<char> buf(8192);
    int n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), add_ass, buf.data(), (int32_t) buf.size());
    if (n > (int) buf.size()) {
        buf.resize(n + 64);
        n = llama_chat_apply_template(tmpl, msgs.data(), msgs.size(), add_ass, buf.data(), (int32_t) buf.size());
    }
    if (n < 0) { fprintf(stderr, "chat template apply failed (n=%d)\n", n); return {}; }
    return std::string(buf.data(), n);
}

// ---- main -----------------------------------------------------------------

// stderr: filtreli (WARN+ veya LLAMA_VERBOSE=1 ile her şey).
// file:   her zaman TÜM log'lar (sessizden bağımsız, sonra bakmak için).
// Log dosyası main()'de açılır, user_data olarak callback'e geçer.
static void chat_log_cb(ggml_log_level level, const char * text, void * ud) {
    FILE * log_file = (FILE *) ud;
    if (log_file) { fputs(text, log_file); fflush(log_file); }
    static const bool verbose = getenv("LLAMA_VERBOSE") != nullptr;
    if (verbose) { fputs(text, stderr); return; }
    if (level >= GGML_LOG_LEVEL_WARN) { fputs(text, stderr); }
}

int main(int argc, char ** argv) {
    args a;
    if (!parse_args(argc, argv, a)) { return 1; }

    const bool sd_on = !a.model_draft.empty() && !a.no_sd;

    // Açık logging dosyası: logs/session-YYYYMMDD-HHMMSS.log
    FILE * log_file = nullptr;
    {
        mkdir("logs", 0755); // var ise -1 dönüyor, ignore
        time_t now = time(nullptr);
        struct tm tm_local;
        localtime_r(&now, &tm_local);
        char fname[256];
        strftime(fname, sizeof(fname), "logs/session-%Y%m%d-%H%M%S.log", &tm_local);
        log_file = fopen(fname, "w");
        if (log_file) {
            fprintf(log_file, "[chat] session start: %s\n", fname);
            fprintf(stderr, "[chat] full log: %s\n", fname);
        } else {
            fprintf(stderr, "[chat] warning: cannot create log file %s (errno=%d)\n", fname, errno);
        }
    }
    llama_log_set(chat_log_cb, log_file);

    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = a.n_gpu_layers;
    llama_model * model = llama_model_load_from_file(a.model.c_str(), mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const int64_t n_embd_back = llama_model_n_embd(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = a.n_ctx;
    cp.n_batch = a.n_batch;
    cp.no_perf = true;
    if (a.kv_q8) {
        cp.type_k = GGML_TYPE_Q8_0;
        cp.type_v = GGML_TYPE_Q8_0;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    // ---- SD setup (only when -md given and SD not disabled) ----
    llama_model   * draft_model = nullptr;
    llama_context * ctx_d       = nullptr;
    std::vector<int32_t> token_to_cluster;
    int32_t n_centroids = 0, top_k_centroids = 0;
    const float embed_scale = sqrtf((float) n_embd_back);
    if (sd_on) {
        // target taps post-norm hidden + last full/swa K/V
        llama_set_embeddings_pre_norm(ctx, true, /*masked=*/false);
        llama_set_assistant_kv_tap(ctx, true);

        draft_model = llama_model_load_from_file(a.model_draft.c_str(), mp);
        if (!draft_model) { fprintf(stderr, "failed to load draft model\n"); return 1; }

        llama_context_params cp_d = cp;
        cp_d.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        ctx_d = llama_init_from_model(draft_model, cp_d);
        if (!ctx_d) { fprintf(stderr, "failed to create draft context\n"); return 1; }
        llama_set_embeddings(ctx_d, true);

        // centroid head metadata
        std::vector<int32_t> tok_ordering(n_vocab);
        if (llama_model_get_token_ordering(draft_model, tok_ordering.data()) <= 0) {
            fprintf(stderr, "draft has no mtp.token_ordering — not a Gemma4 Assistant\n");
            return 1;
        }
        llama_model_get_centroid_params(draft_model, &n_centroids, &top_k_centroids);
        if (n_centroids <= 0 || top_k_centroids <= 0) {
            fprintf(stderr, "draft missing centroid head params\n"); return 1;
        }
        const int vpc = n_vocab / n_centroids;
        token_to_cluster.assign(n_vocab, 0);
        for (int i = 0; i < n_vocab; ++i) {
            const int32_t canon = tok_ordering[i];
            if (canon >= 0 && canon < n_vocab) { token_to_cluster[canon] = i / vpc; }
        }
    }

    const std::string root_abs = realpath_s(a.root);
    if (root_abs.empty()) { fprintf(stderr, "--root path does not exist: %s\n", a.root.c_str()); return 1; }
    fprintf(stderr, "[chat] model loaded. thinking=%s tools=%s root=%s sd=%s draft_max=%d\n",
            a.thinking ? "on" : "off", a.no_tools ? "off" : "on", root_abs.c_str(),
            sd_on ? "on" : "off", a.n_draft_max);

    llama_batch batch   = llama_batch_init(a.n_batch, 0, 1);
    llama_batch batch_d = sd_on ? llama_batch_init(1, (int32_t)(2 * n_embd_back), 1) : llama_batch{};

    // history (system message is constructed once, contains tools/thinking markers)
    const std::string system_text = build_system_content(!a.no_tools, a.thinking);
    std::deque<std::string>           msg_storage;  // deque: push_back never invalidates existing c_str() pointers
    std::vector<llama_chat_message>   msgs;
    auto push_msg = [&](const std::string & role, const std::string & content) {
        msg_storage.push_back(role);
        msg_storage.push_back(content);
        const char * r = msg_storage[msg_storage.size() - 2].c_str();
        const char * c = msg_storage[msg_storage.size() - 1].c_str();
        msgs.push_back({ r, c });
    };
    push_msg("system", system_text);

    int kv_pos = 0;
    std::string last_formatted;
    bool session_always_allow = a.yolo;   // --yolo → session starts in auto-approve mode

    // SD-only persistent state (turn-arası korunur, draft modeli prompt context'i olarak kullanır)
    std::vector<float> acc_kf, acc_vf, acc_ks, acc_vs;
    int64_t acc_nkv = 0;
    int64_t nhkv = 0, hd_full = 0, hd_swa = 0;
    std::vector<float> pending_feat(sd_on ? n_embd_back : 0);
    std::vector<float> concat_buf(sd_on ? 2 * n_embd_back : 0);
    std::vector<float> cur_feat(sd_on ? n_embd_back : 0);

    // per-turn stats (reset on user turn start, printed on turn end)
    // We measure prefill (template + tail decode) and generation (sample/SD loop) separately.
    // "tokens / gen_ms" is the apples-to-apples speed; total wall-clock includes prefill/tools.
    int64_t turn_emitted = 0, turn_rounds = 0, turn_drafted = 0, turn_accepted = 0;
    int64_t turn_prefill_tok = 0;
    int64_t turn_t_us_start = 0;
    int64_t turn_prefill_us = 0;
    int64_t turn_gen_us     = 0;

    // Reset target+draft KV state and shrink the chat history down to {system, last_user}.
    // Used when (a) the KV cache is about to overflow (proactive) or (b) a decode failed
    // mid-turn and the context is in an unknown state (reactive). System message + the most
    // recent user message are preserved so the model can answer the current query fresh.
    // keep_last_user=true (proactive case): user msg pushed but not yet processed → keep it
    // keep_last_user=false (reactive case): user msg ALREADY processed (and the turn died).
    //   Drop everything except system; next REPL iteration will push a fresh user msg.
    auto reset_chat = [&](bool keep_last_user) {
        std::string sys_content, usr_content;
        if (msg_storage.size() >= 2 && msg_storage[0] == "system") {
            sys_content = msg_storage[1];
        }
        if (keep_last_user) {
            for (size_t i = msg_storage.size(); i >= 2; i -= 2) {
                if (msg_storage[i - 2] == "user") {
                    usr_content = msg_storage[i - 1];
                    break;
                }
                if (i < 2) { break; }
            }
        }
        msg_storage.clear();
        msgs.clear();
        if (!sys_content.empty()) { push_msg("system", sys_content); }
        if (!usr_content.empty()) { push_msg("user",   usr_content); }

        llama_memory_seq_rm(llama_get_memory(ctx), /*seq_id=*/0, 0, -1);
        if (sd_on && ctx_d) { llama_memory_seq_rm(llama_get_memory(ctx_d), 0, 0, -1); }
        kv_pos = 0;
        last_formatted.clear();
        acc_kf.clear(); acc_vf.clear(); acc_ks.clear(); acc_vs.clear();
        acc_nkv = 0;
    };

    auto run_inference = [&](int max_new_tokens, std::string & assistant_text) -> bool {
        // Build full prompt, identify the unprocessed tail, decode it, then sample.
        const std::string formatted = apply_template(model, msgs, /*add_ass=*/true);
        if (formatted.empty()) { return false; }

        // The tail we still need to feed: everything past what's already in the KV cache.
        // We approximate by tokenizing both, diffing on token sequences would be O(N);
        // simpler: tokenize the tail string (formatted.substr(last_formatted.size())).
        // The first call has empty last_formatted, so the whole prompt is fed.
        const std::string tail = formatted.substr(std::min(last_formatted.size(), formatted.size()));
        // gemma-4 apply branch zaten "<bos>" literal'ı ekliyor + parse_special=true onu BOS
        // token'a çevirir. add_special=true ayrıca BOS eklerse çift olur → daima false.
        std::vector<llama_token> tail_tokens = tokenize(vocab, tail, /*add_special=*/false);
        if (tail_tokens.empty()) { fprintf(stderr, "[chat] empty tail tokenization\n"); return false; }

        // feed in chunks of n_batch. Each llama_decode resets the assistant K/V tap, so on
        // SD mode we must drain the tap into the persistent accumulator AFTER EVERY chunk
        // (not once at the end — that would only capture the last chunk's positions).
        const int64_t prefill_t0 = ggml_time_us();
        // count tokens AFTER each successful chunk so the rate stays honest if decode fails partway
        for (size_t off = 0; off < tail_tokens.size(); ) {
            const int n = (int) std::min<size_t>(a.n_batch, tail_tokens.size() - off);
            batch.n_tokens = 0;
            for (int i = 0; i < n; ++i) {
                batch.token[i]     = tail_tokens[off + i];
                batch.pos[i]       = kv_pos + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = (off + i + 1 == tail_tokens.size()) ? 1 : 0;
                batch.n_tokens++;
            }
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "[chat] decode failed\n"); return false; }
            kv_pos += n;
            off    += n;
            turn_prefill_tok += n;   // count only what actually completed

            if (sd_on) {
                const float *kf=nullptr,*vf=nullptr,*ks=nullptr,*vs=nullptr,*hid=nullptr;
                int64_t n_tap = llama_get_assistant_kv_tap(ctx, &kf,&vf,&ks,&vs,&hid, &nhkv,&hd_full,&hd_swa);
                if (n_tap != (int64_t) n || !kf) {
                    fprintf(stderr, "[chat] tap mismatch: tap=%lld chunk=%d\n", (long long) n_tap, n);
                    return false;
                }
                acc_kf.insert(acc_kf.end(), kf, kf + n_tap * hd_full * nhkv);
                acc_vf.insert(acc_vf.end(), vf, vf + n_tap * hd_full * nhkv);
                acc_ks.insert(acc_ks.end(), ks, ks + n_tap * hd_swa  * nhkv);
                acc_vs.insert(acc_vs.end(), vs, vs + n_tap * hd_swa  * nhkv);
                // last chunk's last row becomes the pending feature for the upcoming draft
                std::memcpy(pending_feat.data(), hid + (n_tap - 1) * n_embd_back, n_embd_back * sizeof(float));
                acc_nkv += n_tap;
            }
        }

        turn_prefill_us += ggml_time_us() - prefill_t0;
        const int64_t gen_t0 = ggml_time_us();

        assistant_text.clear();

        if (!sd_on) {
            // === GREEDY PATH ===
            for (int t = 0; t < max_new_tokens; ++t) {
                const float * lg = llama_get_logits_ith(ctx, batch.n_tokens - 1);
                llama_token id = greedy(lg, n_vocab);
                if (llama_vocab_is_eog(vocab, id)) { break; }

                const std::string p = piece(vocab, id);
                assistant_text += p;
                std::cout << p;
                std::cout.flush();
                turn_emitted++;

                batch.n_tokens     = 1;
                batch.token[0]     = id;
                batch.pos[0]       = kv_pos;
                batch.n_seq_id[0]  = 1;
                batch.seq_id[0][0] = 0;
                batch.logits[0]    = 1;
                if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "[chat] decode failed\n"); return false; }
                kv_pos++;

                if (assistant_text.find("<tool_call|>") != std::string::npos) { break; }
            }
        } else {
            // === SPECULATIVE DECODING PATH ===
            // First token: target's last prefill logit (this is what the assistant turn opens with).
            llama_token pending_tok = greedy(llama_get_logits_ith(ctx, batch.n_tokens - 1), n_vocab);
            bool tool_seen = false;
            bool eog       = false;
            auto emit = [&](llama_token id) {
                if (llama_vocab_is_eog(vocab, id)) { eog = true; return; }
                const std::string p = piece(vocab, id);
                assistant_text += p;
                std::cout << p; std::cout.flush();
                turn_emitted++;
                if (assistant_text.find("<tool_call|>") != std::string::npos) { tool_seen = true; }
            };
            emit(pending_tok);

            while (!eog && !tool_seen && turn_emitted < max_new_tokens) {
                // ---- 1) draft up to N tokens with the assistant ----
                llama_set_assistant_shared_kv(ctx_d, nhkv,
                        hd_full, acc_kf.data(), acc_vf.data(), acc_nkv,
                        hd_swa,  acc_ks.data(), acc_vs.data(), acc_nkv);
                llama_token cur_tok = pending_tok;
                std::memcpy(cur_feat.data(), pending_feat.data(), n_embd_back * sizeof(float));
                const llama_pos draft_pos = (llama_pos) acc_nkv;

                std::vector<llama_token> drafted;
                drafted.reserve(a.n_draft_max);
                for (int i = 0; i < a.n_draft_max; ++i) {
                    llama_model_get_token_embd(model, cur_tok, concat_buf.data());
                    for (int j = 0; j < n_embd_back; ++j) { concat_buf[j] *= embed_scale; }
                    std::memcpy(concat_buf.data() + n_embd_back, cur_feat.data(), n_embd_back * sizeof(float));

                    batch_d.n_tokens     = 1;
                    std::memcpy(batch_d.embd, concat_buf.data(), concat_buf.size() * sizeof(float));
                    batch_d.pos[0]       = draft_pos;
                    batch_d.n_seq_id[0]  = 1;
                    batch_d.seq_id[0][0] = 0;
                    batch_d.logits[0]    = 1;
                    if (llama_decode(ctx_d, batch_d) != 0) { fprintf(stderr, "[chat] draft decode failed\n"); return false; }

                    const float * lg  = llama_get_logits_ith(ctx_d, 0);
                    const float * emb = llama_get_embeddings_ith(ctx_d, 0);
                    if (!lg || !emb) { break; }
                    llama_token id = masked_argmax(lg, n_vocab, emb + n_embd_back, n_centroids, top_k_centroids, token_to_cluster);
                    if (id < 0) { break; }
                    drafted.push_back(id);
                    std::memcpy(cur_feat.data(), emb, n_embd_back * sizeof(float));
                    cur_tok = id;
                }
                turn_drafted += (int64_t) drafted.size();
                turn_rounds++;

                // ---- 2) verify on target: [pending_tok, draft_1..N] ----
                batch.n_tokens = 0;
                auto push_v = [&](llama_token id, llama_pos pos) {
                    const int n = batch.n_tokens;
                    batch.token[n]     = id;
                    batch.pos[n]       = pos;
                    batch.n_seq_id[n]  = 1;
                    batch.seq_id[n][0] = 0;
                    batch.logits[n]    = 1;
                    batch.n_tokens     = n + 1;
                };
                push_v(pending_tok, (llama_pos) acc_nkv);
                for (size_t i = 0; i < drafted.size(); ++i) {
                    push_v(drafted[i], (llama_pos) (acc_nkv + 1 + (llama_pos) i));
                }
                if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "[chat] verify decode failed\n"); return false; }

                // tap for this verify (per-decode reset)
                const float *kf=nullptr,*vf=nullptr,*ks=nullptr,*vs=nullptr,*hid=nullptr;
                int64_t n_tap = llama_get_assistant_kv_tap(ctx, &kf,&vf,&ks,&vs,&hid, nullptr,nullptr,nullptr);
                if (n_tap != (int64_t) batch.n_tokens || !kf) {
                    fprintf(stderr, "[chat] verify tap mismatch\n"); return false;
                }

                // ---- 3) accept longest matching prefix; pick bonus from first mismatch ----
                int n_accept = 0;
                llama_token next_pending = -1;
                for (int p = 0; p < batch.n_tokens; ++p) {
                    llama_token t = greedy(llama_get_logits_ith(ctx, p), n_vocab);
                    if (p < (int) drafted.size() && t == drafted[p]) { n_accept++; }
                    else { next_pending = t; break; }
                }
                turn_accepted += n_accept;
                const int64_t n_keep = (int64_t) n_accept + 1;

                // ---- 4) rollback target KV for rejected positions ----
                if (batch.n_tokens > n_keep) {
                    llama_memory_seq_rm(llama_get_memory(ctx), 0, (llama_pos) (acc_nkv + n_keep), -1);
                }

                // ---- 5) emit accepted draft tokens (check tool/eog per emit) ----
                for (int i = 0; i < n_accept && !eog && !tool_seen && turn_emitted < max_new_tokens; ++i) {
                    emit(drafted[i]);
                }
                if (!eog && !tool_seen && turn_emitted < max_new_tokens) {
                    if (next_pending < 0) {
                        next_pending = greedy(llama_get_logits_ith(ctx, batch.n_tokens - 1), n_vocab);
                    }
                    emit(next_pending);
                }

                // ---- 6) append accepted positions' K/V to acc + advance pending state ----
                for (int64_t p = 0; p < n_keep; ++p) {
                    acc_kf.insert(acc_kf.end(), kf + p*hd_full*nhkv, kf + (p+1)*hd_full*nhkv);
                    acc_vf.insert(acc_vf.end(), vf + p*hd_full*nhkv, vf + (p+1)*hd_full*nhkv);
                    acc_ks.insert(acc_ks.end(), ks + p*hd_swa *nhkv, ks + (p+1)*hd_swa *nhkv);
                    acc_vs.insert(acc_vs.end(), vs + p*hd_swa *nhkv, vs + (p+1)*hd_swa *nhkv);
                }
                acc_nkv += n_keep;
                kv_pos  += (int) n_keep;
                std::memcpy(pending_feat.data(), hid + (n_keep - 1) * n_embd_back, n_embd_back * sizeof(float));
                pending_tok = next_pending;
            }

            // KV alignment after tool_seen: the emitted <tool_call|> token may not be in the
            // target KV (Case B: it was `next_pending`, never decoded) or the KV may have ghost
            // tokens after it (Case D: accepted drafts past <tool_call|> were also decoded).
            // Recompute the correct KV depth from the actual emitted text and fix both cases.
            if (tool_seen) {
                const std::string lf   = formatted + assistant_text;
                const auto lf_toks     = tokenize(vocab, lf, /*add_special=*/true);
                const int64_t want_kv  = (int64_t) lf_toks.size();
                if (kv_pos > want_kv) {
                    // Ghost tokens: trim target KV and acc vectors
                    llama_memory_seq_rm(llama_get_memory(ctx), 0, (llama_pos) want_kv, -1);
                    const int64_t trim = kv_pos - want_kv;
                    if (nhkv > 0 && hd_full > 0 && hd_swa > 0) {
                        acc_kf.resize(acc_kf.size() - (size_t)(trim * hd_full * nhkv));
                        acc_vf.resize(acc_vf.size() - (size_t)(trim * hd_full * nhkv));
                        acc_ks.resize(acc_ks.size() - (size_t)(trim * hd_swa  * nhkv));
                        acc_vs.resize(acc_vs.size() - (size_t)(trim * hd_swa  * nhkv));
                    }
                    kv_pos = acc_nkv = want_kv;
                } else if (kv_pos < want_kv) {
                    // Missing tokens: decode them and accumulate their K/V tap
                    for (int64_t pos = kv_pos; pos < want_kv; ++pos) {
                        batch.n_tokens     = 1;
                        batch.token[0]     = lf_toks[pos];
                        batch.pos[0]       = (llama_pos) pos;
                        batch.n_seq_id[0]  = 1;
                        batch.seq_id[0][0] = 0;
                        batch.logits[0]    = 0;
                        if (llama_decode(ctx, batch) != 0) { return false; }
                        if (nhkv > 0 && hd_full > 0 && hd_swa > 0) {
                            const float *kf2=nullptr,*vf2=nullptr,*ks2=nullptr,*vs2=nullptr,*hid2=nullptr;
                            int64_t n2 = llama_get_assistant_kv_tap(ctx, &kf2,&vf2,&ks2,&vs2,&hid2, nullptr,nullptr,nullptr);
                            if (n2 == 1 && kf2) {
                                acc_kf.insert(acc_kf.end(), kf2, kf2 + hd_full * nhkv);
                                acc_vf.insert(acc_vf.end(), vf2, vf2 + hd_full * nhkv);
                                acc_ks.insert(acc_ks.end(), ks2, ks2 + hd_swa  * nhkv);
                                acc_vs.insert(acc_vs.end(), vs2, vs2 + hd_swa  * nhkv);
                                acc_nkv++;
                            }
                        }
                        kv_pos++;
                    }
                }
            }
        }

        turn_gen_us += ggml_time_us() - gen_t0;

        // update last_formatted to include the new assistant tokens we just emitted
        last_formatted = formatted + assistant_text;
        return true;
    };

    // Helper to filter the thinking <|channel>...<channel|> block from displayed/history text.
    // The model still produced it (and the KV holds it), but we strip it from the persisted
    // assistant content so subsequent turns aren't polluted; thinking is per-turn.
    auto strip_thinking = [](const std::string & s) -> std::string {
        const std::string open  = "<|channel>";
        const std::string close = "<channel|>";
        std::string out = s;
        for (;;) {
            const auto a = out.find(open);
            if (a == std::string::npos) { break; }
            const auto b = out.find(close, a);
            if (b == std::string::npos) { out.erase(a); break; }
            out.erase(a, (b + close.size()) - a);
        }
        return out;
    };

    // REPL
    std::cout << "[chat] ready. type your message (Ctrl-D to exit).\n";
    while (true) {
        std::cout << "\n> ";
        std::cout.flush();
        std::string line;
        if (!std::getline(std::cin, line)) { std::cout << "\n"; break; }
        if (line.empty()) { continue; }

        push_msg("user", line);

        // Proactive: if last turn's accumulated context is nearing capacity, rotate now —
        // BEFORE we try to feed an arbitrarily long tail (e.g. a fresh tool result). 1024
        // tokens of headroom leaves room for the user msg, assistant prefix, and a few
        // SD verify batches.
        if (acc_nkv > (int64_t) a.n_ctx - 1024) {
            fprintf(stderr, "[chat] context near full (acc=%lld / ctx=%d) -> rotating history (keeping system + last user)\n",
                    (long long) acc_nkv, a.n_ctx);
            reset_chat(/*keep_last_user=*/true);
        }

        // per-turn stats: total across tool hops
        turn_emitted = turn_rounds = turn_drafted = turn_accepted = 0;
        turn_prefill_tok = 0;
        turn_prefill_us = turn_gen_us = 0;
        turn_t_us_start = ggml_time_us();

        // run model, possibly looping for tool calls
        std::string assistant_raw;
        bool decode_broke = false;
        for (int hop = 0; hop < 4; ++hop) { // cap tool-call hops to avoid infinite loops
            assistant_raw.clear();
            if (!run_inference(a.n_predict, assistant_raw)) { decode_broke = true; break; }

            // detect tool call
            std::string tname, targs;
            if (a.no_tools || !parse_tool_call(assistant_raw, tname, targs)) {
                // plain assistant turn: commit RAW. Stripping thinking from history would
                // shorten `last_formatted` vs the next turn's formatted prefix and corrupt
                // the incremental-decode diff. Gemma's <|channel> blocks are designed to live
                // in context anyway.
                push_msg("assistant", assistant_raw);
                break;
            }

            const std::string path = extract_path_arg(targs);
            const int verdict = prompt_tool_approval(tname, path, session_always_allow);
            std::string result;
            if (verdict == 0) {
                fprintf(stderr, "[tool] denied by user\n");
                result = "error: user denied this tool call";
            } else {
                if (verdict == 2) { fprintf(stderr, "[tool] always-allow enabled for this session\n"); }
                if (tname == "read_file") {
                    result = tool_read_file(path, root_abs);
                } else if (tname == "list_dir") {
                    result = tool_list_dir(path, root_abs);
                } else {
                    result = "error: unknown tool";
                }
            }
            // record the assistant's tool-call message RAW (model needs the exact tokens it
            // emitted in its history — and any strip would break the incremental-decode diff)
            push_msg("assistant", assistant_raw);
            // record the tool's response; the gemma4 template applier wraps this in
            // <|tool_response>...<tool_response|>
            push_msg("tool", tname + "{" + result + "}");
            // re-prompt: loop
        }

        // Reactive recovery: if any decode failed during this turn (KV exhausted, allocator
        // issue, etc.) the context is in an unknown state. Rotate down to {system, last user}
        // and clear KV so the NEXT turn can start fresh instead of cascading failures.
        if (decode_broke) {
            fprintf(stderr, "[chat] decode failed during turn -> resetting (system only); next prompt starts fresh\n");
            reset_chat(/*keep_last_user=*/false);
        }

        // UX: if the model produced nothing (first sampled token was EOG and the turn ended
        // empty), make it visible — otherwise the user sees only the next prompt and may
        // think the binary hung.
        if (!decode_broke && turn_emitted == 0) {
            fprintf(stderr, "[chat] (model produced no output for this turn)\n");
        }

        // turn-end stats. We report generation t/s (pure decode time) — the apples-to-apples
        // metric across short/long turns. Prefill t/s is separate; total wall-clock includes
        // tool dispatch and other host overhead.
        const double total_s   = (ggml_time_us() - turn_t_us_start) / 1e6;
        const double prefill_s = turn_prefill_us / 1e6;
        const double gen_s     = turn_gen_us / 1e6;
        const double prefill_tps = prefill_s > 0 ? (double) turn_prefill_tok / prefill_s : 0.0;
        const double gen_tps     = gen_s     > 0 ? (double) turn_emitted     / gen_s     : 0.0;
        if (sd_on) {
            fprintf(stderr,
                "\n[stats] gen=%lld tok / %.2fs = %.1f t/s | prefill=%lld tok / %.2fs = %.0f t/s "
                "| rounds=%lld drafted=%lld accepted=%lld acc_rate=%.1f%% | total=%.2fs\n",
                (long long) turn_emitted, gen_s, gen_tps,
                (long long) turn_prefill_tok, prefill_s, prefill_tps,
                (long long) turn_rounds, (long long) turn_drafted, (long long) turn_accepted,
                turn_drafted ? 100.0 * (double) turn_accepted / (double) turn_drafted : 0.0,
                total_s);
        } else {
            fprintf(stderr,
                "\n[stats] gen=%lld tok / %.2fs = %.1f t/s | prefill=%lld tok / %.2fs = %.0f t/s "
                "| total=%.2fs\n",
                (long long) turn_emitted, gen_s, gen_tps,
                (long long) turn_prefill_tok, prefill_s, prefill_tps,
                total_s);
        }
    }

    if (ctx_d)        { llama_free(ctx_d); }
    if (draft_model)  { llama_model_free(draft_model); }
    if (batch_d.token || batch_d.embd) { llama_batch_free(batch_d); }
    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    if (log_file) {
        fprintf(log_file, "[chat] session end\n");
        fclose(log_file);
    }
    return 0;
}
