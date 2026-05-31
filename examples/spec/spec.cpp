// Minimal speculative decoding example for Gemma 4 + Gemma 4 Assistant (MTP draft head).
//
// Two models share the same vocab. The big target verifies multiple tokens at once
// while the small assistant drafts them — fewer target forwards per emitted token.
//
// Pipeline:
//   1. Tokenize prompt (using the target's chat template, see args).
//   2. Prefill the target on the prompt. The target's tap emits per-token:
//        - K/V for the last full-attn + last sliding-attn has_kv layer (cross-attn input)
//        - post-norm hidden (consumed by the assistant's concat input)
//   3. Sample the first token from the target's last prompt logits.
//   4. Round loop:
//        - Run the assistant for N draft steps. Each step consumes
//            concat[ embed(cur_tok) * sqrt(backbone) , cur_feat ]
//          and outputs the chained `feature` (post-projection) + centroid logits.
//          Pick the next token via the masked-embedding head host-side:
//            top-K centroid clusters of centroid_logits → restrict dense logits
//            to those clusters' tokens → argmax.
//        - Build a verify batch [first_gen, d_1, ..., d_N] and decode the target once.
//        - For each position, argmax the target's logits and compare to the draft.
//          Accept the longest matching prefix and the bonus token at the first mismatch.
//        - Append accepted positions' K/V tap to acc; discard the rejected tail.
//   5. Stop on EOG or after the requested number of generated tokens.

#include "llama.h"
#include "../../src/llama-ext.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <future>
#include <numeric>
#include <string>
#include <vector>

struct args {
    std::string model_target;
    std::string model_draft;
    std::string prompt = "hello";
    int  n_predict         = 128;
    int  n_gpu_layers       = 99;   // target ngl
    int  n_gpu_layers_draft = 0;    // draft ngl (default CPU — enables true paral. with --ssd-async)
    int  n_draft_max       = 5;
    int  n_ctx             = 4096;
    int  n_batch           = 2048;
    bool apply_chat        = true;

    // SSD (Speculative Speculative Decoding, arXiv 2603.03251v3) options.
    // ssd_fan_out = total budget B across draft positions (Thm 12). B=1 disables SSD.
    int   ssd_fan_out  = 1;
    float ssd_r        = 1.0f;   // power-law exponent for geometric fan-out
    float ssd_C        = 1.0f;   // Saguaro sampling downweight (effective only under non-greedy)
    bool  ssd_async    = false;  // overlap target verify with SSD post-pass via std::thread
};

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "\nusage:\n  %s -m <target.gguf> -md <draft.gguf> [-p <prompt>] [-n N] "
        "[--draft-max N] [-ngl N] [-ngl-draft N] [-c N] [-b N] [--no-chat]\n"
        "       [--ssd-fan-out B] [--ssd-r R] [--ssd-sampling-C C] [--ssd-async]\n\n"
        "Backend split:\n"
        "  -ngl N        target GPU layers (default 99 = full Metal)\n"
        "  -ngl-draft N  draft  GPU layers (default 0 = CPU NEON; required for --ssd-async)\n\n"
        "SSD options (paper arXiv 2603.03251v3):\n"
        "  --ssd-fan-out B    total fan-out budget across draft positions (default 1 = SSD off)\n"
        "  --ssd-r R          power-law exponent for geometric fan-out (default 1.0)\n"
        "  --ssd-sampling-C C Saguaro sampling downweight (default 1.0 = off; greedy mode warns)\n"
        "  --ssd-async        overlap target verify with SSD post-pass via std::thread\n"
        "                     (only helpful when draft+target are on different backends)\n\n",
        argv0);
}

static bool parse_args(int argc, char ** argv, args & a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto need = [&](const char * what) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", what); return nullptr; }
            return argv[++i];
        };
        if (s == "-m")  { auto v = need("-m");  if (!v) return false; a.model_target = v; }
        else if (s == "-md") { auto v = need("-md"); if (!v) return false; a.model_draft = v; }
        else if (s == "-p")  { auto v = need("-p");  if (!v) return false; a.prompt = v; }
        else if (s == "-n")  { auto v = need("-n");  if (!v) return false; a.n_predict = atoi(v); }
        else if (s == "--draft-max") { auto v = need("--draft-max"); if (!v) return false; a.n_draft_max = atoi(v); }
        else if (s == "-ngl") { auto v = need("-ngl"); if (!v) return false; a.n_gpu_layers = atoi(v); }
        else if (s == "-ngl-draft" || s == "--ngl-draft") { auto v = need("-ngl-draft"); if (!v) return false; a.n_gpu_layers_draft = atoi(v); }
        else if (s == "-c")   { auto v = need("-c");   if (!v) return false; a.n_ctx   = atoi(v); }
        else if (s == "-b")   { auto v = need("-b");   if (!v) return false; a.n_batch = atoi(v); }
        else if (s == "--no-chat") { a.apply_chat = false; }
        else if (s == "--ssd-fan-out") { auto v = need("--ssd-fan-out"); if (!v) return false; a.ssd_fan_out = atoi(v); }
        else if (s == "--ssd-r")        { auto v = need("--ssd-r");        if (!v) return false; a.ssd_r        = (float) atof(v); }
        else if (s == "--ssd-sampling-C") { auto v = need("--ssd-sampling-C"); if (!v) return false; a.ssd_C = (float) atof(v); }
        else if (s == "--ssd-async") { a.ssd_async = true; }
        else if (s == "-h" || s == "--help") { print_usage(argv[0]); return false; }
        else { fprintf(stderr, "unknown arg: %s\n", s.c_str()); print_usage(argv[0]); return false; }
    }
    if (a.model_target.empty() || a.model_draft.empty()) {
        print_usage(argv[0]);
        return false;
    }
    return true;
}

// Tokenize `text` using the target's vocab. add_special: BOS/EOS, parse_special: handle <|...|> tokens.
static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special, bool parse_special) {
    int32_t n = -llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), nullptr, 0, add_special, parse_special);
    std::vector<llama_token> out(n);
    int32_t r = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), (int32_t) out.size(), add_special, parse_special);
    if (r < 0) { out.clear(); }
    out.resize(std::max(0, r));
    return out;
}

static std::string token_to_piece(const llama_vocab * vocab, llama_token id) {
    char buf[256];
    int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    if (n <= 0) { return {}; }
    return std::string(buf, n);
}

// Append a token sequence to the print stream.
static void emit_token(const llama_vocab * vocab, llama_token id) {
    auto s = token_to_piece(vocab, id);
    printf("%s", s.c_str());
    fflush(stdout);
}

// Pick the top-F masked tokens: top-k clusters of centroid_logits → restrict dense logits
// to the tokens of those clusters → return the F highest-logit tokens (out_tokens[0] = argmax).
// Mathematically equivalent to the reference Gemma4AssistantMaskedEmbedder head,
// since per-token logit = hidden . embedding in both.
static int masked_topf(
        const float * dense_logits, int n_vocab,
        const float * centroid_logits, int n_centroids, int top_k,
        const std::vector<int32_t> & token_to_cluster,
        int F, std::vector<llama_token> & out_tokens) {
    out_tokens.clear();
    if (F <= 0) { return 0; }

    std::vector<int> idx(n_centroids);
    std::iota(idx.begin(), idx.end(), 0);
    const int k = std::min(top_k, n_centroids);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
            [&](int a, int b) { return centroid_logits[a] > centroid_logits[b]; });
    std::vector<uint8_t> sel(n_centroids, 0);
    for (int i = 0; i < k; ++i) { sel[idx[i]] = 1; }

    // Gather allowed (token, logit) pairs then partial_sort by logit desc.
    std::vector<std::pair<float, llama_token>> cand;
    cand.reserve(n_vocab / 4);
    for (int v = 0; v < n_vocab; ++v) {
        if (!sel[token_to_cluster[v]]) { continue; }
        cand.emplace_back(dense_logits[v], (llama_token) v);
    }
    const int take = std::min(F, (int) cand.size());
    std::partial_sort(cand.begin(), cand.begin() + take, cand.end(),
            [](const std::pair<float, llama_token> & a, const std::pair<float, llama_token> & b) {
                return a.first > b.first;
            });
    out_tokens.reserve(take);
    for (int i = 0; i < take; ++i) { out_tokens.push_back(cand[i].second); }
    return take;
}

// Single-best wrapper (preserves the original call sites).
static llama_token masked_argmax(
        const float * dense_logits, int n_vocab,
        const float * centroid_logits, int n_centroids, int top_k,
        const std::vector<int32_t> & token_to_cluster) {
    std::vector<llama_token> out;
    int got = masked_topf(dense_logits, n_vocab, centroid_logits, n_centroids, top_k, token_to_cluster, 1, out);
    return got > 0 ? out[0] : -1;
}

// Geometric fan-out (Saguaro Thm 12): split a budget B across positions k in [0,K),
// with F_k roughly proportional to a_p^(k/(1+r)). Each F_k >= 1, sum(F_k) <= B.
// Returns vector of size K. If B <= K, returns all-ones (no headroom for alternatives).
static std::vector<int> geometric_fanout(int B, int K, float a_p, float r) {
    std::vector<int> F(K, 1);
    if (B <= K || K <= 0) { return F; }
    a_p = std::max(0.05f, std::min(0.99f, a_p));
    r   = std::max(0.0f, r);

    // Raw geometric weights w_k = a_p^(k/(1+r)). Pick F_0 so sum(round(F_0 * w_k)) ~= B.
    std::vector<double> w(K);
    double wsum = 0.0;
    for (int k = 0; k < K; ++k) {
        w[k] = std::pow((double) a_p, (double) k / (1.0 + (double) r));
        wsum += w[k];
    }
    const double F0 = (double) B / wsum;
    int used = 0;
    for (int k = 0; k < K; ++k) {
        int fk = std::max(1, (int) std::lround(F0 * w[k]));
        F[k] = fk;
        used += fk;
    }
    // Trim from the tail (smallest positions in the geometric) until under budget.
    for (int k = K - 1; k >= 0 && used > B; --k) {
        while (F[k] > 1 && used > B) { F[k]--; used--; }
    }
    return F;
}

// Greedy argmax over a dense logit vector.
static llama_token greedy(const float * logits, int n_vocab) {
    llama_token best = 0;
    float bv = logits[0];
    for (int v = 1; v < n_vocab; ++v) {
        if (logits[v] > bv) { bv = logits[v]; best = v; }
    }
    return best;
}

// Quiet log callback: drop GGML_LOG_LEVEL_DEBUG noise (MTP draft prints "calling encode()"
// on every decode otherwise — looks like a crash). Anything INFO and above still prints.
static void spec_quiet_log(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    if (level <= GGML_LOG_LEVEL_DEBUG) { return; }
    fputs(text, stderr);
}

int main(int argc, char ** argv) {
    args a;
    if (!parse_args(argc, argv, a)) { return 1; }

    llama_log_set(spec_quiet_log, nullptr);
    ggml_backend_load_all();

    // ---- load models ----
    // Per-model GPU layer config: target stays on Metal, draft can be CPU (default 0).
    // CPU-side draft enables real verify/spec overlap under --ssd-async.
    llama_model_params mp_tgt = llama_model_default_params();
    mp_tgt.n_gpu_layers = a.n_gpu_layers;
    llama_model_params mp_drf = llama_model_default_params();
    mp_drf.n_gpu_layers = a.n_gpu_layers_draft;

    fprintf(stderr, "[spec] backends: target ngl=%d (%s)   draft ngl=%d (%s)\n",
            a.n_gpu_layers,       a.n_gpu_layers       > 0 ? "GPU" : "CPU",
            a.n_gpu_layers_draft, a.n_gpu_layers_draft > 0 ? "GPU" : "CPU");

    llama_model * tgt = llama_model_load_from_file(a.model_target.c_str(), mp_tgt);
    if (!tgt) { fprintf(stderr, "failed to load target model: %s\n", a.model_target.c_str()); return 1; }
    llama_model * drf = llama_model_load_from_file(a.model_draft.c_str(),  mp_drf);
    if (!drf) { fprintf(stderr, "failed to load draft model:  %s\n",  a.model_draft.c_str()); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(tgt);
    const int n_vocab          = llama_vocab_n_tokens(vocab);
    const int64_t n_embd_back  = llama_model_n_embd(tgt); // == n_embd_backbone for the assistant
    const float embed_scale    = sqrtf((float) n_embd_back); // Gemma4TextScaledWordEmbedding

    // ---- draft-side metadata: token_ordering -> cluster map + centroid head dims ----
    std::vector<int32_t> tok_ordering(n_vocab);
    {
        int32_t got = llama_model_get_token_ordering(drf, tok_ordering.data());
        if (got <= 0) { fprintf(stderr, "draft has no mtp.token_ordering tensor — not a Gemma4 Assistant\n"); return 1; }
    }
    int32_t n_centroids = 0, top_k = 0;
    llama_model_get_centroid_params(drf, &n_centroids, &top_k);
    if (n_centroids <= 0 || top_k <= 0) {
        fprintf(stderr, "draft missing centroid head params (n_centroids=%d, top_k=%d)\n", n_centroids, top_k);
        return 1;
    }
    const int vpc = n_vocab / n_centroids; // vocab per centroid
    std::vector<int32_t> token_to_cluster(n_vocab, 0);
    for (int i = 0; i < n_vocab; ++i) {
        int32_t canon = tok_ordering[i];
        if (canon >= 0 && canon < n_vocab) {
            token_to_cluster[canon] = i / vpc;
        }
    }

    // ---- contexts ----
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = a.n_ctx;
    cp.n_batch = a.n_batch;
    cp.no_perf = true;

    llama_context * ctx_t = llama_init_from_model(tgt, cp);
    if (!ctx_t) { fprintf(stderr, "failed to create target context\n"); return 1; }

    // target taps the post-norm hidden (`embeddings_pre_norm` API is a misnomer; here it means
    // emit a per-token output buffer, which the gemma4 tap relies on) + last-layer K/V.
    llama_set_embeddings_pre_norm(ctx_t, true, /*masked=*/false);
    llama_set_assistant_kv_tap   (ctx_t, true);

    // draft is an MTP-type context; force ctx_type=MTP so MTP-specific init runs.
    llama_context_params cp_d = cp;
    cp_d.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    llama_context * ctx_d = llama_init_from_model(drf, cp_d);
    if (!ctx_d) { fprintf(stderr, "failed to create draft context\n"); return 1; }
    // draft outputs the post-projection feature + centroid logits via embeddings.
    llama_set_embeddings(ctx_d, true);

    // ---- tokenize prompt ----
    // Manual Gemma chat template (the gguf's full jinja with thinking-mode etc. is too rich
    // for the minimal C-side parser). Use the standard non-thinking form:
    //   <start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
    std::vector<llama_token> input;
    if (a.apply_chat) {
        std::string s = "<start_of_turn>user\n" + a.prompt + "<end_of_turn>\n<start_of_turn>model\n";
        input = tokenize(vocab, s, /*add_special=*/true, /*parse_special=*/true);
    } else {
        input = tokenize(vocab, a.prompt, true, true);
    }
    if (input.empty()) { fprintf(stderr, "empty tokenization\n"); return 1; }

    fprintf(stderr, "[spec] prompt tokens=%zu n_predict=%d n_draft_max=%d n_centroids=%d top_k=%d backbone=%lld\n",
            input.size(), a.n_predict, a.n_draft_max, n_centroids, top_k, (long long) n_embd_back);

    // SSD config sanity (paper arXiv 2603.03251v3).
    const bool ssd_enabled = a.ssd_fan_out > a.n_draft_max;
    fprintf(stderr, "[spec] SSD: fan_out_budget=%d r=%.2f sampling_C=%.2f -> %s\n",
            a.ssd_fan_out, a.ssd_r, a.ssd_C,
            ssd_enabled ? "ON (cache built per round)" : "OFF (budget <= K, no alternatives)");
    if (a.ssd_C < 1.0f) {
        fprintf(stderr, "[spec] WARN: --ssd-sampling-C<1 has no effect under greedy decoding (current mode is greedy).\n");
    }

    // ---- prefill: decode the prompt on the target, capture tap ----
    llama_batch batch_t = llama_batch_init(a.n_batch, 0, 1);

    auto push_token_batch = [&](llama_token id, llama_pos pos, bool want_logits) {
        const int n = batch_t.n_tokens;
        batch_t.token[n]     = id;
        batch_t.pos[n]       = pos;
        batch_t.n_seq_id[n]  = 1;
        batch_t.seq_id[n][0] = 0;
        batch_t.logits[n]    = want_logits ? 1 : 0;
        batch_t.n_tokens     = n + 1;
    };

    batch_t.n_tokens = 0;
    for (size_t i = 0; i < input.size(); ++i) {
        push_token_batch(input[i], (llama_pos) i, /*want_logits=*/ i + 1 == input.size());
    }
    if (llama_decode(ctx_t, batch_t) != 0) { fprintf(stderr, "prefill decode failed\n"); return 1; }

    // grab the full tap (entire prompt) into the accumulator
    std::vector<float> acc_kf, acc_vf, acc_ks, acc_vs;
    int64_t acc_nkv = 0;
    int64_t nhkv = 0, hd_full = 0, hd_swa = 0;
    {
        const float * kf=nullptr, *vf=nullptr, *ks=nullptr, *vs=nullptr, *hid=nullptr;
        int64_t n_tap = llama_get_assistant_kv_tap(ctx_t, &kf, &vf, &ks, &vs, &hid, &nhkv, &hd_full, &hd_swa);
        if (n_tap != (int64_t) input.size() || !kf) { fprintf(stderr, "prefill tap missing\n"); return 1; }

        acc_kf.assign(kf, kf + n_tap * hd_full * nhkv);
        acc_vf.assign(vf, vf + n_tap * hd_full * nhkv);
        acc_ks.assign(ks, ks + n_tap * hd_swa  * nhkv);
        acc_vs.assign(vs, vs + n_tap * hd_swa  * nhkv);
        acc_nkv = n_tap;

        // pending: the last prompt token's hidden -> the very first draft input uses it.
        // (We need this saved before the next decode overwrites the tap buffers.)
        // pending_feat is just the last row of the hidden tap.
        // We also remember the last prompt token id so the first round drafts t_1 from
        // input concat[ embed(t_0=first_gen), pending_feat=last_prompt_hidden ].
        // Save now.
    }
    // last prompt token's hidden (last row of the tap's hidden buffer)
    std::vector<float> pending_feat(n_embd_back);
    {
        const float * hid = nullptr;
        int64_t n_tap = llama_get_assistant_kv_tap(ctx_t, nullptr, nullptr, nullptr, nullptr, &hid, nullptr, nullptr, nullptr);
        std::memcpy(pending_feat.data(), hid + (n_tap - 1) * n_embd_back, n_embd_back * sizeof(float));
    }

    // sample the first generated token from the target's last logits
    llama_token first_gen = greedy(llama_get_logits_ith(ctx_t, batch_t.n_tokens - 1), n_vocab);

    // echo the prompt then the first token
    for (auto id : input) { emit_token(vocab, id); }
    emit_token(vocab, first_gen);

    // running state: pending_tok is the last sampled-but-not-yet-verified token;
    // its K/V is NOT in acc yet (it goes into the next verify batch's first slot).
    llama_token pending_tok = first_gen;
    int n_emitted = 1;
    int total_drafted = 0, total_accepted = 0, n_rounds = 0;

    llama_batch batch_d = llama_batch_init(1, (int32_t)(2 * n_embd_back), 1); // embd batch for draft

    std::vector<float> concat(2 * n_embd_back);
    std::vector<float> cur_feat(n_embd_back);

    // ---- SSD state (Speculative Speculative Decoding) ----
    // spec_cache[k] holds up to (F_k - 1) "what if bonus at draft pos k = alt" entries,
    // each with a pre-speculated next draft token + its chained feature for round T+1.
    struct cache_entry {
        llama_token alt_bonus;
        llama_token next_tok;
        std::vector<float> next_feat;
    };
    std::vector<std::vector<cache_entry>> spec_cache;

    // Carry-over seed when a cache hit lets us skip round T+1's first draft decode.
    bool ssd_have_seed = false;
    llama_token ssd_seed_tok = 0;
    std::vector<float> ssd_seed_feat(n_embd_back);

    // Stats.
    int64_t ssd_hits = 0, ssd_misses = 0;
    int64_t ssd_extra_decodes = 0;
    int64_t ssd_steps_saved   = 0;
    int64_t verify_wall_us    = 0;   // total wall-time spent in target verify decodes
    int64_t post_pass_wall_us = 0;   // total wall-time spent in SSD post-pass alt decodes
    int64_t overlap_wall_us   = 0;   // total wall-time of the verify/post-pass parallel region

    bool eog = llama_vocab_is_eog(vocab, pending_tok);

    const int64_t t_gen_start = ggml_time_us();
    while (!eog && n_emitted < a.n_predict) {
        // -------- DRAFT: assistant proposes up to N tokens --------
        // Feed the accumulated K/V to the draft context (forces graph reserve).
        llama_set_assistant_shared_kv(ctx_d, nhkv,
                hd_full, acc_kf.data(), acc_vf.data(), acc_nkv,
                hd_swa,  acc_ks.data(), acc_vs.data(), acc_nkv);

        // EAGLE/MTP input: cat[ embed(t_N) , hidden_{N-1} ].
        // - t_N = the just-sampled token (pending_tok)
        // - hidden_{N-1} = the target hidden at the position that produced t_N = pending_feat
        // Position: candidate generator uses input_ids.shape[1]-1 == acc_nkv (since pending_tok
        //   adds one slot to seq_len), but for cross-attention with constant position the assistant
        //   is robust to this choice; we use acc_nkv to match the reference.
        llama_token cur_tok = pending_tok;
        std::memcpy(cur_feat.data(), pending_feat.data(), n_embd_back * sizeof(float));
        const llama_pos draft_pos = (llama_pos) acc_nkv;

        std::vector<llama_token> drafted;
        drafted.reserve(a.n_draft_max);

        // Reset per-round speculation cache.
        spec_cache.assign(a.n_draft_max, {});

        // Estimate running primary acceptance rate for fan-out shaping (Thm 12).
        const float a_p_est = (n_rounds > 0 && total_drafted > 0)
            ? (float) total_accepted / (float) total_drafted
            : 0.8f;
        const std::vector<int> fanout = geometric_fanout(a.ssd_fan_out, a.n_draft_max, a_p_est, a.ssd_r);

        // SSD cache hit from previous round? Skip first draft decode and seed from cache.
        int start_k = 0;
        if (ssd_have_seed) {
            drafted.push_back(ssd_seed_tok);
            cur_tok = ssd_seed_tok;
            std::memcpy(cur_feat.data(), ssd_seed_feat.data(), n_embd_back * sizeof(float));
            ssd_steps_saved++;
            start_k = 1;
            ssd_have_seed = false;
        }

        // Scratch for top-F lookup.
        std::vector<llama_token> topF;

        auto build_concat_for = [&](llama_token tok, const std::vector<float> & feat) {
            llama_model_get_token_embd(tgt, tok, concat.data());
            for (int j = 0; j < n_embd_back; ++j) { concat[j] *= embed_scale; }
            std::memcpy(concat.data() + n_embd_back, feat.data(), n_embd_back * sizeof(float));
        };

        auto decode_draft_one = [&]() -> bool {
            batch_d.n_tokens     = 1;
            std::memcpy(batch_d.embd, concat.data(), concat.size() * sizeof(float));
            batch_d.pos[0]       = draft_pos;
            batch_d.n_seq_id[0]  = 1;
            batch_d.seq_id[0][0] = 0;
            batch_d.logits[0]    = 1;
            return llama_decode(ctx_d, batch_d) == 0;
        };

        // Per-step state we need to replay alt decodes AFTER the greedy chain completes.
        // (Interleaving alt decodes with greedy steps pollutes draft K/V at draft_pos and
        //  noticeably drops acceptance.)
        struct step_state { llama_token in_tok; std::vector<float> in_feat; std::vector<llama_token> alts; };
        std::vector<step_state> steps;
        steps.reserve(a.n_draft_max);

        for (int k = start_k; k < a.n_draft_max; ++k) {
            // Snapshot the INPUT for this step before greedy advances state.
            step_state st;
            st.in_tok = cur_tok;
            st.in_feat.assign(cur_feat.begin(), cur_feat.end());

            // ---- main greedy step ----
            build_concat_for(cur_tok, cur_feat);
            if (!decode_draft_one()) { fprintf(stderr, "draft decode failed\n"); goto done; }

            const float * lg  = llama_get_logits_ith    (ctx_d, 0);
            const float * emb = llama_get_embeddings_ith(ctx_d, 0); // [backbone + n_centroids]
            if (!lg || !emb) { break; }

            const int F_k = fanout[k];
            llama_token id;
            if (F_k > 1) {
                masked_topf(lg, n_vocab, emb + n_embd_back, n_centroids, top_k, token_to_cluster, F_k, topF);
                if (topF.empty()) { break; }
                id = topF[0];
                // Record the non-greedy alternatives for the post-pass.
                for (size_t i = 1; i < topF.size(); ++i) {
                    if (topF[i] != id) { st.alts.push_back(topF[i]); }
                }
            } else {
                id = masked_argmax(lg, n_vocab, emb + n_embd_back, n_centroids, top_k, token_to_cluster);
                if (id < 0) { break; }
            }
            drafted.push_back(id);

            // chain main loop: next step's hidden = this step's post-projection feature
            std::memcpy(cur_feat.data(), emb, n_embd_back * sizeof(float));
            cur_tok = id;
            steps.push_back(std::move(st));
        }
        total_drafted += (int) drafted.size();
        n_rounds++;

        // -------- Build VERIFY batch first so post-pass/verify can run in either order/parallel.
        batch_t.n_tokens = 0;
        push_token_batch(pending_tok, (llama_pos) acc_nkv, /*logits=*/true);
        for (size_t i = 0; i < drafted.size(); ++i) {
            push_token_batch(drafted[i], (llama_pos) (acc_nkv + 1 + (llama_pos) i), /*logits=*/true);
        }

        // ---- SSD post-pass body (lambda). Mutates ctx_d, batch_d, spec_cache, concat,
        //      ssd_extra_decodes. Does NOT touch ctx_t / batch_t. Safe to run concurrently
        //      with target verify (different context, different backend).
        auto run_post_pass = [&]() -> int64_t {
            const int64_t t0 = ggml_time_us();
            bool need_restore = false;
            for (size_t kk = 0; kk < steps.size(); ++kk) {
                const auto & st = steps[kk];
                for (llama_token alt : st.alts) {
                    build_concat_for(alt, st.in_feat);
                    if (!decode_draft_one()) { goto post_pass_done; }
                    const float * alt_lg  = llama_get_logits_ith    (ctx_d, 0);
                    const float * alt_emb = llama_get_embeddings_ith(ctx_d, 0);
                    if (!alt_lg || !alt_emb) { goto post_pass_done; }
                    llama_token alt_next = masked_argmax(alt_lg, n_vocab, alt_emb + n_embd_back,
                                                          n_centroids, top_k, token_to_cluster);
                    if (alt_next < 0) { goto post_pass_done; }
                    const int gk = start_k + (int) kk;
                    cache_entry e;
                    e.alt_bonus = alt;
                    e.next_tok  = alt_next;
                    e.next_feat.assign(alt_emb, alt_emb + n_embd_back);
                    spec_cache[gk].push_back(std::move(e));
                    ssd_extra_decodes++;
                    need_restore = true;
                }
            }
            // Restore K/V at draft_pos to match vanilla path (greedy's last step).
            if (need_restore && !steps.empty()) {
                const auto & last = steps.back();
                build_concat_for(last.in_tok, last.in_feat);
                if (decode_draft_one()) { ssd_extra_decodes++; }
            }
        post_pass_done:
            return ggml_time_us() - t0;
        };

        bool need_post_pass = false;
        for (const auto & s_ : steps) { if (!s_.alts.empty()) { need_post_pass = true; break; } }

        // ---- Dispatch: async overlap vs serial ----
        const int64_t t_dispatch = ggml_time_us();
        int verify_rc = 0;
        if (a.ssd_async && need_post_pass) {
            // Verify on Metal, post-pass on draft backend (CPU) — true parallel.
            std::future<int> verify_fut = std::async(std::launch::async, [&]() {
                const int64_t t0 = ggml_time_us();
                int rc = llama_decode(ctx_t, batch_t);
                verify_wall_us += (ggml_time_us() - t0);
                return rc;
            });
            post_pass_wall_us += run_post_pass();
            verify_rc = verify_fut.get();
        } else {
            // Serial path: post-pass then verify (preserves prior behavior).
            if (need_post_pass) { post_pass_wall_us += run_post_pass(); }
            const int64_t t0 = ggml_time_us();
            verify_rc = llama_decode(ctx_t, batch_t);
            verify_wall_us += (ggml_time_us() - t0);
        }
        overlap_wall_us += (ggml_time_us() - t_dispatch);
        if (verify_rc != 0) { fprintf(stderr, "verify decode failed\n"); break; }

        // grab the tap for this batch (will be appended fresh — context.cpp resets tap on
        // each decode that starts with n_tokens_prev==0, which is our case here)
        const float * kf=nullptr,*vf=nullptr,*ks=nullptr,*vs=nullptr,*hid=nullptr;
        int64_t n_tap = llama_get_assistant_kv_tap(ctx_t, &kf, &vf, &ks, &vs, &hid, nullptr, nullptr, nullptr);
        if (n_tap != (int64_t) batch_t.n_tokens || !kf) { fprintf(stderr, "verify tap mismatch (tap=%lld vs %d)\n", (long long) n_tap, batch_t.n_tokens); break; }

        // find how many drafts the target accepts.
        // target's argmax at position p (0-based within this batch) predicts the token at p+1.
        // - argmax at pos 0 should equal drafted[0]   to accept it
        // - argmax at pos 1 should equal drafted[1]   ...
        // - bonus is the target's argmax at the first mismatch position (or after all match).
        int n_accept = 0;
        llama_token next_pending = -1;
        for (int p = 0; p < batch_t.n_tokens; ++p) {
            const float * lg = llama_get_logits_ith(ctx_t, p);
            llama_token tgt_tok = greedy(lg, n_vocab);
            if (p < (int) drafted.size() && tgt_tok == drafted[p]) {
                n_accept++;
            } else {
                next_pending = tgt_tok;
                break;
            }
        }
        // commit n_accept drafted tokens + the bonus (next_pending).
        // verified positions 0..n_accept inputs are accepted; their K/V (positions acc_nkv..acc_nkv+n_accept)
        // is appended to acc. Drop the rest.
        const int64_t n_keep = (int64_t) n_accept + 1;

        // roll back the target's KV cache: keep positions [acc_nkv .. acc_nkv+n_keep), drop the rest
        // of this verify batch (positions [acc_nkv+n_keep .. acc_nkv+batch_t.n_tokens)).
        if ((int) batch_t.n_tokens > n_keep) {
            llama_memory_seq_rm(llama_get_memory(ctx_t), /*seq_id=*/0,
                    (llama_pos) (acc_nkv + n_keep), -1);
        }

        // append accepted positions' K/V
        for (int64_t p = 0; p < n_keep; ++p) {
            const float * kfp = kf  + p * hd_full * nhkv;
            const float * vfp = vf  + p * hd_full * nhkv;
            const float * ksp = ks  + p * hd_swa  * nhkv;
            const float * vsp = vs  + p * hd_swa  * nhkv;
            acc_kf.insert(acc_kf.end(), kfp, kfp + hd_full * nhkv);
            acc_vf.insert(acc_vf.end(), vfp, vfp + hd_full * nhkv);
            acc_ks.insert(acc_ks.end(), ksp, ksp + hd_swa  * nhkv);
            acc_vs.insert(acc_vs.end(), vsp, vsp + hd_swa  * nhkv);
        }
        acc_nkv += n_keep;

        // emit accepted draft tokens then the bonus
        for (int i = 0; i < n_accept && n_emitted < a.n_predict; ++i) {
            if (llama_vocab_is_eog(vocab, drafted[i])) { eog = true; break; }
            emit_token(vocab, drafted[i]);
            n_emitted++;
        }
        total_accepted += n_accept;

        if (eog || n_emitted >= a.n_predict) { break; }
        if (next_pending < 0) {
            // all drafts matched and we exhausted the batch; pick bonus from the last logits
            next_pending = greedy(llama_get_logits_ith(ctx_t, batch_t.n_tokens - 1), n_vocab);
        }
        if (llama_vocab_is_eog(vocab, next_pending)) { eog = true; break; }
        emit_token(vocab, next_pending);
        n_emitted++;

        // pending_feat for next round = the post-norm hidden at the last accepted position
        // (this is what predicts next_pending). pending_tok = next_pending.
        std::memcpy(pending_feat.data(), hid + (n_keep - 1) * n_embd_back, n_embd_back * sizeof(float));
        pending_tok = next_pending;

        // ---- SSD cache lookup ----
        // If the bonus token matches one of the speculation-cache alternatives at the
        // accept position, seed round T+1 with the pre-speculated next draft token + feature.
        // This skips the first draft decode of the next round (saves ~1 step on hit).
        if (a.ssd_fan_out > a.n_draft_max && n_accept < (int) spec_cache.size()) {
            bool hit = false;
            for (const auto & e : spec_cache[n_accept]) {
                if (e.alt_bonus == next_pending) {
                    ssd_seed_tok = e.next_tok;
                    std::memcpy(ssd_seed_feat.data(), e.next_feat.data(), n_embd_back * sizeof(float));
                    ssd_have_seed = true;
                    ssd_hits++;
                    hit = true;
                    break;
                }
            }
            if (!hit) { ssd_misses++; }
        }
    }

done:
    const int64_t t_gen_end = ggml_time_us();
    const double dt = (t_gen_end - t_gen_start) / 1e6;
    printf("\n");
    fprintf(stderr, "\n[spec] emitted=%d in %.2fs = %.1f t/s | rounds=%d drafted=%d accepted=%d accept_rate=%.1f%% acc_per_round=%.2f\n",
            n_emitted, dt, dt > 0 ? n_emitted / dt : 0.0,
            n_rounds, total_drafted, total_accepted,
            total_drafted ? 100.0 * (double) total_accepted / (double) total_drafted : 0.0,
            n_rounds ? (double) total_accepted / (double) n_rounds : 0.0);

    {
        const int64_t ssd_attempts = ssd_hits + ssd_misses;
        fprintf(stderr, "[spec] SSD: hits=%lld misses=%lld hit_rate=%.1f%% extra_draft_decodes=%lld steps_saved=%lld net_extra=%lld\n",
                (long long) ssd_hits, (long long) ssd_misses,
                ssd_attempts ? 100.0 * (double) ssd_hits / (double) ssd_attempts : 0.0,
                (long long) ssd_extra_decodes, (long long) ssd_steps_saved,
                (long long) (ssd_extra_decodes - ssd_steps_saved));
        const double v_ms = verify_wall_us    / 1000.0;
        const double p_ms = post_pass_wall_us / 1000.0;
        const double o_ms = overlap_wall_us   / 1000.0;
        const double hidden_pct = (v_ms + p_ms > 0)
            ? 100.0 * ((v_ms + p_ms) - o_ms) / (v_ms + p_ms) : 0.0;
        fprintf(stderr, "[spec] timing: verify=%.0fms post_pass=%.0fms overlap_region=%.0fms hidden=%.1f%%   async=%s draft_backend=%s\n",
                v_ms, p_ms, o_ms, hidden_pct,
                a.ssd_async ? "on" : "off",
                a.n_gpu_layers_draft > 0 ? "GPU" : "CPU");
    }

    llama_batch_free(batch_t);
    llama_batch_free(batch_d);
    llama_free(ctx_t);
    llama_free(ctx_d);
    llama_model_free(tgt);
    llama_model_free(drf);
    return 0;
}
