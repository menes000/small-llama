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
#include <numeric>
#include <string>
#include <vector>

struct args {
    std::string model_target;
    std::string model_draft;
    std::string prompt = "hello";
    int  n_predict     = 128;
    int  n_gpu_layers  = 99;
    int  n_draft_max   = 5;
    int  n_ctx         = 4096;
    int  n_batch       = 2048;
    bool apply_chat    = true;
};

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "\nusage:\n  %s -m <target.gguf> -md <draft.gguf> [-p <prompt>] [-n N] "
        "[--draft-max N] [-ngl N] [-c N] [-b N] [--no-chat]\n\n", argv0);
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
        else if (s == "-c")   { auto v = need("-c");   if (!v) return false; a.n_ctx   = atoi(v); }
        else if (s == "-b")   { auto v = need("-b");   if (!v) return false; a.n_batch = atoi(v); }
        else if (s == "--no-chat") { a.apply_chat = false; }
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

// Pick the masked argmax: top-k clusters of centroid_logits → restrict dense logits
// to the tokens of those clusters → argmax.
// Mathematically equivalent to the reference Gemma4AssistantMaskedEmbedder head,
// since per-token logit = hidden . embedding in both.
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

// Greedy argmax over a dense logit vector.
static llama_token greedy(const float * logits, int n_vocab) {
    llama_token best = 0;
    float bv = logits[0];
    for (int v = 1; v < n_vocab; ++v) {
        if (logits[v] > bv) { bv = logits[v]; best = v; }
    }
    return best;
}

int main(int argc, char ** argv) {
    args a;
    if (!parse_args(argc, argv, a)) { return 1; }

    ggml_backend_load_all();

    // ---- load models ----
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = a.n_gpu_layers;

    llama_model * tgt = llama_model_load_from_file(a.model_target.c_str(), mp);
    if (!tgt) { fprintf(stderr, "failed to load target model: %s\n", a.model_target.c_str()); return 1; }
    llama_model * drf = llama_model_load_from_file(a.model_draft.c_str(),  mp);
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

        for (int i = 0; i < a.n_draft_max; ++i) {
            // build the concat row: scaled embedding + hidden
            llama_model_get_token_embd(tgt, cur_tok, concat.data());
            for (int j = 0; j < n_embd_back; ++j) { concat[j] *= embed_scale; }
            std::memcpy(concat.data() + n_embd_back, cur_feat.data(), n_embd_back * sizeof(float));

            batch_d.n_tokens     = 1;
            std::memcpy(batch_d.embd, concat.data(), concat.size() * sizeof(float));
            batch_d.pos[0]       = draft_pos;
            batch_d.n_seq_id[0]  = 1;
            batch_d.seq_id[0][0] = 0;
            batch_d.logits[0]    = 1;

            if (llama_decode(ctx_d, batch_d) != 0) { fprintf(stderr, "draft decode failed\n"); goto done; }

            const float * lg  = llama_get_logits_ith    (ctx_d, 0);
            const float * emb = llama_get_embeddings_ith(ctx_d, 0); // [backbone + n_centroids]
            if (!lg || !emb) { break; }

            // masked-embedding head: argmax dense logits restricted to top-k centroid clusters
            llama_token id = masked_argmax(lg, n_vocab, emb + n_embd_back, n_centroids, top_k, token_to_cluster);
            if (id < 0) { break; }

            drafted.push_back(id);

            // chain: next step's hidden = this step's post-projection feature
            std::memcpy(cur_feat.data(), emb, n_embd_back * sizeof(float));
            cur_tok = id;
        }
        total_drafted += (int) drafted.size();
        n_rounds++;

        // -------- VERIFY: target decodes [pending_tok, d_1, ..., d_N] --------
        // build the target batch starting at position acc_nkv
        batch_t.n_tokens = 0;
        push_token_batch(pending_tok, (llama_pos) acc_nkv, /*logits=*/true);
        for (size_t i = 0; i < drafted.size(); ++i) {
            push_token_batch(drafted[i], (llama_pos) (acc_nkv + 1 + (llama_pos) i), /*logits=*/true);
        }
        if (llama_decode(ctx_t, batch_t) != 0) { fprintf(stderr, "verify decode failed\n"); break; }

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

    llama_batch_free(batch_t);
    llama_batch_free(batch_d);
    llama_free(ctx_t);
    llama_free(ctx_d);
    llama_model_free(tgt);
    llama_model_free(drf);
    return 0;
}
