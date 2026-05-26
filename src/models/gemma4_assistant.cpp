#include "models.h"

// Gemma 4 Assistant — MTP/EAGLE-style draft head for Gemma 4.
//
// This file implements model loading (hparams + tensors). The decode graph
// (cross-attention against the target model's shared K/V + the centroid logit
// head) and the speculative-loop wiring are tracked separately as Phase C/D.
// See plan: continue-sleepy-kite.md

void llama_model_gemma4_assistant::load_arch_hparams(llama_model_loader & ml) {
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);

    hparams.f_attention_scale = 1.0f; // Gemma 4 uses scaling = 1.0 (no pre-attn scaling)

    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,          hparams.rope_freq_base_train_swa, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // MTP / assistant specific
    ml.get_key(LLM_KV_N_EMBD_BACKBONE,        hparams.n_embd_backbone);
    ml.get_key(LLM_KV_N_CENTROIDS,            hparams.n_centroids,            false);
    ml.get_key(LLM_KV_CENTROID_TOP_K,         hparams.centroid_top_k,         false);
    ml.get_key(LLM_KV_USE_ORDERED_EMBEDDINGS, hparams.use_ordered_embeddings, false);

    // the model "embedding" output carries the post_projection feature (backbone width) that the
    // speculative loop chains, plus the centroid logits (when the ordered/masked-embedding head is
    // used) appended after it so the loop can restrict the dense logits to the top-k clusters.
    hparams.n_embd_out_impl = hparams.n_embd_backbone +
        (hparams.use_ordered_embeddings ? hparams.n_centroids : 0);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_gemma4_assistant::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_backbone = hparams.n_embd_backbone;
    const int64_t n_centroids     = hparams.n_centroids;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // lm_head is tied to token_embd (no separate output tensor in the gguf)
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    // global MTP head tensors
    mtp_pre_projection  = create_tensor(tn(LLM_TENSOR_MTP_PRE_PROJECTION,  "weight"), {2 * n_embd_backbone, n_embd}, 0);
    mtp_post_projection = create_tensor(tn(LLM_TENSOR_MTP_POST_PROJECTION, "weight"), {n_embd, n_embd_backbone}, 0);
    if (hparams.use_ordered_embeddings) {
        mtp_centroids      = create_tensor(tn(LLM_TENSOR_MTP_CENTROIDS,      "weight"), {n_embd, n_centroids}, 0);
        mtp_token_ordering = create_tensor(tn(LLM_TENSOR_MTP_TOKEN_ORDERING, "weight"), {n_vocab}, 0);
    }
    fprintf(stderr, "G4A load: use_ordered_embeddings=%d n_centroids=%d n_vocab=%d mtp_token_ordering=%p name='%s'\n",
            (int) hparams.use_ordered_embeddings, (int) n_centroids, (int) n_vocab,
            (void*) mtp_token_ordering, mtp_token_ordering ? mtp_token_ordering->name : "(null)");

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const int64_t n_head_i      = hparams.n_head(i);
        const int64_t n_embd_head_i = hparams.n_embd_head_k(i); // 512 (full) / 256 (swa)

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);

        // Q-only attention: K/V come from the target model (shared), so no wk/wv here.
        layer.wq             = create_tensor(tn(LLM_TENSOR_ATTN_Q,         "weight", i), {n_embd, n_embd_head_i * n_head_i}, 0);
        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), {n_embd_head_i}, 0);
        layer.wo             = create_tensor(tn(LLM_TENSOR_ATTN_OUT,       "weight", i), {n_embd_head_i * n_head_i, n_embd}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        // full-attention layers use rope_freqs for proportional rope (name template ignores %d -> single global tensor)
        if (!hparams.is_swa(i)) {
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_embd_head_i / 2}, 0);
        }

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_gate      = create_tensor(tn(LLM_TENSOR_FFN_GATE,      "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_up        = create_tensor(tn(LLM_TENSOR_FFN_UP,        "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down      = create_tensor(tn(LLM_TENSOR_FFN_DOWN,      "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.out_scale     = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), {1u}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_gemma4_assistant::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_gemma4_assistant::graph::graph(const llama_model_gemma4_assistant & model, const llm_graph_params & params) :
        llm_graph_context(params),
        model(model) {
    const int64_t n_embd_backbone = hparams.n_embd_backbone;

    // input: concat[target_token_embed, target_hidden] of width 2*n_embd_backbone
    // (fed via batch.embd; n_embd_inp() reports 2*n_embd_backbone for this arch)
    auto inp_embd = std::make_unique<llm_graph_input_embd>(2 * n_embd_backbone);
    // tokens tensor is unused (input is always embd), but allocate it so set_input never
    // dereferences a null pointer if the batch happens to also carry token ids
    inp_embd->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp_embd->tokens);
    inp_embd->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 2 * n_embd_backbone, n_tokens);
    ggml_set_input(inp_embd->embd);
    ggml_tensor * concat = inp_embd->embd;
    res->add_input(std::move(inp_embd));

    // pre_projection: 2*backbone -> n_embd
    ggml_tensor * inpL = build_lora_mm(model.mtp_pre_projection, concat);
    cb(inpL, "pre_proj", -1);

    // NOTE: Gemma4TextModel only applies the sqrt(hidden) embedding scale when it computes
    // embeddings from input_ids. The assistant feeds inputs_embeds DIRECTLY (pre_projection
    // output), so embed_tokens is bypassed and NO scaling is applied. (Verified against the
    // transformers reference.)
    cb(inpL, "pre_proj_out", -1);

    std::vector<ggml_tensor *> dbg_probe; // [input, L0, L1, L2, L3]
    dbg_probe.push_back(inpL);

    ggml_tensor * inp_pos = build_inp_pos();

    // shared K/V from the target model (cross-attention source for every layer)
    auto * inp_kv = build_inp_assistant_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const bool    is_swa      = hparams.is_swa(il);
        const int64_t n_head      = hparams.n_head(il);
        const int64_t n_embd_head = hparams.n_embd_head_k(il);

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);
        const int   n_rot_l      = hparams.n_rot(il);

        ggml_tensor * cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // Q only (no K/V projections — K/V come from the target)
        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);

        ggml_tensor * freq_factors = is_swa ? nullptr : model.layers[il].rope_freqs;
        if (!getenv("G4A_NOROPE")) {
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig,
                                 freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
        }
        cb(Qcur, "Qcur", il);

        // cross-attention against the target's shared K/V (full or sliding by layer type)
        ggml_tensor * K    = is_swa ? inp_kv->k_swa           : inp_kv->k_full;
        ggml_tensor * V    = is_swa ? inp_kv->v_swa           : inp_kv->v_full;
        ggml_tensor * mask = is_swa ? inp_kv->kq_mask_swa_cnv : inp_kv->kq_mask_full_cnv;

        float ascale = hparams.f_attention_scale;
        if (getenv("G4A_ASCALE")) { ascale = 1.0f / sqrtf((float) n_embd_head); }
        cur = build_attn_mha(Qcur, K, V, nullptr, mask, nullptr, nullptr, ascale, il);
        cb(cur, "kqv_out", il);

        if (getenv("G4A_NOATTN")) { cur = ggml_scale(ctx0, cur, 0.0f); }

        cur = build_lora_mm(model.layers[il].wo, cur);

        if (il == n_layer - 1 && inp_out_ids) {
            cur  = ggml_get_rows(ctx0, cur,  inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }

        cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        // feed-forward
        cur = build_norm(attn_out, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cur = build_ffn(cur,
                model.layers[il].ffn_up,   nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr, LLM_FFN_GELU, LLM_FFN_PAR, il);
        cur = build_norm(cur, model.layers[il].ffn_post_norm, nullptr, LLM_NORM_RMS, -1);
        cb(cur, "ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, attn_out);

        // per-layer output scale
        if (model.layers[il].out_scale) {
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
            cb(cur, "out_scaled", il);
        }

        inpL = cur;
        dbg_probe.push_back(inpL);
    }

    // debug: route an intermediate (G4A_PROBE = -1 scaled-input, 0..3 layer output) through
    // the output_norm + post_projection so it can be compared against the reference per-layer.
    if (const char * pe = getenv("G4A_PROBE")) {
        int k = atoi(pe) + 1; // -1 -> index 0 (scaled input)
        fprintf(stderr, "G4A PROBE routing: k=%d dbg_probe.size=%d n_tokens=%d\n", k, (int) dbg_probe.size(), (int) n_tokens);
        if (k >= 0 && k < (int) dbg_probe.size()) { inpL = dbg_probe[k]; }
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    // chained feature: post_projection back to backbone width (consumed by the spec loop)
    ggml_tensor * feature = build_lora_mm(model.mtp_post_projection, cur);
    cb(feature, "result_feature", -1);

    // masked-embedding (centroid) logit head: the reference restricts the dense lm_head to the
    // tokens of the top-k centroid clusters. Since the per-token logit is the same dot product
    // (hidden . embedding) in both, we expose the centroid logits here and let the spec loop mask
    // the dense logits to the top-k clusters host-side. The centroid logits are appended to the
    // embeddings output (feature | centroid_logits); see n_embd_out_impl in load_arch_hparams.
    if (model.mtp_centroids) {
        ggml_tensor * centroid_logits = build_lora_mm(model.mtp_centroids, cur); // [n_centroids, T]
        cb(centroid_logits, "result_centroids", -1);
        res->t_embd = ggml_concat(ctx0, feature, centroid_logits, 0);
    } else {
        res->t_embd = feature;
    }
    cb(res->t_embd, "result_embd", -1);

    // dense lm_head (tied to token_embd), in CANONICAL token space. The spec loop masks these to
    // the top-k centroid clusters (= the masked_embedding head) before argmax.
    ggml_tensor * logits = build_lora_mm(model.output, cur);
    cb(logits, "result_output", -1);
    res->t_logits = logits;

    ggml_build_forward_expand(gf, res->t_embd);
    ggml_build_forward_expand(gf, logits);
}
