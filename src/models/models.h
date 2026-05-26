#pragma once

#include "llama-model.h"
#include "llama-graph.h"
#include "llama-model-loader.h"

#include <cmath>

//
// models
//

struct llama_model_gemma4 : public llama_model_base {
    llama_model_gemma4(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    struct graph : public llm_graph_context {
        const llama_model & model;

        const int64_t n_embd_per_layer;

        graph(const llama_model & model, const llm_graph_params & params);

        // TODO: refactor in common "per-layer" functionality [TAG_PER_LAYER]
        ggml_tensor * build_inp_per_layer();
        ggml_tensor * project_per_layer_inputs(ggml_tensor * inp_batch, ggml_tensor * inp_per_layer);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};

// Gemma 4 Assistant: a small MTP/EAGLE-style draft head for Gemma 4.
// It has no K/V of its own — every layer cross-attends to the target model's
// shared K/V (fed in via the assistant_kv carrier).
struct llama_model_gemma4_assistant : public llama_model_base {
    llama_model_gemma4_assistant(const struct llama_model_params & params) : llama_model_base(params) {}
    void load_arch_hparams(llama_model_loader & ml) override;
    void load_arch_tensors(llama_model_loader & ml) override;

    // global (non-per-layer) MTP tensors
    ggml_tensor * mtp_pre_projection  = nullptr; // Linear(2*n_embd_backbone -> n_embd)
    ggml_tensor * mtp_post_projection = nullptr; // Linear(n_embd -> n_embd_backbone), produces the chained feature
    ggml_tensor * mtp_centroids       = nullptr; // Linear(n_embd -> n_centroids), VQ codebook classifier
    ggml_tensor * mtp_token_ordering  = nullptr; // I32 [n_vocab], cluster -> canonical token-id map

    struct graph : public llm_graph_context {
        const llama_model_gemma4_assistant & model;

        graph(const llama_model_gemma4_assistant & model, const llm_graph_params & params);
    };

    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params & params) const override;
};
