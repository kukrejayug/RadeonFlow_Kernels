#include "../checker/checker.h"
#include <ATen/ops/dot.h>
#include <ATen/ops/silu.h>
#include <dlfcn.h>
#include <c10/core/ScalarType.h>
#include <cstdio>
#include <iostream>
#include <vector>
#include "../../src/moe/moe.h"
#include <hip/hip_runtime.h>
#include <ATen/hip/HIPContext.h>

int input_params_count = 5;

InputParams input_params[] = {{.batch_size = 1,
                               .seq_len = 512,
                               .d_hidden = 7168,
                               .d_expert = 2048,
                               .n_routed_experts = 4,
                               .n_experts_per_token = 4,
                               .n_shared_experts = 1,
                               .seed = 9371},
                              {.batch_size = 1,
                               .seq_len = 512,
                               .d_hidden = 7168,
                               .d_expert = 2048,
                               .n_routed_experts = 8,
                               .n_experts_per_token = 4,
                               .n_shared_experts = 1,
                               .seed = 2291},
                              {.batch_size = 2,
                               .seq_len = 512,
                               .d_hidden = 7168,
                               .d_expert = 2048,
                               .n_routed_experts = 8,
                               .n_experts_per_token = 4,
                               .n_shared_experts = 1,
                               .seed = 2291},
                              {.batch_size = 1,
                               .seq_len = 8192,
                               .d_hidden = 7168,
                               .d_expert = 2048,
                               .n_routed_experts = 8,
                               .n_experts_per_token = 4,
                               .n_shared_experts = 1,
                               .seed = 81934},
                              {.batch_size = 1,
                               .seq_len = 2048,
                               .d_hidden = 7168,
                               .d_expert = 2048,
                               .n_routed_experts = 32,
                               .n_experts_per_token = 4,
                               .n_shared_experts = 1,
                               .seed = 9371},
                              {.batch_size = 1,
                               .seq_len = 8192,
                               .d_hidden = 7168,
                               .d_expert = 2048,
                               .n_routed_experts = 32,
                               .n_experts_per_token = 4,
                               .n_shared_experts = 1,
                               .seed = 1212}};

struct MoEInput {
    InputParams params;
    std::vector<torch::Tensor> expert_weight_gate;
    std::vector<torch::Tensor> expert_weight_up;
    std::vector<torch::Tensor> expert_weight_down;
    torch::Tensor expert_weight_gate_transposed;
    torch::Tensor expert_weight_up_transposed;
    torch::Tensor expert_weight_down_transposed;

    torch::Tensor shared_expert_weight_gate;
    torch::Tensor shared_expert_weight_up;
    torch::Tensor shared_expert_weight_down;

    torch::Tensor shared_expert_weight_gate_transposed;
    torch::Tensor shared_expert_weight_up_transposed;
    torch::Tensor shared_expert_weight_down_transposed;

    torch::Tensor router_weight; // n_shared_experts x d_hidden

    torch::Tensor input_seq; // batch_size x seq_len x d_hidden

    // for testing
    torch::Tensor expert_scores_ref;
    torch::Tensor expert_scores;

    torch::Tensor seq_experts_index;
    torch::Tensor seq_experts_index_ref;

    torch::Tensor seq_experts_softmax;
    torch::Tensor seq_experts_softmax_ref;

    torch::Tensor source_rows;
    torch::Tensor source_rows_ref;

    torch::Tensor permuted_experts;
    torch::Tensor permuted_experts_ref;

    torch::Tensor permuted_rows; // (batch_size x seq_len) x k
    torch::Tensor permuted_rows_ref;

    torch::Tensor first_token_offset;
    torch::Tensor first_token_offset_ref;

    torch::Tensor expanded_seq_input; // (batch_size x seq_len) x k x d_hidden
    torch::Tensor expanded_seq_input_ref;

    torch::Tensor expanded_seq_softmax; // (batch_size x seq_len) x k (the softmax value)
    torch::Tensor expanded_seq_softmax_ref;

    torch::Tensor rev_permuted_rows; // (batch_size x seq_len) x k
    torch::Tensor rev_permuted_rows_ref;

    torch::Tensor expanded_fc1_gate_splitk;
    torch::Tensor expanded_fc1_up_splitk;
    torch::Tensor expanded_fc2_splitk;

    torch::Tensor expanded_fc1_output; // (batch_size x seq_len) x d_expert
    torch::Tensor expanded_fc1_output_ref;

    torch::Tensor expanded_fc2_output; // (batch_size x seq_len) x d_hidden
    torch::Tensor expanded_fc2_output_ref;

    torch::Tensor shared_fc1_gate_splitk;
    torch::Tensor shared_fc1_up_splitk;
    torch::Tensor shared_fc2_splitk;

    torch::Tensor shared_fc1_output; // (batch_size x seq_len) x d_expert
    torch::Tensor shared_fc1_output_ref;
    torch::Tensor shared_fc2_output; // (batch_size x seq_len) x d_hidden
    torch::Tensor shared_fc2_output_ref;

    torch::Tensor final_output;
    torch::Tensor final_output_ref;

    MoEInputRaw *to_input_raw() {
        MoEInputRaw *input_raw = new MoEInputRaw();
        input_raw->expert_weight_gate = new void *[params.n_routed_experts];
        input_raw->expert_weight_up = new void *[params.n_routed_experts];
        input_raw->expert_weight_down = new void *[params.n_routed_experts];
        input_raw->params = params;

        input_raw->expert_weight_gate_transposed = new void *[params.n_routed_experts];
        input_raw->expert_weight_up_transposed = new void *[params.n_routed_experts];
        input_raw->expert_weight_down_transposed = new void *[params.n_routed_experts];
        for (int i = 0; i < params.n_routed_experts; i++) {
            input_raw->expert_weight_gate[i] = expert_weight_gate[i].data_ptr();
            input_raw->expert_weight_up[i] = expert_weight_up[i].data_ptr();
            input_raw->expert_weight_down[i] = expert_weight_down[i].data_ptr();
        }

        input_raw->expert_weight_gate_transposed = expert_weight_gate_transposed.data_ptr();
        input_raw->expert_weight_up_transposed = expert_weight_up_transposed.data_ptr();
        input_raw->expert_weight_down_transposed = expert_weight_down_transposed.data_ptr();

        input_raw->shared_expert_weight_gate = shared_expert_weight_gate.data_ptr();
        input_raw->shared_expert_weight_up = shared_expert_weight_up.data_ptr();
        input_raw->shared_expert_weight_down = shared_expert_weight_down.data_ptr();

        input_raw->shared_expert_weight_gate_transposed = shared_expert_weight_gate_transposed.data_ptr();
        input_raw->shared_expert_weight_up_transposed = shared_expert_weight_up_transposed.data_ptr();
        input_raw->shared_expert_weight_down_transposed = shared_expert_weight_down_transposed.data_ptr();

        input_raw->router_weight = router_weight.data_ptr();
        input_raw->expert_scores = expert_scores.data_ptr();
        input_raw->seq_experts_index = seq_experts_index.data_ptr();
        input_raw->seq_experts_softmax = seq_experts_softmax.data_ptr();
        input_raw->source_rows = source_rows.data_ptr();
        input_raw->permuted_experts = permuted_experts.data_ptr();
        input_raw->permuted_rows = permuted_rows.data_ptr();
        input_raw->first_token_offset = first_token_offset.data_ptr();
        input_raw->input_seq = input_seq.data_ptr();

        input_raw->expanded_seq_input = expanded_seq_input.data_ptr();
        input_raw->expanded_seq_softmax = expanded_seq_softmax.data_ptr();
        input_raw->rev_permuted_rows = rev_permuted_rows.data_ptr();

        input_raw->expanded_fc1_output = expanded_fc1_output.data_ptr();
        input_raw->expanded_fc2_output = expanded_fc2_output.data_ptr();
        input_raw->expanded_fc1_gate_splitk = expanded_fc1_gate_splitk.data_ptr();
        input_raw->expanded_fc1_up_splitk = expanded_fc1_up_splitk.data_ptr();
        input_raw->expanded_fc2_splitk = expanded_fc2_splitk.data_ptr();

        input_raw->shared_fc1_gate_splitk = shared_fc1_gate_splitk.data_ptr();
        input_raw->shared_fc1_up_splitk = shared_fc1_up_splitk.data_ptr();
        input_raw->shared_fc2_splitk = shared_fc2_splitk.data_ptr();

        input_raw->shared_fc1_output = shared_fc1_output.data_ptr();
        input_raw->shared_fc2_output = shared_fc2_output.data_ptr();

        input_raw->final_output = final_output.data_ptr();
        return input_raw;
    }
};

void ref_kernel(MoEInput &input) {
    torch::manual_seed(input.params.seed);
    torch::NoGradGuard no_grad;
    int token_count = input.params.batch_size * input.params.seq_len;
    int expanded_token_count = token_count * input.params.n_experts_per_token;

    torch::Tensor input_seq_flat = input.input_seq.view({token_count, input.params.d_hidden});

    input.expert_scores_ref = torch::matmul(input_seq_flat, input.router_weight.transpose(0, 1));
    input.expert_scores_ref = torch::softmax(input.expert_scores_ref, 1);

    auto [seq_experts_softmax_ref, seq_experts_index_ref] =
        torch::topk(input.expert_scores_ref, input.params.n_experts_per_token, 1, true, true);

    input.seq_experts_softmax_ref = seq_experts_softmax_ref;
    input.seq_experts_index_ref = seq_experts_index_ref;

    // Fill source_rows_ref with values: column_index * token_count + row_index
    torch::Tensor row_indices = torch::arange(token_count, input.source_rows_ref.options())
                                    .view({token_count, 1})
                                    .expand({token_count, input.params.n_experts_per_token});
    torch::Tensor col_indices = torch::arange(input.params.n_experts_per_token, input.source_rows_ref.options())
                                    .view({1, input.params.n_experts_per_token})
                                    .expand({token_count, input.params.n_experts_per_token});
    input.source_rows_ref = col_indices * token_count + row_indices;

    // Sort pairs where seq_experts_index is the key and source_rows is the value
    torch::Tensor flattened_experts = input.seq_experts_index_ref.flatten();
    torch::Tensor flattened_rows = input.source_rows_ref.flatten();

    // Sort by expert indices
    auto [sorted_experts, sort_indices] = torch::sort(flattened_experts);

    // Use the sort indices to reorder the source rows
    torch::Tensor sorted_rows = torch::index_select(flattened_rows, 0, sort_indices);

    // Reshape to match the expected tensor shapes
    input.permuted_experts_ref = sorted_experts.view({1, -1});
    input.permuted_rows_ref = sorted_rows.view({1, -1});

    // Calculate rev_permuted_rows_ref (mapping from original to sorted positions)
    torch::Tensor sorted_positions = torch::arange(flattened_rows.size(0), input.rev_permuted_rows_ref.options());
    torch::Tensor flattened_permuted_rows = input.permuted_rows_ref.flatten();
    input.rev_permuted_rows_ref.zero_();
    input.rev_permuted_rows_ref.scatter_(0, flattened_permuted_rows.to(torch::kInt64), sorted_positions);

    // Calculate first_token_offset_ref - stores starting positions for each expert
    torch::Tensor permuted_experts_flat = input.permuted_experts_ref.flatten();

    // Initialize first element to 0 (start of the first expert)
    input.first_token_offset_ref[0][0] = 0;

    // Count tokens for each expert and compute cumulative offsets
    int curr_offset = 0;
    int prev_expert = -1;

    for (int i = 0; i < permuted_experts_flat.size(0); i++) {
        int curr_expert = permuted_experts_flat[i].item<int>();

        // When we encounter a new expert, fill all skipped experts and the current one
        if (curr_expert != prev_expert) {
            // Fill any skipped experts with the current offset
            for (int e = prev_expert + 1; e <= curr_expert; e++) {
                input.first_token_offset_ref[0][e] = curr_offset;
            }
            prev_expert = curr_expert;
        }
        curr_offset++;
    }

    // Fill the remaining experts and the final boundary
    for (int e = prev_expert + 1; e <= input.params.n_routed_experts; e++) {
        input.first_token_offset_ref[0][e] = curr_offset;
    }

    // Calculate expanded_seq_input_ref - vectorized approach
    torch::Tensor permuted_rows_flat = input.permuted_rows_ref.flatten();
    torch::Tensor source_rows = torch::remainder(permuted_rows_flat, token_count);
    input.expanded_seq_input_ref = torch::index_select(input_seq_flat, 0, source_rows);

    // Calculate expanded_seq_softmax_ref - vectorized approach
    torch::Tensor rows = torch::remainder(permuted_rows_flat, token_count);
    torch::Tensor cols = torch::div(permuted_rows_flat, token_count, "trunc");
    input.expanded_seq_softmax_ref = input.seq_experts_softmax_ref.index({rows, cols});

    // Calculate expanded_fc1_output_ref - process each expert's batch
    for (int expert_idx = 0; expert_idx < input.params.n_routed_experts; expert_idx++) {
        int start_idx = input.first_token_offset_ref[0][expert_idx].item<int>();
        int end_idx = input.first_token_offset_ref[0][expert_idx + 1].item<int>();

        if (end_idx > start_idx) {
            // Get the rows assigned to this expert
            torch::Tensor expert_inputs = input.expanded_seq_input_ref.slice(0, start_idx, end_idx);

            // Perform matrix multiplication with the expert's gate weights
            torch::Tensor expert_outputs_gate =
                torch::silu(torch::matmul(expert_inputs, input.expert_weight_gate[expert_idx]));
            torch::Tensor expert_outputs_up = torch::matmul(expert_inputs, input.expert_weight_up[expert_idx]);
            torch::Tensor expert_outputs = expert_outputs_gate.mul(expert_outputs_up);

            // Write results back to the output tensor
            input.expanded_fc1_output_ref.slice(0, start_idx, end_idx) = expert_outputs;
        }
    }

    // Calculate expanded_fc2_output_ref - process each expert's batch
    for (int expert_idx = 0; expert_idx < input.params.n_routed_experts; expert_idx++) {
        int start_idx = input.first_token_offset_ref[0][expert_idx].item<int>();
        int end_idx = input.first_token_offset_ref[0][expert_idx + 1].item<int>();

        if (end_idx > start_idx) {
            // Get the rows assigned to this expert
            torch::Tensor expert_inputs = input.expanded_fc1_output_ref.slice(0, start_idx, end_idx);

            // Perform matrix multiplication with the expert's gate weights
            torch::Tensor expert_outputs_down = torch::matmul(expert_inputs, input.expert_weight_down[expert_idx]);

            // We don't do scale here.
            // // Scale expert_outputs_down by the corresponding value from expanded_seq_softmax_ref
            // torch::Tensor scale = input.expanded_seq_softmax_ref.slice(0, start_idx, end_idx);
            // expert_outputs_down = expert_outputs_down * scale.unsqueeze(1);

            // Write results back to the output tensor
            input.expanded_fc2_output_ref.slice(0, start_idx, end_idx) = expert_outputs_down;
        }
    }

    torch::Tensor shared_fc2_gate = torch::silu(torch::matmul(input_seq_flat, input.shared_expert_weight_gate));
    torch::Tensor shared_fc2_up = torch::matmul(input_seq_flat, input.shared_expert_weight_up);
    torch::Tensor shared_fc2_right = torch::mul(shared_fc2_gate, shared_fc2_up);
    input.shared_fc2_output_ref = torch::matmul(shared_fc2_right, input.shared_expert_weight_down);

    // Calculate final_output_ref - aggregate the expert outputs
    input.final_output_ref.zero_();

    for (int k = 0; k < input.params.n_experts_per_token; k++) {
        // Calculate the base indices: k*token_count + i for all i
        torch::Tensor base_indices =
            torch::arange(token_count, input.rev_permuted_rows_ref.options()) + k * token_count;

        // Get the sorted indices using rev_permuted_rows_ref
        torch::Tensor sorted_indices = input.rev_permuted_rows_ref.index({base_indices});

        // Get the corresponding outputs and weights
        torch::Tensor expert_outputs = input.expanded_fc2_output_ref.index({sorted_indices});
        torch::Tensor expert_weights = input.expanded_seq_softmax_ref.index({sorted_indices}).unsqueeze(1);

        // Add the weighted outputs to final_output_ref
        input.final_output_ref += expert_outputs * expert_weights;
    }

    // Add the shared expert output (no need to multiply by weight as it's already scaled)
    input.final_output_ref += input.shared_fc2_output_ref;
}

void exact_ref_kernel(MoEInput &input) {
    torch::NoGradGuard no_grad;
    int token_count = input.params.batch_size * input.params.seq_len;

    // Flatten the input to [token_count, d_hidden]
    torch::Tensor x = input.input_seq.reshape({token_count, input.params.d_hidden});

    // Calculate router scores
    torch::Tensor router_logits = torch::matmul(x, input.router_weight.transpose(0, 1));
    torch::Tensor router_probs = torch::softmax(router_logits, 1);

    // Get top-k experts and their scores
    auto [topk_scores, topk_indices] = torch::topk(router_probs, input.params.n_experts_per_token, 1, true, true);

    // Initialize the output tensor with zeros
    torch::Tensor output = torch::zeros_like(x);

    // Process the shared expert
    torch::Tensor shared_gate = torch::silu(torch::matmul(x, input.shared_expert_weight_gate));
    torch::Tensor shared_up = torch::matmul(x, input.shared_expert_weight_up);
    torch::Tensor shared_output = torch::matmul(shared_gate * shared_up, input.shared_expert_weight_down);
    output += shared_output;

    // Process each token through its assigned experts
    for (int token_idx = 0; token_idx < token_count; token_idx++) {
        for (int k = 0; k < input.params.n_experts_per_token; k++) {
            int expert_id = topk_indices[token_idx][k].item<int>();
            float score = topk_scores[token_idx][k].item<float>();

            // Get token input
            torch::Tensor token_input = x[token_idx].unsqueeze(0);

            // Forward through the expert
            torch::Tensor gate_output = torch::silu(torch::matmul(token_input, input.expert_weight_gate[expert_id]));
            torch::Tensor up_output = torch::matmul(token_input, input.expert_weight_up[expert_id]);
            torch::Tensor expert_intermediate = gate_output * up_output;
            torch::Tensor expert_output = torch::matmul(expert_intermediate, input.expert_weight_down[expert_id]);

            // Scale by routing probability and add to output
            output[token_idx] += expert_output[0] * score;
        }
    }

    // Store the result in the output field
    input.final_output_ref.copy_(output);
}

typedef void (*run_t)(MoEInputRaw *, hipStream_t, PerfMetrics *);
run_t run_topk_func;

void case_initialize() {
    input_params_count = sizeof(input_params) / sizeof(input_params[0]);

    // Load the symbol
    void *handle = dlopen("libmoe.so", RTLD_NOW);
    if (!handle) {
        std::cerr << "Cannot open library: " << dlerror() << std::endl;
        abort();
    }

    run_topk_func = (run_t)dlsym(handle, "run_topk");
    if (!run_topk_func) {
        std::cerr << "Cannot load symbol 'run_topk': " << dlerror() << std::endl;
        dlclose(handle);
        abort();
    }
}

void generate_input(MoEInput &input) {
    torch::manual_seed(input.params.seed);
    auto cuda_options = torch::TensorOptions().device(torch::kCUDA);
    auto fp16_options = cuda_options.dtype(torch::kFloat16);
    auto fp32_options = cuda_options.dtype(torch::kFloat32);
    auto bf16_options = cuda_options.dtype(torch::kBFloat16);
    auto i16_options = cuda_options.dtype(torch::kInt16);
    auto i32_options = cuda_options.dtype(torch::kInt32);

    for (int i = 0; i < input.params.n_routed_experts; i++) {
        input.expert_weight_gate.push_back(torch::randn({input.params.d_hidden, input.params.d_expert}, fp16_options) /
                                           std::sqrt(static_cast<float>(input.params.d_expert)));
        input.expert_weight_up.push_back(torch::randn({input.params.d_hidden, input.params.d_expert}, fp16_options) /
                                         std::sqrt(static_cast<float>(input.params.d_expert)));
        input.expert_weight_down.push_back(torch::randn({input.params.d_expert, input.params.d_hidden}, fp16_options) /
                                           std::sqrt(static_cast<float>(input.params.d_hidden)));
    }

    input.expert_weight_gate_transposed =
        torch::zeros({input.params.n_routed_experts, input.params.d_expert, input.params.d_hidden}, fp16_options);
    input.expert_weight_up_transposed =
        torch::zeros({input.params.n_routed_experts, input.params.d_expert, input.params.d_hidden}, fp16_options);
    input.expert_weight_down_transposed =
        torch::zeros({input.params.n_routed_experts, input.params.d_hidden, input.params.d_expert}, fp16_options);

    input.shared_expert_weight_gate =
        torch::randn({input.params.d_hidden, input.params.d_expert}, fp16_options) /
        std::sqrt(static_cast<float>(input.params.d_expert * input.params.n_shared_experts));
    input.shared_expert_weight_up =
        torch::randn({input.params.d_hidden, input.params.d_expert}, fp16_options) /
        std::sqrt(static_cast<float>(input.params.d_expert * input.params.n_shared_experts));
    input.shared_expert_weight_down = torch::randn({input.params.d_expert, input.params.d_hidden}, fp16_options) /
                                      std::sqrt(static_cast<float>(input.params.d_hidden));

    input.shared_expert_weight_gate_transposed =
        torch::zeros({input.params.d_expert, input.params.d_hidden}, fp16_options);
    input.shared_expert_weight_up_transposed =
        torch::zeros({input.params.d_expert, input.params.d_hidden}, fp16_options);
    input.shared_expert_weight_down_transposed =
        torch::zeros({input.params.d_hidden, input.params.d_expert}, fp16_options);

    input.router_weight = torch::randn({input.params.n_routed_experts, input.params.d_hidden}, fp16_options) /
                          std::sqrt(static_cast<float>(input.params.d_hidden));

    input.input_seq =
        torch::randn({input.params.batch_size, input.params.seq_len, input.params.d_hidden}, fp16_options);

    input.expert_scores_ref =
        torch::zeros({input.params.seq_len * input.params.batch_size, input.params.n_routed_experts}, fp16_options);
    input.expert_scores =
        torch::zeros({input.params.seq_len * input.params.batch_size, input.params.n_routed_experts}, fp16_options);

    input.seq_experts_index =
        torch::zeros({input.params.seq_len * input.params.batch_size, input.params.n_experts_per_token}, i32_options);
    input.seq_experts_index_ref =
        torch::zeros({input.params.seq_len * input.params.batch_size, input.params.n_experts_per_token}, i32_options);

    input.seq_experts_softmax =
        torch::zeros({input.params.seq_len * input.params.batch_size, input.params.n_experts_per_token}, fp16_options);
    input.seq_experts_softmax_ref =
        torch::zeros({input.params.seq_len * input.params.batch_size, input.params.n_experts_per_token}, fp16_options);

    input.source_rows =
        torch::zeros({input.params.seq_len * input.params.batch_size, input.params.n_experts_per_token}, i32_options);
    input.source_rows_ref =
        torch::zeros({input.params.seq_len * input.params.batch_size, input.params.n_experts_per_token}, i32_options);

    input.permuted_experts = torch::zeros(
        {1, input.params.seq_len * input.params.batch_size * input.params.n_experts_per_token}, i32_options);
    input.permuted_experts_ref = torch::zeros(
        {1, input.params.seq_len * input.params.batch_size * input.params.n_experts_per_token}, i32_options);

    input.permuted_rows = torch::zeros(
        {1, input.params.seq_len * input.params.batch_size * input.params.n_experts_per_token}, i32_options);
    input.permuted_rows_ref = torch::zeros(
        {1, input.params.seq_len * input.params.batch_size * input.params.n_experts_per_token}, i32_options);

    input.first_token_offset = torch::zeros({1, input.params.n_routed_experts + 1}, i32_options);
    input.first_token_offset_ref = torch::zeros({1, input.params.n_routed_experts + 1}, i32_options);

    input.expanded_seq_input = torch::zeros(
        {input.params.batch_size * input.params.seq_len * input.params.n_experts_per_token, input.params.d_hidden},
        fp16_options);
    input.expanded_seq_input_ref = torch::zeros(
        {input.params.batch_size * input.params.seq_len * input.params.n_experts_per_token, input.params.d_hidden},
        fp16_options);

    input.expanded_seq_softmax =
        torch::zeros({input.params.batch_size * input.params.seq_len * input.params.n_experts_per_token}, fp16_options);
    input.expanded_seq_softmax_ref =
        torch::zeros({input.params.batch_size * input.params.seq_len * input.params.n_experts_per_token}, fp16_options);

    input.rev_permuted_rows =
        torch::zeros({input.params.batch_size * input.params.seq_len * input.params.n_experts_per_token}, i32_options);
    input.rev_permuted_rows_ref =
        torch::zeros({input.params.batch_size * input.params.seq_len * input.params.n_experts_per_token}, i32_options);

    input.expanded_fc1_gate_splitk =
        torch::zeros({input.params.batch_size * input.params.seq_len * input.params.n_experts_per_token *
                      input.params.d_expert * MAX_SPLITK_FACTOR},
                     fp32_options);

    input.expanded_fc1_up_splitk =
        torch::zeros({input.params.batch_size * input.params.seq_len * input.params.n_experts_per_token *
                      input.params.d_expert * MAX_SPLITK_FACTOR},
                     fp32_options);

    input.expanded_fc2_splitk =
        torch::zeros({input.params.batch_size * input.params.seq_len * input.params.n_experts_per_token *
                      input.params.d_hidden * MAX_SPLITK_FACTOR},
                     fp32_options);

    input.expanded_fc1_output = torch::zeros(
        {input.params.batch_size * input.params.seq_len * input.params.n_experts_per_token, input.params.d_expert},
        fp16_options);
    input.expanded_fc1_output_ref = torch::zeros(
        {input.params.batch_size * input.params.seq_len * input.params.n_experts_per_token, input.params.d_expert},
        fp16_options);

    input.expanded_fc2_splitk =
        torch::zeros({input.params.batch_size * input.params.seq_len * input.params.n_experts_per_token *
                      input.params.d_hidden * MAX_SPLITK_FACTOR},
                     fp32_options);

    input.expanded_fc2_output = torch::zeros(
        {input.params.batch_size * input.params.seq_len * input.params.n_experts_per_token, input.params.d_hidden},
        fp16_options);
    input.expanded_fc2_output_ref = torch::zeros(
        {input.params.batch_size * input.params.seq_len * input.params.n_experts_per_token, input.params.d_hidden},
        fp16_options);

    input.shared_fc1_gate_splitk = torch::zeros(
        {input.params.batch_size * input.params.seq_len * input.params.d_expert * MAX_SPLITK_FACTOR}, fp32_options);

    input.shared_fc1_up_splitk = torch::zeros(
        {input.params.batch_size * input.params.seq_len * input.params.d_expert * MAX_SPLITK_FACTOR}, fp32_options);

    input.shared_fc1_output = torch::zeros({input.params.batch_size * input.params.seq_len, input.params.d_expert},
                                           fp16_options); // no ref for this data

    input.shared_fc2_splitk = torch::zeros(
        {input.params.batch_size * input.params.seq_len * input.params.d_hidden * MAX_SPLITK_FACTOR}, fp32_options);

    input.shared_fc2_output =
        torch::zeros({input.params.batch_size * input.params.seq_len, input.params.d_hidden}, fp16_options);
    input.shared_fc2_output_ref =
        torch::zeros({input.params.batch_size * input.params.seq_len, input.params.d_hidden}, fp16_options);

    input.final_output =
        torch::zeros({input.params.batch_size * input.params.seq_len, input.params.d_hidden}, fp16_options);
    input.final_output_ref =
        torch::zeros({input.params.batch_size * input.params.seq_len, input.params.d_hidden}, fp16_options);
}

int get_params_count() { return input_params_count; }
void *case_get_input(int index) {
    MoEInput *input = new MoEInput();
    input->params = input_params[index];
    generate_input(*input);
    return input;
}
std::vector<Checkee> case_run_kernel(void *input, PerfMetrics *metrics) {
    MoEInput *data = (MoEInput *)input;

    int token_count = data->params.batch_size * data->params.seq_len;
    torch::Tensor input_seq_flat = data->input_seq.view({token_count, data->params.d_hidden});

    data->expert_scores = torch::matmul(input_seq_flat, data->router_weight.transpose(0, 1));
    // for(int i=0; i<data->params.n_routed_experts; i++) {
    //     data->expert_weight_
    //     torch::transpose()
    // }
    // data->expert_scores = torch::softmax(data->expert_scores, 1);
    MoEInputRaw *input_raw = data->to_input_raw();
    // at::hip::HIPStream stream = at::hip::getCurrentHIPStream();
    // hipDeviceSynchronize(); // FIXME: use this to make things correct
    run_topk_func(input_raw, 0, metrics);
    hipDeviceSynchronize();
    return {// Checkee{&data->seq_experts_index, CheckerMode::kElementWise, "seq_experts_index"},
            Checkee{&data->seq_experts_softmax, CheckerMode::kElementWise, "seq_experts_softmax"},
            Checkee{&data->source_rows, CheckerMode::kElementWise, "source_rows"},
            Checkee{&data->permuted_experts, CheckerMode::kElementWise, "permuted_experts"},
            // Checkee{&data->permuted_rows, CheckerMode::kElementWise, "permuted_rows"},
            Checkee{&data->first_token_offset, CheckerMode::kElementWise, "first_token_offset"},
            Checkee{&data->expanded_seq_input, CheckerMode::kElementWise, "expanded_seq_input"},
            Checkee{&data->expanded_seq_softmax, CheckerMode::kElementWise, "expanded_seq_softmax"},
            Checkee{&data->expanded_fc1_output, CheckerMode::kElementWise, "expanded_fc1_output"},
            Checkee{&data->expanded_fc2_output, CheckerMode::kElementWise, "expanded_fc2_output"},
            Checkee{&data->shared_fc2_output, CheckerMode::kElementWise, "shared_fc2_output"},
            // Checkee{&data->rev_permuted_rows, CheckerMode::kElementWise, "rev_permuted_rows"},
            Checkee{&data->final_output, CheckerMode::kElementWise, "final_output"}};
    // return {Checkee{&data->expanded_fc1_output, CheckerMode::kElementWise, "final_output"}};
    // return {Checkee{&data->shared_fc2_output, CheckerMode::kElementWise, "final_output"}};
}
std::vector<Checkee> case_run_ref_kernel(void *input) {
    MoEInput *data = (MoEInput *)input;
    ref_kernel(*data);
    // exact_ref_kernel(*data);
    return {// Checkee{&data->seq_experts_index_ref, CheckerMode::kElementWise, "seq_experts_index"},
            Checkee{&data->seq_experts_softmax_ref, CheckerMode::kElementWise, "seq_experts_softmax"},
            Checkee{&data->source_rows_ref, CheckerMode::kElementWise, "source_rows"},
            Checkee{&data->permuted_experts_ref, CheckerMode::kElementWise, "permuted_experts"},
            // Checkee{&data->permuted_rows_ref, CheckerMode::kElementWise, "permuted_rows"},
            Checkee{&data->first_token_offset_ref, CheckerMode::kElementWise, "first_token_offset"},
            Checkee{&data->expanded_seq_input_ref, CheckerMode::kElementWise, "expanded_seq_input"},
            Checkee{&data->expanded_seq_softmax_ref, CheckerMode::kElementWise, "expanded_seq_softmax"},
            Checkee{&data->expanded_fc1_output_ref, CheckerMode::kElementWise, "expanded_fc1_output"},
            Checkee{&data->expanded_fc2_output_ref, CheckerMode::kElementWise, "expanded_fc2_output"},
            Checkee{&data->shared_fc2_output_ref, CheckerMode::kElementWise, "shared_fc2_output"},
            // Checkee{&data->rev_permuted_rows_ref, CheckerMode::kElementWise, "rev_permuted_rows"},
            Checkee{&data->final_output_ref, CheckerMode::kElementWise, "final_output"}};
    // return {Checkee{&data->final_output_ref, CheckerMode::kElementWise, "final_output"}};
    // return {Checkee{&data->shared_fc2_output_ref, CheckerMode::kElementWise, "final_output"}};
}
const char *case_get_name() { return "MoE TopK"; }
void get_error_tolerance(float *rtol, float *atol) {
    *rtol = 1e-2;
    *atol = 1e-2;
}
void case_destroy(void *input) {
    MoEInput *data = (MoEInput *)input;
    delete data;
}