#include "../checker/checker.h"
#include <ATen/ops/dot.h>
#include <ATen/ops/silu.h>
#include <dlfcn.h>
#include <c10/core/ScalarType.h>
#include <cstdio>
#include <iostream>
#include <vector>
#include "../../src/mla/mla.h"

torch::Tensor forward_rope(const torch::Tensor &x, const torch::Tensor &theta, const int start_pos);

int input_params_count = 1;

InputParams input_params[] = {
    {.prefill = 128, .seed = 9247},
    {.prefill = 512, .seed = 2197},
    {.prefill = 1024, .seed = 9107},
    {.prefill = 2048, .seed = 5291},
    {.prefill = 4096, .seed = 9817},
    {.prefill = 6144, .seed = 5291},
};

struct MLAInput {
    InputParams params;

    torch::Tensor Q_proj_down_weight;
    torch::Tensor KV_proj_down_weight;
    torch::Tensor Q_proj_up_weight;
    torch::Tensor KV_proj_up_weight;
    torch::Tensor wo_weight;
    torch::Tensor x;
    torch::Tensor output_ref;
    torch::Tensor output;
    torch::Tensor kv_cache_input;
    torch::Tensor prefilled_cache;

    torch::Tensor kv_cache_step0;
    torch::Tensor kv_cache_step0_ref;
    torch::Tensor kv_lora;
    torch::Tensor kv_lora_ref;
    torch::Tensor k_rope;
    torch::Tensor k_rope_ref;
    
    torch::Tensor q_lora;
    torch::Tensor q_lora_ref;
    
    torch::Tensor qup_nope;
    torch::Tensor qup_nope_ref;
    torch::Tensor qup_rope;
    torch::Tensor qup_rope_ref;
    torch::Tensor kup_nope;
    torch::Tensor kup_nope_ref;
    torch::Tensor vup;
    torch::Tensor vup_ref;
    torch::Tensor q_rope_step2;
    torch::Tensor q_rope_step2_ref;
    torch::Tensor q_nope;
    torch::Tensor q_nope_ref;
    torch::Tensor q_absorb;
    torch::Tensor q_absorb_ref;
    
    torch::Tensor q_rope_permuted;
    torch::Tensor q_rope_permuted_ref;
    torch::Tensor q_absorb_permuted;
    torch::Tensor q_absorb_permuted_ref;
    torch::Tensor theta;
    torch::Tensor theta_ref;
    torch::Tensor q_rope_final;
    torch::Tensor q_rope_final_ref;
    torch::Tensor k_rope_final;
    torch::Tensor k_rope_final_ref;
    
    torch::Tensor attn_nope;
    torch::Tensor attn_nope_ref;
    torch::Tensor attn_rope;
    torch::Tensor attn_rope_ref;
    torch::Tensor scores;
    torch::Tensor scores_ref;
    torch::Tensor attention;
    torch::Tensor attention_ref;
    
    torch::Tensor o_step4_0;
    torch::Tensor o_step4_0_ref;
    torch::Tensor o_step4_1;
    torch::Tensor o_step4_1_ref;

    MLAInputRaw *to_input_raw() {
        MLAInputRaw *input_raw = new MLAInputRaw();
        input_raw->params = params;
        input_raw->Q_proj_down_weight = Q_proj_down_weight.data_ptr();
        input_raw->KV_proj_down_weight = KV_proj_down_weight.data_ptr();
        input_raw->Q_proj_up_weight = Q_proj_up_weight.data_ptr();
        input_raw->KV_proj_up_weight = KV_proj_up_weight.data_ptr();
        input_raw->wo_weight = wo_weight.data_ptr();
        input_raw->x = x.data_ptr();
        input_raw->output_ref = output_ref.data_ptr();
        input_raw->output = output.data_ptr();
        input_raw->kv_cache_input = kv_cache_input.data_ptr();
        input_raw->prefilled_cache = prefilled_cache.data_ptr();
        input_raw->kv_cache_step0 = kv_cache_step0.data_ptr();
        input_raw->kv_cache_step0_ref = kv_cache_step0_ref.data_ptr();
        input_raw->kv_lora = kv_lora.data_ptr();
        input_raw->kv_lora_ref = kv_lora_ref.data_ptr();
        input_raw->k_rope = k_rope.data_ptr();
        input_raw->k_rope_ref = k_rope_ref.data_ptr();
        input_raw->q_lora = q_lora.data_ptr();
        input_raw->q_lora_ref = q_lora_ref.data_ptr();
        input_raw->qup_nope = qup_nope.data_ptr();
        input_raw->qup_nope_ref = qup_nope_ref.data_ptr();
        input_raw->qup_rope = qup_rope.data_ptr();
        input_raw->qup_rope_ref = qup_rope_ref.data_ptr();
        input_raw->kup_nope = kup_nope.data_ptr();
        input_raw->kup_nope_ref = kup_nope_ref.data_ptr();
        input_raw->vup = vup.data_ptr();
        input_raw->vup_ref = vup_ref.data_ptr();
        input_raw->q_rope_step2 = q_rope_step2.data_ptr();
        input_raw->q_rope_step2_ref = q_rope_step2_ref.data_ptr();
        input_raw->q_nope = q_nope.data_ptr();
        input_raw->q_nope_ref = q_nope_ref.data_ptr();
        input_raw->q_absorb = q_absorb.data_ptr();
        input_raw->q_absorb_ref = q_absorb_ref.data_ptr();
        input_raw->q_rope_permuted = q_rope_permuted.data_ptr();
        input_raw->q_rope_permuted_ref = q_rope_permuted_ref.data_ptr();
        input_raw->q_absorb_permuted = q_absorb_permuted.data_ptr();
        input_raw->q_absorb_permuted_ref = q_absorb_permuted_ref.data_ptr();
        input_raw->theta = theta.data_ptr();
        input_raw->theta_ref = theta_ref.data_ptr();
        input_raw->q_rope_final = q_rope_final.data_ptr();
        input_raw->q_rope_final_ref = q_rope_final_ref.data_ptr();
        input_raw->k_rope_final = k_rope_final.data_ptr();
        input_raw->k_rope_final_ref = k_rope_final_ref.data_ptr();
        input_raw->attn_nope = attn_nope.data_ptr();
        input_raw->attn_nope_ref = attn_nope_ref.data_ptr();
        input_raw->attn_rope = attn_rope.data_ptr();
        input_raw->attn_rope_ref = attn_rope_ref.data_ptr();
        input_raw->scores = scores.data_ptr();
        input_raw->scores_ref = scores_ref.data_ptr();
        input_raw->attention = attention.data_ptr();
        input_raw->attention_ref = attention_ref.data_ptr();
        input_raw->o_step4_0 = o_step4_0.data_ptr();
        input_raw->o_step4_0_ref = o_step4_0_ref.data_ptr();
        input_raw->o_step4_1 = o_step4_1.data_ptr();
        input_raw->o_step4_1_ref = o_step4_1_ref.data_ptr();
        return input_raw;
    }
};

void ref_kernel(MLAInput &input) {
    torch::manual_seed(input.params.seed);
    torch::NoGradGuard no_grad;
    
    auto cuda_options = torch::TensorOptions().device(torch::kCUDA);
    auto bf16_options = cuda_options.dtype(torch::kBFloat16);
    
    int prefill = input.params.prefill;
    auto kv_cache = input.kv_cache_input;
    auto x = input.x;
    auto qdown = input.Q_proj_down_weight;
    auto qup = input.Q_proj_up_weight;
    auto kvdown = input.KV_proj_down_weight;
    auto kvup = input.KV_proj_up_weight;
    auto wo = input.wo_weight;
    
    TORCH_CHECK(seq_len == 1, "seq_len must be 1");
    
    // step 0: hidden -> kv_lora[-1] (kvdown_nope), hidden -> k_rope[-1] (kvdown_rope)
    // kv_cache -> kv_lora, k_rope
    input.kv_cache_step0_ref = kv_cache.slice(1, 0, prefill + seq_len);
    input.kv_cache_step0_ref.slice(1, -1, prefill + seq_len).copy_(torch::einsum("bsd,ld->bsl", {x, kvdown}));
    auto kv_splits = input.kv_cache_step0_ref.split({kv_lora_dim, rope_dim}, -1);
    input.kv_lora_ref = kv_splits[0];
    input.k_rope_ref = kv_splits[1];
    
    // step 1: hidden -> q_lora (qdown)
    input.q_lora_ref = torch::einsum("bsd,ld->bsl", {x, qdown});
    
    // step 2: q_lora -> q_rope (qup_rope), q_lora -> q_nope @ qup_nope @ kup_nope
    auto qup_reshaped = qup.view({n_heads, nope_dim + rope_dim, q_lora_dim});
    auto qup_splits = qup_reshaped.split({nope_dim, rope_dim}, 1);
    input.qup_nope_ref = qup_splits[0];
    input.qup_rope_ref = qup_splits[1];
    
    auto kvup_reshaped = kvup.view({n_heads, nope_dim + v_dim, kv_lora_dim});
    auto kvup_splits = kvup_reshaped.split({nope_dim, v_dim}, 1);
    input.kup_nope_ref = kvup_splits[0];
    input.vup_ref = kvup_splits[1];
    
    input.q_rope_step2_ref = torch::einsum("bsl,hdl->bshd", {input.q_lora_ref, input.qup_rope_ref});
    input.q_nope_ref = torch::einsum("bsl,hdl->bshd", {input.q_lora_ref, input.qup_nope_ref});
    input.q_absorb_ref = torch::einsum("bshd,hdl->bshl", {input.q_nope_ref, input.kup_nope_ref});
    
    // step 2.5: permute and apply rope
    input.q_rope_permuted_ref = input.q_rope_step2_ref.permute({0, 2, 1, 3});
    input.q_absorb_permuted_ref = input.q_absorb_ref.permute({0, 2, 1, 3});
    input.theta_ref = torch::pow(10000.0, -torch::arange(0, rope_dim / 2, bf16_options) / (rope_dim / 2.0));
    input.q_rope_final_ref = forward_rope(input.q_rope_permuted_ref, input.theta_ref, prefill + seq_len - 1);
    input.k_rope_final_ref = forward_rope(input.k_rope_ref, input.theta_ref, 0);
    
    // step 3: q_nope @ kv_lora -> attn_nope, q_rope @ kv_rope -> attn_rope
    input.attn_nope_ref = torch::einsum("bhsl,bpl->bhsp", {input.q_absorb_permuted_ref, input.kv_lora_ref});
    input.attn_rope_ref = torch::einsum("bhsd,bpd->bhsp", {input.q_rope_final_ref, input.k_rope_final_ref});
    input.scores_ref = input.attn_nope_ref + input.attn_rope_ref;
    input.scores_ref = input.scores_ref / std::sqrt(rope_dim + nope_dim);
    input.attention_ref = torch::softmax(input.scores_ref, -1);
    
    // step 4: attn @ kv_lora @ vup @ wo -> output
    input.o_step4_0_ref = torch::einsum("bhsp,bpl->bhsl", {input.attention_ref, input.kv_lora_ref});
    input.o_step4_1_ref = torch::einsum("bhsl,hvl->bhsv", {input.o_step4_0_ref, input.vup_ref});
    auto wo_reshaped = wo.view({hidden_dim, n_heads, v_dim});
    input.output_ref = torch::einsum("bhsv,dhv->bsd", {input.o_step4_1_ref, wo_reshaped});
}

typedef void (*run_t)(MLAInputRaw *, PerfMetrics *);
run_t run_mla_func;

void case_initialize() {
    input_params_count = sizeof(input_params) / sizeof(input_params[0]);

    // Load the symbol
    void *handle = dlopen("libmla.so", RTLD_NOW | RTLD_DEEPBIND);
    if (!handle) {
        std::cerr << "Cannot open library: " << dlerror() << std::endl;
        abort();
    }

    run_mla_func = (run_t)dlsym(handle, "run_mla");
    if (!run_mla_func) {
        std::cerr << "Cannot load symbol 'run_mla': " << dlerror() << std::endl;
        dlclose(handle);
        abort();
    }
}

torch::Tensor forward_kv_cache(const torch::Tensor &c_kv, int &kv_cache_seq_len, torch::Tensor &kv_cache) {
    TORCH_CHECK(kv_cache_seq_len + c_kv.size(1) <= kv_cache.size(1), "KV Cache Exceeded");
    kv_cache = kv_cache.to(c_kv.dtype());
    kv_cache.slice(1, kv_cache_seq_len, kv_cache_seq_len + c_kv.size(1)).copy_(c_kv);
    kv_cache_seq_len += c_kv.size(1);
    return kv_cache.slice(1, 0, kv_cache_seq_len);
}

torch::Tensor rotate_half(const torch::Tensor &x) {
    int d = x.size(-1);
    auto x1 = x.slice(-1, 0, d / 2);
    auto x2 = x.slice(-1, d / 2, d);
    return torch::cat({-x2, x1}, -1);
}

torch::Tensor forward_rope(const torch::Tensor &x, const torch::Tensor &theta, const int start_pos) {
    int seq_len = x.size(-2);
    int d_model = x.size(-1);

    TORCH_CHECK(d_model == rope_dim, "d_model must equal rope_dim");

    auto seq_idx = torch::arange(start_pos, start_pos + seq_len, torch::TensorOptions().device(x.device()));

    auto idx_theta = torch::outer(seq_idx, theta);

    auto idx_theta2 = torch::cat({idx_theta, idx_theta}, -1);

    auto cos = idx_theta2.cos().to(torch::kBFloat16);
    auto sin = idx_theta2.sin().to(torch::kBFloat16);

    return x * cos + rotate_half(x) * sin;
}

void generate_input(MLAInput &input) {
    torch::manual_seed(input.params.seed);
    auto cuda_options = torch::TensorOptions().device(torch::kCUDA);
    auto fp16_options = cuda_options.dtype(torch::kFloat16);
    auto fp32_options = cuda_options.dtype(torch::kFloat32);
    auto bf16_options = cuda_options.dtype(torch::kBFloat16);
    auto i16_options = cuda_options.dtype(torch::kInt16);
    auto i32_options = cuda_options.dtype(torch::kInt32);

    input.x = torch::randn({batch_size, 1, hidden_dim}, bf16_options);

    input.Q_proj_down_weight = torch::randn({dq, hidden_dim}, bf16_options) / std::sqrt(hidden_dim);
    input.KV_proj_down_weight = torch::randn({512 + 64, hidden_dim}, bf16_options) / std::sqrt(hidden_dim);
    input.Q_proj_up_weight = torch::randn({(128 + 64) * 128, dq}, bf16_options) / std::sqrt(dq);
    input.KV_proj_up_weight = torch::randn({(128 + 128) * 128, 512}, bf16_options) / std::sqrt(512);
    input.wo_weight = torch::randn({hidden_dim, 128 * 128}, bf16_options) / std::sqrt(128 * 128);

    input.output = torch::zeros({batch_size, 1, hidden_dim}, bf16_options);
    input.output_ref = torch::zeros({batch_size, 1, hidden_dim}, bf16_options);

    input.kv_cache_step0_ref = torch::zeros({batch_size, max_seq_len, kv_lora_dim + rope_dim}, bf16_options);
    input.kv_cache_step0 = torch::zeros({batch_size, max_seq_len, kv_lora_dim + rope_dim}, bf16_options);
    input.kv_lora_ref = torch::zeros({batch_size, max_seq_len, kv_lora_dim}, bf16_options);
    input.kv_lora = torch::zeros({batch_size, max_seq_len, kv_lora_dim}, bf16_options);
    input.k_rope_ref = torch::zeros({batch_size, max_seq_len, rope_dim}, bf16_options);
    input.k_rope = torch::zeros({batch_size, max_seq_len, rope_dim}, bf16_options);
    
    input.q_lora_ref = torch::zeros({batch_size, seq_len, q_lora_dim}, bf16_options);
    input.q_lora = torch::zeros({batch_size, seq_len, q_lora_dim}, bf16_options);
    
    input.qup_nope_ref = torch::zeros({n_heads, nope_dim, q_lora_dim}, bf16_options);
    input.qup_nope = torch::zeros({n_heads, nope_dim, q_lora_dim}, bf16_options);
    input.qup_rope_ref = torch::zeros({n_heads, rope_dim, q_lora_dim}, bf16_options);
    input.qup_rope = torch::zeros({n_heads, rope_dim, q_lora_dim}, bf16_options);
    input.kup_nope_ref = torch::zeros({n_heads, nope_dim, kv_lora_dim}, bf16_options);
    input.kup_nope = torch::zeros({n_heads, nope_dim, kv_lora_dim}, bf16_options);
    input.vup_ref = torch::zeros({n_heads, v_dim, kv_lora_dim}, bf16_options);
    input.vup = torch::zeros({n_heads, v_dim, kv_lora_dim}, bf16_options);
    input.q_rope_step2_ref = torch::zeros({batch_size, seq_len, n_heads, rope_dim}, bf16_options);
    input.q_rope_step2 = torch::zeros({batch_size, seq_len, n_heads, rope_dim}, bf16_options);
    input.q_nope_ref = torch::zeros({batch_size, seq_len, n_heads, nope_dim}, bf16_options);
    input.q_nope = torch::zeros({batch_size, seq_len, n_heads, nope_dim}, bf16_options);
    input.q_absorb_ref = torch::zeros({batch_size, seq_len, n_heads, kv_lora_dim}, bf16_options);
    input.q_absorb = torch::zeros({batch_size, seq_len, n_heads, kv_lora_dim}, bf16_options);
    
    input.q_rope_permuted_ref = torch::zeros({batch_size, n_heads, seq_len, rope_dim}, bf16_options);
    input.q_rope_permuted = torch::zeros({batch_size, n_heads, seq_len, rope_dim}, bf16_options);
    input.q_absorb_permuted_ref = torch::zeros({batch_size, n_heads, seq_len, kv_lora_dim}, bf16_options);
    input.q_absorb_permuted = torch::zeros({batch_size, n_heads, seq_len, kv_lora_dim}, bf16_options);
    input.theta_ref = torch::zeros({rope_dim / 2}, bf16_options);
    input.theta = torch::zeros({rope_dim / 2}, bf16_options);
    input.q_rope_final_ref = torch::zeros({batch_size, n_heads, seq_len, rope_dim}, bf16_options);
    input.q_rope_final = torch::zeros({batch_size, n_heads, seq_len, rope_dim}, bf16_options);
    input.k_rope_final_ref = torch::zeros({batch_size, max_seq_len, rope_dim}, bf16_options);
    input.k_rope_final = torch::zeros({batch_size, max_seq_len, rope_dim}, bf16_options);
    
    input.attn_nope_ref = torch::zeros({batch_size, n_heads, seq_len, max_seq_len}, bf16_options);
    input.attn_nope = torch::zeros({batch_size, n_heads, seq_len, max_seq_len}, bf16_options);
    input.attn_rope_ref = torch::zeros({batch_size, n_heads, seq_len, max_seq_len}, bf16_options);
    input.attn_rope = torch::zeros({batch_size, n_heads, seq_len, max_seq_len}, bf16_options);
    input.scores_ref = torch::zeros({batch_size, n_heads, seq_len, max_seq_len}, bf16_options);
    input.scores = torch::zeros({batch_size, n_heads, seq_len, max_seq_len}, bf16_options);
    input.attention_ref = torch::zeros({batch_size, n_heads, seq_len, max_seq_len}, bf16_options);
    input.attention = torch::zeros({batch_size, n_heads, seq_len, max_seq_len}, bf16_options);
    
    input.o_step4_0_ref = torch::zeros({batch_size, n_heads, seq_len, kv_lora_dim}, bf16_options);
    input.o_step4_0 = torch::zeros({batch_size, n_heads, seq_len, kv_lora_dim}, bf16_options);
    input.o_step4_1_ref = torch::zeros({batch_size, n_heads, seq_len, v_dim}, bf16_options);
    input.o_step4_1 = torch::zeros({batch_size, n_heads, seq_len, v_dim}, bf16_options);

    int kv_cache_seq_len = 0;

    input.kv_cache_input = torch::zeros({batch_size, max_seq_len, kv_lora_dim + rope_dim}, bf16_options);
    input.prefilled_cache = torch::randn({batch_size, input.params.prefill, kv_lora_dim + rope_dim}, bf16_options);
    input.kv_cache_input = forward_kv_cache(input.prefilled_cache, kv_cache_seq_len, input.kv_cache_input);
}

int get_params_count() { return input_params_count; }
void *case_get_input(int index) {
    MLAInput *input = new MLAInput();
    input->params = input_params[index];
    generate_input(*input);
    return input;
}
std::vector<Checkee> case_run_kernel(void *input, PerfMetrics *metrics) {
    MLAInput *data = (MLAInput *)input;
    run_mla_func(data->to_input_raw(), metrics);

    return {Checkee{&data->output, CheckerMode::kElementWise, "final_result"}};
}
std::vector<Checkee> case_run_ref_kernel(void *input) {
    MLAInput *data = (MLAInput *)input;
    ref_kernel(*data);
    // exact_ref_kernel(*data);
    return {Checkee{&data->output_ref, CheckerMode::kElementWise, "final_result"}};
}
const char *case_get_name() { return "MLA"; }
void get_error_tolerance(float *rtol, float *atol) {
    *rtol = 2e-2;
    *atol = 1e-3;
}
void case_destroy(void *input) {
    MLAInput *data = (MLAInput *)input;
    delete data;
}