#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/amd_detail/amd_hip_fp8.h>
#include "moe.h"
#include "gpu_types.h"
#include "gpu_libs.h"
#include "moe.h"
#include "timer.h"
#include <hip/hip_runtime.h>
#include <stdio.h> // Include for printf

#include "ck/ck.hpp"

#include "ck/utility/data_type.hpp"

#include <hipcub/hipcub.hpp>
#include <hipcub/block/block_radix_sort.hpp>

#include "moe_topk_kernel.cpp"
#include "moe_gemm_pipeline_kernel.cpp"
#include "transpose.cpp"

#include "../../tests/checker/metrics.h"

#include <hipblas-common/hipblas-common.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include "gemm_thirdparty.cpp"
#include <c10/hip/HIPStream.h>

HOST_CODE_BELOW

size_t required_temp_storage_bytes = 0;

std::vector<std::shared_ptr<KernelTimer>> timers;

constexpr int MAX_SEQ_LEN = 8192;
constexpr int MAX_D_HIDDEN = 7168;
constexpr int MAX_D_EXPERT = 2048;
constexpr int MAX_N_ROUTED_EXPERTS = 32;
constexpr int MAX_N_EXPERT_PER_TOKEN = 4;

int h_IDX[33];

hipStream_t stream0, stream1, stream2;

hipEvent_t event0, event1, event11;

const __FP16_TYPE (**expert_weights_gate_d)[MAX_D_EXPERT] = nullptr;
const __FP16_TYPE (**expert_weights_up_d)[MAX_D_EXPERT] = nullptr;
const __FP16_TYPE (**expert_weights_down_d)[MAX_D_HIDDEN] = nullptr;

__FP16_TYPE (**expert_weights_gate_transposed_d)[MAX_D_HIDDEN] = nullptr;
__FP16_TYPE (**expert_weights_up_transposed_d)[MAX_D_HIDDEN] = nullptr;
__FP16_TYPE (**expert_weights_down_transposed_d)[MAX_D_EXPERT] = nullptr;

void *temp_storage_for_hipcub = nullptr;

bool env_initialized = false;

GemmThirdParty gemm_thirdparty;

template <int N_ROUTED_EXPERTS> constexpr inline int experts_to_bits() {
    return (int)(log2(2 * N_ROUTED_EXPERTS - 1)) + 1;
}

void initialize_env() {
    if (env_initialized) {
        return;
    }
    LIB_CALL(hipStreamCreateWithFlags(&stream0, hipStreamNonBlocking));
    LIB_CALL(hipStreamCreateWithFlags(&stream1, hipStreamNonBlocking));
    LIB_CALL(hipStreamCreateWithFlags(&stream2, hipStreamNonBlocking));

    LIB_CALL(hipEventCreate(&event0));
    LIB_CALL(hipEventCreate(&event1));
    LIB_CALL(hipEventCreate(&event11));

    LIB_CALL(hipMalloc(&expert_weights_gate_d, MAX_N_ROUTED_EXPERTS * sizeof(void *)));
    LIB_CALL(hipMalloc(&expert_weights_up_d, MAX_N_ROUTED_EXPERTS * sizeof(void *)));
    LIB_CALL(hipMalloc(&expert_weights_down_d, MAX_N_ROUTED_EXPERTS * sizeof(void *)));

    LIB_CALL(hipMalloc(&expert_weights_gate_transposed_d, MAX_N_ROUTED_EXPERTS * sizeof(void *)));
    LIB_CALL(hipMalloc(&expert_weights_up_transposed_d, MAX_N_ROUTED_EXPERTS * sizeof(void *)));
    LIB_CALL(hipMalloc(&expert_weights_down_transposed_d, MAX_N_ROUTED_EXPERTS * sizeof(void *)));

    int *dummy_index = nullptr;

    LIB_CALL(hipcub::DeviceRadixSort::SortPairs(
        temp_storage_for_hipcub, required_temp_storage_bytes, dummy_index, dummy_index, dummy_index, dummy_index,
        MAX_SEQ_LEN * MAX_N_EXPERT_PER_TOKEN, 0, experts_to_bits<MAX_N_ROUTED_EXPERTS>()));
    LIB_CALL(HOST_TYPE(Malloc)(&temp_storage_for_hipcub, required_temp_storage_bytes));

    initialize_gemm_thirdparty(gemm_thirdparty);

    env_initialized = true;
}

#define DISPATCH_MOE(SEQ_LEN_T, D_HIDDEN_T, D_EXPERT_T, N_ROUTED_EXPERTS, N_EXPERT_PER_TOKEN, N_SHARED_EXPERTS,        \
                     SPLITK_FACTOR_ROUTED_FC1, SPLITK_FACTOR_ROUTED_FC2, SPLITK_FACTOR_SHARED_FC1,                     \
                     SPLITK_FACTOR_SHARED_FC2, SCHEDULER_BLOCK_COUNT, BM, BN, BK, WARP_M, WARP_N)                      \
    if (seq_len == (SEQ_LEN_T) && d_hidden == (D_HIDDEN_T) && d_expert == (D_EXPERT_T) &&                              \
        n_routed_experts == (N_ROUTED_EXPERTS) && n_experts_per_token == (N_EXPERT_PER_TOKEN) &&                       \
        n_shared_experts == (N_SHARED_EXPERTS)) {                                                                      \
        run_topk_impl<SEQ_LEN_T, D_HIDDEN_T, D_EXPERT_T, N_ROUTED_EXPERTS, N_EXPERT_PER_TOKEN, N_SHARED_EXPERTS,       \
                      SPLITK_FACTOR_ROUTED_FC1, SPLITK_FACTOR_ROUTED_FC2, SPLITK_FACTOR_SHARED_FC1,                    \
                      SPLITK_FACTOR_SHARED_FC2, SCHEDULER_BLOCK_COUNT, BM, BN, BK, WARP_M, WARP_N>(                    \
            router_weight, input_seq, expert_scores, seq_experts_index, seq_experts_softmax, source_rows,              \
            permuted_experts, permuted_rows, first_token_offset, expanded_input_seq, expanded_experts_softmax,         \
            rev_permuted_rows, expanded_fc1_output, expanded_fc2_output, expert_weight_gate, expert_weight_up,         \
            expert_weight_down, expert_weight_gate_transposed, expert_weight_up_transposed,                            \
            expert_weight_down_transposed, shared_expert_weight_gate, shared_expert_weight_up,                         \
            shared_expert_weight_down, shared_expert_weight_gate_transposed, shared_expert_weight_up_transposed,       \
            shared_expert_weight_down_transposed, shared_fc2_output, final_output, shared_fc1_output,                  \
            expanded_fc1_gate_splitk, expanded_fc1_up_splitk, expanded_fc2_splitk, shared_fc1_gate_splitk,             \
            shared_fc1_up_splitk, shared_fc2_splitk, stream, metrics);                                                 \
        return 0; /* Exit after dispatching */                                                                         \
    }

template <int SEQ_LEN, int N_ROUTED_EXPERTS, int N_EXPERT_PER_TOKEN>
void sort_experts(int *seq_experts_index, int *source_rows, int *permuted_experts, int *permuted_rows,
                  hipStream_t stream) {

    LIB_CALL(hipcub::DeviceRadixSort::SortPairs(
        temp_storage_for_hipcub, required_temp_storage_bytes, seq_experts_index, permuted_experts, source_rows,
        permuted_rows, SEQ_LEN * N_EXPERT_PER_TOKEN, 0, experts_to_bits<N_ROUTED_EXPERTS>(), stream));
}

template <ck::index_t... Is> using S = ck::Sequence<Is...>;
template <int SEQ_LEN, int D_HIDDEN, int D_EXPERT, int N_ROUTED_EXPERTS, int N_EXPERT_PER_TOKEN, int N_SHARED_EXPERTS,
          int SPLITK_FACTOR_ROUTED_FC1, int SPLITK_FACTOR_ROUTED_FC2, int SPLITK_FACTOR_SHARED_FC1,
          int SPLITK_FACTOR_SHARED_FC2, int SCHEDULER_BLOCK_COUNT, int BM, int BN, int BK, int WARP_M, int WARP_N>
int run_topk_impl(const __FP16_TYPE *router_weight, const __FP16_TYPE *input_seq, __FP16_TYPE *expert_scores,
                  int *seq_experts_index, __FP16_TYPE *seq_experts_softmax, int *source_rows, int *permuted_experts,
                  int *permuted_rows, int *first_token_offset, __FP16_TYPE *expanded_input_seq,
                  __FP16_TYPE *expanded_experts_softmax, int *rev_permuted_rows, __FP16_TYPE *expanded_fc1_output,
                  __FP16_TYPE *expanded_fc2_output, void **expert_weight_gate, void **expert_weight_up,
                  void **expert_weight_down, __FP16_TYPE *expert_weight_gate_transposed,
                  __FP16_TYPE *expert_weight_up_transposed, __FP16_TYPE *expert_weight_down_transposed,
                  __FP16_TYPE *shared_expert_weight_gate, __FP16_TYPE *shared_expert_weight_up,
                  __FP16_TYPE *shared_expert_weight_down, __FP16_TYPE *shared_expert_weight_gate_transposed,
                  __FP16_TYPE *shared_expert_weight_up_transposed, __FP16_TYPE *shared_expert_weight_down_transposed,
                  __FP16_TYPE *shared_fc2_output, __FP16_TYPE *final_output, __FP16_TYPE *shared_fc1_output,
                  float *expanded_fc1_gate_splitk, float *expanded_fc1_up_splitk, float *expanded_fc2_splitk,
                  float *shared_fc1_gate_splitk, float *shared_fc1_up_splitk, float *shared_fc2_splitk,
                  hipStream_t stream0, PerfMetrics *metrics) {

    static constexpr int block_dim_x = WAVE_SIZE;
    static constexpr int block_dim_y = 16;

    LIB_CALL(hipEventRecord(event0, 0));
    LIB_CALL(hipStreamWaitEvent(stream0, event0, 0));

    ExpertWeights expert_weights_gate_up_h, expert_weights_down_h;

    for (int i = 0; i < N_ROUTED_EXPERTS; i++) {
        expert_weights_gate_up_h.ptr[i] = expert_weight_gate[i];
        expert_weights_gate_up_h.ptr[i + N_ROUTED_EXPERTS] = expert_weight_up[i];
        expert_weights_down_h.ptr[i] = expert_weight_down[i];
    }

    dim3 block_dim(block_dim_x, block_dim_y);
    constexpr int VPT = N_ROUTED_EXPERTS == 4 ? 4 : (N_ROUTED_EXPERTS == 8 ? 8 : 16); // TODO: tune this!
    static constexpr int ELEMS_PER_ROW = N_ROUTED_EXPERTS;
    static constexpr int ELEMS_PER_WARP = WAVE_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELEMS_PER_WARP / ELEMS_PER_ROW;
    static constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;
    // static_assert(SEQ_LEN % ROWS_PER_BLOCK == 0, "SEQ_LEN must be divisible by ROWS_PER_BLOCK");

    if (metrics) {
        metrics->entries[2].time = 0;
        metrics->entries[2].gflops = 0;
    }

    const int block_count = ceil_div(SEQ_LEN, ROWS_PER_BLOCK);
    {

        KernelTimerScoped moe_misc_timer(timers, 0, metrics ? &metrics->entries[1].time : nullptr,
                                         metrics ? &metrics->entries[1].gflops : nullptr, stream0);

        // blockDim.x should be WAVE_SIZE
        hipLaunchKernelGGL(HIP_KERNEL_NAME(topk_kernel < VPT, N_EXPERT_PER_TOKEN, N_ROUTED_EXPERTS, VPT <= 4 ? 4 : 8,
                                           SEQ_LEN, true,
                                           typename std::conditional<VPT <= 4, ck::half4_t, ck::half8_t>::type >),
                           dim3(block_count), dim3(block_dim), 0, stream0, expert_scores, seq_experts_index,
                           seq_experts_softmax, source_rows);

        sort_experts<SEQ_LEN, N_ROUTED_EXPERTS, N_EXPERT_PER_TOKEN>(seq_experts_index, source_rows, permuted_experts,
                                                                    permuted_rows, stream0);

        compute_first_token_offset<SEQ_LEN, N_ROUTED_EXPERTS, N_EXPERT_PER_TOKEN>
            <<<ceil_div(SEQ_LEN * N_EXPERT_PER_TOKEN, 256), 256, 0, stream0>>>(permuted_experts, first_token_offset);
        {
            constexpr int VPT = 16;
            constexpr int BLOCK_SIZE = 768 / (D_HIDDEN / VPT) * (D_HIDDEN / VPT);
            constexpr int THREADS_PER_ROW = D_HIDDEN / VPT;
            static_assert(BLOCK_SIZE % THREADS_PER_ROW == 0);
            constexpr int BLOCK_COUNT = ceil_div(SEQ_LEN * N_EXPERT_PER_TOKEN * THREADS_PER_ROW, BLOCK_SIZE);
            expand_input<SEQ_LEN, N_ROUTED_EXPERTS, N_EXPERT_PER_TOKEN, D_HIDDEN, VPT, 8, ck::half8_t>
                <<<BLOCK_COUNT, BLOCK_SIZE, 0, stream0>>>(input_seq, expanded_input_seq, seq_experts_softmax,
                                                          expanded_experts_softmax, permuted_rows, rev_permuted_rows);
        }
    }

    LIB_CALL(hipMemcpyAsync(h_IDX, first_token_offset, (N_ROUTED_EXPERTS + 1) * sizeof(int), hipMemcpyDeviceToHost, stream0));
    // CAUTION: should we sync here?

    // routed_fc1 calculation
    {
        KernelTimerScoped routed_fc1_timer(timers, 2ll * SEQ_LEN * N_EXPERT_PER_TOKEN * D_EXPERT * D_HIDDEN * 2,
                                           metrics ? &metrics->entries[3].time : nullptr,
                                           metrics ? &metrics->entries[3].gflops : nullptr, stream0);

        LaunchGroupGEMM<false, false, true, SEQ_LEN * N_EXPERT_PER_TOKEN, D_EXPERT, D_HIDDEN, N_ROUTED_EXPERTS>(
            gemm_thirdparty, expanded_input_seq, nullptr, expert_weights_gate_up_h,
            reinterpret_cast<__FP16_TYPE *>(expanded_fc1_gate_splitk),
            reinterpret_cast<__FP16_TYPE *>(expanded_fc1_up_splitk), h_IDX,
            best_algo_index<SEQ_LEN * N_EXPERT_PER_TOKEN, D_EXPERT, D_HIDDEN, true>(), stream0);
    }

    constexpr uint32_t REDUCE_BLOCK = 512;

    {
        KernelTimerScoped routed_fc1_reduce_timer(timers, 0, metrics ? &metrics->entries[7].time : nullptr,
                                                  metrics ? &metrics->entries[7].gflops : nullptr, stream0);

        constexpr int BLOCK_SIZE = 512;

        constexpr int BLOCK_COUNT = ceil_div(SEQ_LEN * N_EXPERT_PER_TOKEN * D_EXPERT, BLOCK_SIZE * 4);

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                moe_gemm_reduce_ab<true, __FP16_TYPE, SEQ_LEN * N_EXPERT_PER_TOKEN, D_EXPERT, 4, ck::half4_t>),
            dim3(BLOCK_COUNT), dim3(BLOCK_SIZE), 0, stream0, reinterpret_cast<__FP16_TYPE *>(expanded_fc1_gate_splitk),
            reinterpret_cast<__FP16_TYPE *>(expanded_fc1_up_splitk), expanded_fc1_output);
    }


    // routed_fc2 calculation
    {
        KernelTimerScoped routed_fc2_timer(timers, 2ll * SEQ_LEN * N_EXPERT_PER_TOKEN * D_HIDDEN * D_EXPERT,
                                           metrics ? &metrics->entries[4].time : nullptr,
                                           metrics ? &metrics->entries[4].gflops : nullptr, stream0);

        LaunchGroupGEMM<false, false, false, SEQ_LEN * N_EXPERT_PER_TOKEN, D_HIDDEN, D_EXPERT, N_ROUTED_EXPERTS,
                        __FP16_TYPE>(gemm_thirdparty, expanded_fc1_output, nullptr, expert_weights_down_h,
                                     reinterpret_cast<__FP16_TYPE *>(expanded_fc2_output), nullptr, h_IDX,
                                     best_algo_index<SEQ_LEN * N_EXPERT_PER_TOKEN, D_EXPERT, D_HIDDEN, true>(),
                                     stream0);
    }

    {
        KernelTimerScoped routed_fc2_reduce_timer(timers, 0, metrics ? &metrics->entries[8].time : nullptr,
                                                  metrics ? &metrics->entries[8].gflops : nullptr, stream0);

        constexpr int BLOCK_SIZE = 256;
        constexpr int BLOCK_COUNT = ceil_div(SEQ_LEN * N_EXPERT_PER_TOKEN * D_HIDDEN, BLOCK_SIZE * 4);
    }

    // LIB_CALL(hipEventRecord(event11, stream0));
    // LIB_CALL(hipStreamWaitEvent(stream1, event11, 0));
    // LIB_CALL(hipStreamWaitEvent(stream2, event11, 0));

    // shared_fc1 calculation

    {

        KernelTimerScoped shared_fc1_timer(timers, 2ll * SEQ_LEN * D_EXPERT * D_HIDDEN * 2,
                                           metrics ? &metrics->entries[5].time : nullptr,
                                           metrics ? &metrics->entries[5].gflops : nullptr, stream0);

        LaunchGEMM<false, SEQ_LEN, D_EXPERT, D_HIDDEN>(gemm_thirdparty, input_seq, shared_expert_weight_gate,
                                                       reinterpret_cast<__FP16_TYPE *>(shared_fc1_gate_splitk),
                                                       best_algo_index<SEQ_LEN, D_EXPERT, D_HIDDEN, false>(), stream0);

        LaunchGEMM<false, SEQ_LEN, D_EXPERT, D_HIDDEN>(gemm_thirdparty, input_seq, shared_expert_weight_up,
                                                       reinterpret_cast<__FP16_TYPE *>(shared_fc1_up_splitk),
                                                       best_algo_index<SEQ_LEN, D_EXPERT, D_HIDDEN, false>(), stream0);
    }

    {
        KernelTimerScoped shared_fc1_reduce_timer(timers, 0, metrics ? &metrics->entries[9].time : nullptr,
                                                  metrics ? &metrics->entries[9].gflops : nullptr, stream0);

        constexpr int BLOCK_SIZE = 512;

        constexpr int BLOCK_COUNT = ceil_div(SEQ_LEN * D_EXPERT, BLOCK_SIZE * 4);

        moe_gemm_reduce_ab<true, __FP16_TYPE, SEQ_LEN, D_EXPERT, 4, ck::half4_t>
            <<<BLOCK_COUNT, BLOCK_SIZE, 0, stream0>>>(reinterpret_cast<__FP16_TYPE *>(shared_fc1_gate_splitk),
                                                      reinterpret_cast<__FP16_TYPE *>(shared_fc1_up_splitk),
                                                      shared_fc1_output);
    }

    // shared_fc2 calculation

    {

        KernelTimerScoped shared_fc2_timer(timers, 2ll * SEQ_LEN * D_HIDDEN * D_EXPERT,
                                           metrics ? &metrics->entries[6].time : nullptr,
                                           metrics ? &metrics->entries[6].gflops : nullptr, stream0);

        LaunchGEMM<false, SEQ_LEN, D_HIDDEN, D_EXPERT>(gemm_thirdparty, shared_fc1_output, shared_expert_weight_down,
                                                       reinterpret_cast<__FP16_TYPE *>(shared_fc2_output),
                                                       best_algo_index<SEQ_LEN, D_HIDDEN, D_EXPERT, false>(), stream0);
    }

    {
        KernelTimerScoped shared_fc2_reduce_timer(timers, 0, metrics ? &metrics->entries[10].time : nullptr,
                                                  metrics ? &metrics->entries[10].gflops : nullptr, stream0);
    }

    // LIB_CALL(hipEventRecord(event0, stream0));
    // LIB_CALL(hipEventRecord(event1, stream1));

    // LIB_CALL(hipStreamWaitEvent(stream2, event0, 0));
    // LIB_CALL(hipStreamWaitEvent(stream2, event1, 0));

    {

        KernelTimerScoped reduce_output_timer(timers, 0, metrics ? &metrics->entries[11].time : nullptr,
                                              metrics ? &metrics->entries[11].gflops : nullptr, stream0);

        constexpr int VPT = 32;
        constexpr int BLOCK_SIZE = 768 / (D_HIDDEN / VPT) * (D_HIDDEN / VPT);
        constexpr int THREADS_PER_ROW = D_HIDDEN / VPT;
        static_assert(BLOCK_SIZE % THREADS_PER_ROW == 0);
        constexpr int BLOCK_COUNT = ceil_div(SEQ_LEN * THREADS_PER_ROW, BLOCK_SIZE);
        reduce_output<SEQ_LEN, N_ROUTED_EXPERTS, N_EXPERT_PER_TOKEN, D_HIDDEN, VPT, 8, ck::half8_t>
            <<<BLOCK_COUNT, BLOCK_SIZE, 0, stream0>>>(
                reinterpret_cast<const __FP16_TYPE(*)[D_HIDDEN]>(expanded_fc2_output),
                reinterpret_cast<const __FP16_TYPE(*)[D_HIDDEN]>(shared_fc2_output),
                reinterpret_cast<__FP16_TYPE(*)[D_HIDDEN]>(final_output), rev_permuted_rows, permuted_experts,
                expanded_experts_softmax);
    }

    // LIB_CALL(hipStreamSynchronize(stream0));

    return 0;
}

extern "C" {

int run_topk(MoEInputRaw *input, hipStream_t stream, PerfMetrics *metrics) {
    initialize_env();

    // if (stream == nullptr) {
    //     stream = stream0;
    // }

    if (metrics) {
        metrics->count = 12;
        strcpy(metrics->entries[0].name, "overall");
        strcpy(metrics->entries[1].name, "moe_misc");
        strcpy(metrics->entries[2].name, "transpose");
        strcpy(metrics->entries[3].name, "routed_fc1");
        strcpy(metrics->entries[4].name, "routed_fc2");
        strcpy(metrics->entries[5].name, "shared_fc1");
        strcpy(metrics->entries[6].name, "shared_fc2");
        strcpy(metrics->entries[7].name, "routed_fc1_reduce");
        strcpy(metrics->entries[8].name, "routed_fc2_reduce");
        strcpy(metrics->entries[9].name, "shared_fc1_reduce");
        strcpy(metrics->entries[10].name, "shared_fc2_reduce");
        strcpy(metrics->entries[11].name, "reduce_output");
    }

    const int seq_len = input->params.seq_len * input->params.batch_size;
    const int d_hidden = input->params.d_hidden;
    const int d_expert = input->params.d_expert;
    const int n_routed_experts = input->params.n_routed_experts;
    const int n_experts_per_token = input->params.n_experts_per_token;
    const int n_shared_experts = input->params.n_shared_experts;
    KernelTimerScoped overall_timer(timers, 0, metrics ? &metrics->entries[0].time : nullptr,
                                    metrics ? &metrics->entries[0].gflops : nullptr);
    const __FP16_TYPE *router_weight = static_cast<const __FP16_TYPE *>(input->router_weight);
    const __FP16_TYPE *input_seq = static_cast<const __FP16_TYPE *>(input->input_seq);
    __FP16_TYPE *expert_scores = static_cast<__FP16_TYPE *>(input->expert_scores);
    int *seq_experts_index = static_cast<int *>(input->seq_experts_index);
    __FP16_TYPE *seq_experts_softmax = static_cast<__FP16_TYPE *>(input->seq_experts_softmax);
    int *source_rows = static_cast<int *>(input->source_rows);
    int *permuted_experts = static_cast<int *>(input->permuted_experts);
    int *permuted_rows = static_cast<int *>(input->permuted_rows);
    int *first_token_offset = static_cast<int *>(input->first_token_offset);
    __FP16_TYPE *expanded_input_seq = static_cast<__FP16_TYPE *>(input->expanded_seq_input);
    __FP16_TYPE *expanded_experts_softmax = static_cast<__FP16_TYPE *>(input->expanded_seq_softmax);
    float *expanded_fc1_gate_splitk = static_cast<float *>(input->expanded_fc1_gate_splitk);
    float *expanded_fc1_up_splitk = static_cast<float *>(input->expanded_fc1_up_splitk);
    __FP16_TYPE *expanded_fc1_output = static_cast<__FP16_TYPE *>(input->expanded_fc1_output);
    float *expanded_fc2_splitk = static_cast<float *>(input->expanded_fc2_splitk);
    __FP16_TYPE *expanded_fc2_output = static_cast<__FP16_TYPE *>(input->expanded_fc2_output);
    int *rev_permuted_rows = static_cast<int *>(input->rev_permuted_rows);
    void **expert_weight_gate = input->expert_weight_gate;
    void **expert_weight_up = input->expert_weight_up;
    void **expert_weight_down = input->expert_weight_down;

    __FP16_TYPE *expert_weight_gate_transposed = static_cast<__FP16_TYPE *>(input->expert_weight_gate_transposed);
    __FP16_TYPE *expert_weight_up_transposed = static_cast<__FP16_TYPE *>(input->expert_weight_up_transposed);
    __FP16_TYPE *expert_weight_down_transposed = static_cast<__FP16_TYPE *>(input->expert_weight_down_transposed);

    __FP16_TYPE *shared_expert_weight_gate = static_cast<__FP16_TYPE *>(input->shared_expert_weight_gate);
    __FP16_TYPE *shared_expert_weight_up = static_cast<__FP16_TYPE *>(input->shared_expert_weight_up);
    __FP16_TYPE *shared_expert_weight_down = static_cast<__FP16_TYPE *>(input->shared_expert_weight_down);

    __FP16_TYPE *shared_expert_weight_gate_transposed =
        static_cast<__FP16_TYPE *>(input->shared_expert_weight_gate_transposed);
    __FP16_TYPE *shared_expert_weight_up_transposed =
        static_cast<__FP16_TYPE *>(input->shared_expert_weight_up_transposed);
    __FP16_TYPE *shared_expert_weight_down_transposed =
        static_cast<__FP16_TYPE *>(input->shared_expert_weight_down_transposed);

    float *shared_fc1_gate_splitk = static_cast<float *>(input->shared_fc1_gate_splitk);
    float *shared_fc1_up_splitk = static_cast<float *>(input->shared_fc1_up_splitk);
    __FP16_TYPE *shared_fc1_output = static_cast<__FP16_TYPE *>(input->shared_fc1_output);
    float *shared_fc2_splitk = static_cast<float *>(input->shared_fc2_splitk);
    __FP16_TYPE *shared_fc2_output = static_cast<__FP16_TYPE *>(input->shared_fc2_output);
    __FP16_TYPE *final_output = static_cast<__FP16_TYPE *>(input->final_output);

    // clang-format off

    // SEQ_LEN, D_HIDDEN, D_EXPERT, N_ROUTED_EXPERTS, N_EXPERT_PER_TOKEN, N_SHARED_EXPERTS, 
    // (Tuneables)       SPLITK_FACTOR{_ROUTED_FC1, _ROUTED_FC2, _SHARED_FC1, _SHARED_FC2}, SCHEDULER_BLOCK_COUNT, BM,    BN,     BK,   WARP_M,   WARP_N
    DISPATCH_MOE(512, 7168, 2048, 4, 4, 1,
                                            1,           1,           1,           1,                 768,         128,   128,    64,       2,      2);
    DISPATCH_MOE(512, 7168, 2048, 8, 4, 1,
                                            1,           1,           1,           1,                 768,         128,   128,    64,       2,      2);

    DISPATCH_MOE(512, 7168, 2048, 4, 4, 1,
                                            1,           1,           1,           1,                 768,         128,   128,    64,       2,      2);
    DISPATCH_MOE(1024, 7168, 2048, 8, 4, 1,
                                            1,           1,           1,           1,                 768,         128,   128,    64,       2,      2);
    DISPATCH_MOE(8192, 7168, 2048, 8, 4, 1,
                                            1,           1,           1,           1,                 768,         128,   128,    64,       2,      2);
    DISPATCH_MOE(2048, 7168, 2048, 32, 4, 1,
                                            1,           1,           1,           1,                 512,         128,   128,    64,       2,      2);
    DISPATCH_MOE(8192, 7168, 2048, 32, 4, 1,
                                            1,           1,           1,           1,                 512,         128,   128,    64,       2,      2);

    // clang-format on
    std::cerr << "Failed to dispatch" << std::endl;

    return 0;
}

void initialize_workspace(void **workspace) {

    static void *static_ws = nullptr;

    if (static_ws) {
        *workspace = static_ws;
        return;
    }

    constexpr size_t MAX_WORKSPACE_SIZE =
        MAX_SEQ_LEN * MAX_N_EXPERT_PER_TOKEN * sizeof(int) +                        // seq_experts_index
        MAX_SEQ_LEN * MAX_N_EXPERT_PER_TOKEN * sizeof(__FP16_TYPE) +                // seq_experts_softmax
        MAX_SEQ_LEN * MAX_N_EXPERT_PER_TOKEN * sizeof(int) +                        // source_rows
        MAX_SEQ_LEN * MAX_N_EXPERT_PER_TOKEN * sizeof(int) +                        // permuted_experts
        MAX_SEQ_LEN * MAX_N_EXPERT_PER_TOKEN * sizeof(int) +                        // permuted_rows
        (MAX_N_ROUTED_EXPERTS + 1) * sizeof(int) +                                  // first_token_offset
        MAX_SEQ_LEN * MAX_N_EXPERT_PER_TOKEN * MAX_D_HIDDEN * sizeof(__FP16_TYPE) + // expanded_seq_input
        MAX_SEQ_LEN * MAX_N_EXPERT_PER_TOKEN * sizeof(__FP16_TYPE) +                // expanded_seq_softmax
        MAX_SEQ_LEN * MAX_N_EXPERT_PER_TOKEN * sizeof(int) +                        // rev_permuted_rows
        MAX_SEQ_LEN * MAX_N_EXPERT_PER_TOKEN * MAX_D_EXPERT * MAX_SPLITK_FACTOR *
            sizeof(float) + // expanded_fc1_gate_splitk
        MAX_SEQ_LEN * MAX_N_EXPERT_PER_TOKEN * MAX_D_EXPERT * MAX_SPLITK_FACTOR *
            sizeof(float) +                                                         // expanded_fc1_up_splitk
        MAX_SEQ_LEN * MAX_N_EXPERT_PER_TOKEN * MAX_D_EXPERT * sizeof(__FP16_TYPE) + // expanded_fc1_output
        MAX_SEQ_LEN * MAX_N_EXPERT_PER_TOKEN * MAX_D_HIDDEN * MAX_SPLITK_FACTOR * sizeof(float) + // expanded_fc2_splitk
        MAX_SEQ_LEN * MAX_N_EXPERT_PER_TOKEN * MAX_D_HIDDEN * sizeof(__FP16_TYPE) +               // expanded_fc2_output
        MAX_SEQ_LEN * MAX_D_EXPERT * MAX_SPLITK_FACTOR * sizeof(float) + // shared_fc1_gate_splitk
        MAX_SEQ_LEN * MAX_D_EXPERT * MAX_SPLITK_FACTOR * sizeof(float) + // shared_fc1_up_splitk
        MAX_SEQ_LEN * MAX_D_EXPERT * sizeof(__FP16_TYPE) +               // shared_fc1_output
        MAX_SEQ_LEN * MAX_D_HIDDEN * MAX_SPLITK_FACTOR * sizeof(float) + // shared_fc2_splitk
        MAX_SEQ_LEN * MAX_D_HIDDEN * sizeof(__FP16_TYPE) +               // shared_fc2_output
        MAX_SEQ_LEN * MAX_D_HIDDEN * MAX_SPLITK_FACTOR * sizeof(float);  // final_output_splitk

    LIB_CALL(hipMalloc(&static_ws, MAX_WORKSPACE_SIZE));
    *workspace = static_ws;
}
}

void run_from_python(int seq_len, int batch_size, int d_hidden, int d_expert, int n_routed_experts,
                     int n_experts_per_token, int n_shared_experts, unsigned long long input_seq,
                     unsigned long long expert_scores, std::vector<unsigned long long> expert_weight_gate_p,
                     std::vector<unsigned long long> expert_weight_up_p,
                     std::vector<unsigned long long> expert_weight_down_p, unsigned long long shared_expert_weight_gate,
                     unsigned long long shared_expert_weight_up, unsigned long long shared_expert_weight_down,
                     unsigned long long router_weight, unsigned long long final_output) {
    void *workspace = nullptr;
    initialize_workspace(&workspace);
    initialize_env();
    MoEInputRaw input;
    input.params = {.batch_size = batch_size,
                    .seq_len = seq_len,
                    .d_hidden = d_hidden,
                    .d_expert = d_expert,
                    .n_routed_experts = n_routed_experts,
                    .n_experts_per_token = n_experts_per_token,
                    .n_shared_experts = n_shared_experts,
                    .seed = 0};
    input.expert_weight_gate = reinterpret_cast<void **>(expert_weight_gate_p.data());
    input.expert_weight_up = reinterpret_cast<void **>(expert_weight_up_p.data());
    input.expert_weight_down = reinterpret_cast<void **>(expert_weight_down_p.data());

    input.input_seq = reinterpret_cast<void *>(input_seq);

    input.expert_scores = reinterpret_cast<void *>(expert_scores);

    input.shared_expert_weight_gate = reinterpret_cast<void *>(shared_expert_weight_gate);
    input.shared_expert_weight_up = reinterpret_cast<void *>(shared_expert_weight_up);
    input.shared_expert_weight_down = reinterpret_cast<void *>(shared_expert_weight_down);

    input.router_weight = reinterpret_cast<void *>(router_weight);
    input.final_output = reinterpret_cast<void *>(final_output);

    int total_seq_len = batch_size * seq_len;

    // Allocate memory for intermediate variables
    size_t offset = 0;
    input.seq_experts_index = static_cast<char *>(workspace) + offset;
    offset += total_seq_len * n_experts_per_token * sizeof(int);

    input.seq_experts_softmax = static_cast<char *>(workspace) + offset;
    offset += total_seq_len * n_experts_per_token * sizeof(__FP16_TYPE);

    input.source_rows = static_cast<char *>(workspace) + offset;
    offset += total_seq_len * n_experts_per_token * sizeof(int);

    input.permuted_experts = static_cast<char *>(workspace) + offset;
    offset += total_seq_len * n_experts_per_token * sizeof(int);

    input.permuted_rows = static_cast<char *>(workspace) + offset;
    offset += total_seq_len * n_experts_per_token * sizeof(int);

    input.first_token_offset = static_cast<char *>(workspace) + offset;
    offset += (n_routed_experts + 1) * sizeof(int);

    input.expanded_seq_input = static_cast<char *>(workspace) + offset;
    offset += total_seq_len * n_experts_per_token * d_hidden * sizeof(__FP16_TYPE);

    input.expanded_seq_softmax = static_cast<char *>(workspace) + offset;
    offset += total_seq_len * n_experts_per_token * sizeof(__FP16_TYPE);

    input.rev_permuted_rows = static_cast<char *>(workspace) + offset;
    offset += total_seq_len * n_experts_per_token * sizeof(int);

    input.expanded_fc1_gate_splitk = static_cast<char *>(workspace) + offset;
    offset += total_seq_len * n_experts_per_token * d_expert * MAX_SPLITK_FACTOR * sizeof(float);

    input.expanded_fc1_up_splitk = static_cast<char *>(workspace) + offset;
    offset += total_seq_len * n_experts_per_token * d_expert * MAX_SPLITK_FACTOR * sizeof(float);

    input.expanded_fc1_output = static_cast<char *>(workspace) + offset;
    offset += total_seq_len * n_experts_per_token * d_expert * sizeof(__FP16_TYPE);

    input.expanded_fc2_splitk = static_cast<char *>(workspace) + offset;
    offset += total_seq_len * n_experts_per_token * d_hidden * MAX_SPLITK_FACTOR * sizeof(float);

    input.expanded_fc2_output = static_cast<char *>(workspace) + offset;
    offset += total_seq_len * n_experts_per_token * d_hidden * sizeof(__FP16_TYPE);

    input.shared_fc1_gate_splitk = static_cast<char *>(workspace) + offset;
    offset += total_seq_len * d_expert * MAX_SPLITK_FACTOR * sizeof(float);

    input.shared_fc1_up_splitk = static_cast<char *>(workspace) + offset;
    offset += total_seq_len * d_expert * MAX_SPLITK_FACTOR * sizeof(float);

    input.shared_fc1_output = static_cast<char *>(workspace) + offset;
    offset += total_seq_len * d_expert * sizeof(__FP16_TYPE);

    input.shared_fc2_splitk = static_cast<char *>(workspace) + offset;
    offset += total_seq_len * d_hidden * MAX_SPLITK_FACTOR * sizeof(float);

    input.shared_fc2_output = static_cast<char *>(workspace) + offset;
    offset += total_seq_len * d_hidden * sizeof(__FP16_TYPE);

    // Allocate space for final_output_splitk if needed
    input.final_output_splitk = static_cast<char *>(workspace) + offset;
    offset += total_seq_len * d_hidden * MAX_SPLITK_FACTOR * sizeof(float);

    auto stream = at::hip::getCurrentHIPStream().stream();

    // Run the actual computation
    run_topk(&input, stream0, nullptr);

    // Make sure all GPU operations have completed
}