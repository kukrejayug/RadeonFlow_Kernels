#include "moe.h"
#include "../../include/gpu_libs.h"
#include "../../include/gpu_types.h"
#include <hip/amd_detail/amd_warp_sync_functions.h>
#include "ck/utility/data_type.hpp"
#include <hipcub/hipcub.hpp>
#include <hipcub/block/block_radix_sort.hpp>
#include "moe_kernels.h"
#include "../../include/clangd_workaround.h"

DEVICE_CODE_BELOW

// BlockDim.x should be WAVE_SIZE

template <int VPT, int K, int NUM_EXPERTS, int ELEM_PER_LDG, int SEQ_LEN, bool FUSE_SOFTMAX, typename PER_LDG_TYPE>
__global__ void topk_kernel(__FP16_TYPE *score, int *seq_experts_index, __FP16_TYPE *seq_experts_softmax,
                            int *source_rows) {
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(ELEM_PER_LDG == 1 || ELEM_PER_LDG == 2 || ELEM_PER_LDG == 4 || ELEM_PER_LDG == 8);
    if constexpr (ELEM_PER_LDG == 1) {
        static_assert(std::is_same_v<PER_LDG_TYPE, ck::half_t>, "PER_LDG_TYPE must be float");
    } else if constexpr (ELEM_PER_LDG == 2) {
        static_assert(std::is_same_v<PER_LDG_TYPE, ck::half2_t>, "PER_LDG_TYPE must be float2");
    } else if constexpr (ELEM_PER_LDG == 4) {
        static_assert(std::is_same_v<PER_LDG_TYPE, ck::half4_t>, "PER_LDG_TYPE must be float4");
    } else if constexpr (ELEM_PER_LDG == 8) {
        static_assert(std::is_same_v<PER_LDG_TYPE, ck::half8_t>, "PER_LDG_TYPE must be float8");
    }
    static_assert(NUM_EXPERTS % ELEM_PER_LDG == 0, "NUM_EXPERTS must be divisible by ELEM_PER_LDG");

    static constexpr int THREADS_PER_ROW = NUM_EXPERTS / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELEM_PER_LDG;

    static_assert(VPT % ELEM_PER_LDG == 0, "VPT must be divisible by ELEM_PER_LDG");
    static_assert(WAVE_SIZE % THREADS_PER_ROW == 0, "WAVE_SIZE must be divisible by THREADS_PER_ROW");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WAVE_SIZE, "THREADS_PER_ROW can be at most warp size");

    static constexpr int ELEMS_PER_WARP = WAVE_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELEMS_PER_WARP / NUM_EXPERTS;
    static constexpr int ROWS_PER_BLOCK = WARPS_PER_BLOCK * ROWS_PER_WARP;

    static_assert(ELEMS_PER_WARP % NUM_EXPERTS == 0, "ELEMS_PER_WARP must be divisible by NUM_EXPERTS");

    int const block_base_row = blockIdx.x * ROWS_PER_BLOCK;
    int const warp_base_row = block_base_row + threadIdx.y * ROWS_PER_WARP;

    int const thread_row = warp_base_row + threadIdx.x / THREADS_PER_ROW;
    if (thread_row >= SEQ_LEN)
        return;
    __FP16_TYPE *thread_row_ptr = score + thread_row * NUM_EXPERTS;

    int const thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    int const first_elem_read_by_thread = thread_group_idx * ELEM_PER_LDG;
    __FP16_TYPE *thread_read_ptr = thread_row_ptr + first_elem_read_by_thread;
    __FP16_TYPE row_chunk[VPT];
    float row_chunk_high_prec[VPT];
    PER_LDG_TYPE *row_chunk_vec = reinterpret_cast<PER_LDG_TYPE *>(row_chunk);
    PER_LDG_TYPE *thread_read_vec = reinterpret_cast<PER_LDG_TYPE *>(thread_read_ptr);
    for (int ii = 0; ii < LDG_PER_THREAD; ii++) {
        row_chunk_vec[ii] = thread_read_vec[ii * THREADS_PER_ROW];
    }

    if constexpr (FUSE_SOFTMAX) {
        __FP16_TYPE thread_max = row_chunk[0];
#pragma unroll
        for (int ii = 1; ii < VPT; ii++) {
            thread_max = max(thread_max, row_chunk[ii]);
        }

#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
            thread_max = __hmax(thread_max, __shfl_xor_sync(0xffffffffffffffffull, thread_max, mask, THREADS_PER_ROW));
        }

        __FP16_TYPE row_max = thread_max;

        float row_sum = 0;
#pragma unroll
        for (int ii = 0; ii < VPT; ii++) {
            row_chunk_high_prec[ii] = expf((float)(row_chunk[ii]) - (float)row_max);
            row_sum += row_chunk_high_prec[ii];
        }

#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
            row_sum += __shfl_xor_sync(0xffffffffffffffff, row_sum, mask, THREADS_PER_ROW);
        }

        float const reciprocal_row_sum = 1.0f / row_sum;

#pragma unroll
        for (int ii = 0; ii < VPT; ii++) {
            row_chunk[ii] = (ck::half_t)(row_chunk_high_prec[ii] * reciprocal_row_sum);
        }
    }

    static constexpr int COLS_PER_GROUP_LDG = ELEM_PER_LDG * THREADS_PER_ROW;
    // start to find topk
    for (int k_idx = 0; k_idx < K; k_idx++) {
        // find local max
        __FP16_TYPE max_val = row_chunk[0];
        int expert = first_elem_read_by_thread;
#pragma unroll
        for (int ldg = 0, col = first_elem_read_by_thread; ldg < LDG_PER_THREAD; ldg++, col += COLS_PER_GROUP_LDG) {
#pragma unroll
            for (int ii = 0; ii < ELEM_PER_LDG; ii++) {
                __FP16_TYPE val = row_chunk[ldg * ELEM_PER_LDG + ii];
                if (val > max_val) {
                    max_val = val;
                    expert = col + ii;
                }
            }
        }

        // reduce max_val and expert

        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
            __FP16_TYPE other_max = __shfl_xor_sync(0xffffffffffffffff, max_val, mask, THREADS_PER_ROW);
            int other_expert = __shfl_xor_sync(0xffffffffffffffff, expert, mask, THREADS_PER_ROW);

            if (other_max > max_val || (max_val == other_max && other_expert < expert)) {
                max_val = other_max;
                expert = other_expert;
            }
        }

        // the "leader" of the row
        if (thread_group_idx == 0) {
            int const index = K * thread_row + k_idx;
            seq_experts_index[index] = expert;
            seq_experts_softmax[index] = max_val;
            // with source_rows, you can: source_rows[index] / SEQ_LEN to get k_idx,
            // source_rows[index] % SEQ_LEN to get thread_row
            source_rows[index] = k_idx * SEQ_LEN + thread_row;
        }

        // clear the picked value in current iteration
        if (k_idx + 1 < K) {
            int const ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
            int const thread_to_clear_in_group = (expert / ELEM_PER_LDG) % THREADS_PER_ROW;

            if (thread_group_idx == thread_to_clear_in_group) {
                int const offset_for_expert = expert % ELEM_PER_LDG;
                row_chunk[ldg_group_for_expert * ELEM_PER_LDG + offset_for_expert] = -100000.f;
            }
        }
    }
}

template <int ARR_LENGTH> __device__ int findLT(int *permuted_experts, int expert_idx) {
    int low = 0, high = ARR_LENGTH - 1, target_location = -1;
    while (low <= high) {
        int mid = (low + high) / 2;
        if (permuted_experts[mid] >= expert_idx) {
            high = mid - 1;
        } else {
            low = mid + 1;
            target_location = mid;
        }
    }
    return target_location + 1;
}

template <int SEQ_LEN, int N_ROUTED_EXPERTS, int N_EXPERT_PER_TOKEN>
__global__ __launch_bounds__(256) void compute_first_token_offset(int *permuted_experts, int *first_token_offset) {
    int const expert_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert_idx > N_ROUTED_EXPERTS)
        return;
    first_token_offset[expert_idx] = findLT<SEQ_LEN * N_EXPERT_PER_TOKEN>(permuted_experts, expert_idx);
}

template <int SEQ_LEN, int N_ROUTED_EXPERTS, int N_EXPERT_PER_TOKEN, int D_HIDDEN, int VPT, int ELEM_PER_LDG,
          typename PER_LDG_TYPE>
__global__ __launch_bounds__(768) void expand_input(const __FP16_TYPE *input_seq, __FP16_TYPE *expanded_input_seq,
                             __FP16_TYPE *seq_experts_softmax, __FP16_TYPE *expanded_experts_softmax,
                             int *permuted_rows, int *rev_permuted_rows) {
    static_assert(D_HIDDEN % VPT == 0);
    static_assert(VPT % ELEM_PER_LDG == 0);
    static_assert(D_HIDDEN % VPT == 0);
    int const THREADS_PER_ROW = D_HIDDEN / VPT;
    int const COLS_PER_GROUP_LDG = ELEM_PER_LDG * THREADS_PER_ROW;
    int const ROWS_PER_BLOCK = blockDim.x * blockDim.y / THREADS_PER_ROW;

    int const thread_id = blockDim.x * threadIdx.y + threadIdx.x;

    int const dest_row = ROWS_PER_BLOCK * blockIdx.x + thread_id / THREADS_PER_ROW;

    if (dest_row >= SEQ_LEN * N_EXPERT_PER_TOKEN)
        return;
    int const source_idx = permuted_rows[dest_row];

    int thread_id_in_group = thread_id % THREADS_PER_ROW;

    int const source_k_rank = source_idx / SEQ_LEN;
    int const source_row = source_idx % SEQ_LEN;

    if (thread_id_in_group == 0) {
        rev_permuted_rows[source_idx] = dest_row;
        expanded_experts_softmax[dest_row] = seq_experts_softmax[source_row * N_EXPERT_PER_TOKEN + source_k_rank];
    }

    const __FP16_TYPE *source_row_ptr = input_seq + source_row * D_HIDDEN;
    __FP16_TYPE *dest_row_ptr = expanded_input_seq + dest_row * D_HIDDEN;

    const PER_LDG_TYPE *source_row_vec = reinterpret_cast<const PER_LDG_TYPE *>(source_row_ptr);
    PER_LDG_TYPE *dest_row_vec = reinterpret_cast<PER_LDG_TYPE *>(dest_row_ptr);

    for (int ldg = thread_id_in_group; ldg < D_HIDDEN / ELEM_PER_LDG; ldg += THREADS_PER_ROW) {
        dest_row_vec[ldg] = source_row_vec[ldg];
    }
}

template <int SEQ_LEN, int N_ROUTED_EXPERTS, int N_EXPERT_PER_TOKEN, int D_HIDDEN, int VPT, int ELEM_PER_LDG,
          typename PER_LDG_TYPE>
__global__ __launch_bounds__(768) void reduce_output(const __FP16_TYPE expanded_fc2_output[][D_HIDDEN],
                              const __FP16_TYPE shared_fc2_output[][D_HIDDEN], __FP16_TYPE final_output[][D_HIDDEN],
                              const int *rev_permuted_rows, const int *permuted_experts,
                              const __FP16_TYPE *expanded_experts_softmax) {
    static_assert(D_HIDDEN % VPT == 0);
    static_assert(VPT % ELEM_PER_LDG == 0);
    static_assert(D_HIDDEN % VPT == 0);
    int const THREADS_PER_ROW = D_HIDDEN / VPT;
    int const COLS_PER_GROUP_LDG = ELEM_PER_LDG * THREADS_PER_ROW;
    int const ROWS_PER_BLOCK = blockDim.x * blockDim.y / THREADS_PER_ROW;

    int const thread_id = blockDim.x * threadIdx.y + threadIdx.x;

    int const dest_row = ROWS_PER_BLOCK * blockIdx.x + thread_id / THREADS_PER_ROW;

    PER_LDG_TYPE *dest_row_vec = reinterpret_cast<PER_LDG_TYPE *>(final_output[dest_row]);
    const PER_LDG_TYPE *shared_dest_row_vec = reinterpret_cast<const PER_LDG_TYPE *>(shared_fc2_output[dest_row]);

    if (dest_row >= SEQ_LEN)
        return;

    int thread_id_in_group = thread_id % THREADS_PER_ROW;

    for (int ldg = thread_id_in_group; ldg < D_HIDDEN / ELEM_PER_LDG; ldg += THREADS_PER_ROW) {
        PER_LDG_TYPE output = 0;
        for (int k_idx = 0; k_idx < N_EXPERT_PER_TOKEN; k_idx++) {
            int const source_row = rev_permuted_rows[k_idx * SEQ_LEN + dest_row];
            float const row_scale = expanded_experts_softmax[source_row];
            const PER_LDG_TYPE *expanded_row_vec =
                reinterpret_cast<const PER_LDG_TYPE *>(expanded_fc2_output[source_row]);
            output += *(expanded_row_vec + ldg) * row_scale;
        }
        dest_row_vec[ldg] = output + shared_dest_row_vec[ldg];
    }
}