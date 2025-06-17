#pragma once

#include "../../include/gpu_libs.h"
#include "../../include/gpu_types.h"

constexpr int WARPS_PER_BLOCK = 16;

template <typename in_data_type,
          typename acc_data_type, // Accumulator type (e.g., float)
          typename out_data_type, // Output type (e.g., __hip_bfloat16)
          int K,                  // Matrix dimensions
          int BM, int BN, int BK, // Tile dimensions
          int BLOCK_SIZE,         // Block size
          int WARP_M, int WARP_N  // Warp dimensions
          >
__device__ void moe_gemm_kernel(const in_data_type *a_tile, const in_data_type *b_tile, out_data_type *c_tile,
                                const int M_tile, const int N_tile);

template <int VPT, int K, int NUM_EXPERTS, int ELEM_PER_LDG, int SEQ_LEN, bool FUSE_SOFTMAX, typename PER_LDG_TYPE>
__global__ void topk_kernel(__FP16_TYPE *score, int *seq_experts_index, __FP16_TYPE *seq_experts_softmax,
                            int *source_rows);

template <int SEQ_LEN, int N_ROUTED_EXPERTS, int N_EXPERT_PER_TOKEN>
__global__ void compute_first_token_offset(int *permuted_experts, int *first_token_offset);

template <int SEQ_LEN, int N_ROUTED_EXPERTS, int N_EXPERT_PER_TOKEN, int D_HIDDEN, int VPT, int ELEM_PER_LDG,
          typename PER_LDG_TYPE>

__global__ void expand_input(const __FP16_TYPE *input_seq, __FP16_TYPE *expanded_input_seq,
                             __FP16_TYPE *seq_experts_softmax, __FP16_TYPE *expanded_experts_softmax,
                             int *permuted_rows, int *rev_permuted_rows);