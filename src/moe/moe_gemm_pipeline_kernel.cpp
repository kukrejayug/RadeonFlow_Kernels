#include "../../include/clangd_workaround.h"
#include "../../include/gpu_libs.h"
#include "../../include/gpu_types.h"
#include <ck/utility/data_type.hpp>
#include <ck/utility/amd_buffer_addressing.hpp>
#include "../utils/arithmetic.h"
#include "moe_kernels.h"
#include <cfloat>
#include <cstdlib>
using int32x4_t = ck::int32x4_t;

DEVICE_CODE_BELOW

template <typename data_type, int BATCH_SIZE> __device__ inline void read_batch(data_type *dst, const data_type *src) {
    if constexpr ((sizeof(data_type) * BATCH_SIZE) == 2 * sizeof(ulong4)) {
        *(reinterpret_cast<ulong4 *>(dst) + 0) = *(reinterpret_cast<const ulong4 *>(src) + 0);
        *(reinterpret_cast<ulong4 *>(dst) + 1) = *(reinterpret_cast<const ulong4 *>(src) + 1);
    } else if constexpr ((sizeof(data_type) * BATCH_SIZE) == sizeof(ulong4)) {
        *reinterpret_cast<ulong4 *>(dst) = *reinterpret_cast<const ulong4 *>(src);
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(ulong2)) {
        *reinterpret_cast<ulong2 *>(dst) = *reinterpret_cast<const ulong2 *>(src);
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(ulong1)) {
        *reinterpret_cast<ulong1 *>(dst) = *reinterpret_cast<const ulong1 *>(src);
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(uint1)) {
        *reinterpret_cast<uint1 *>(dst) = *reinterpret_cast<const uint1 *>(src);
    } else {
#pragma unroll
        for (int b = 0; b < BATCH_SIZE; ++b) {
            dst[b] = src[b];
        }
    }
}

template <typename data_type, int BATCH_SIZE> __device__ inline void zero_batch(data_type *dst) {
    if constexpr ((sizeof(data_type) * BATCH_SIZE) == sizeof(ulong4)) {
        *reinterpret_cast<ulong4 *>(dst) = ulong4{};
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(ulong2)) {
        *reinterpret_cast<ulong2 *>(dst) = ulong2{};
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(ulong1)) {
        *reinterpret_cast<ulong1 *>(dst) = ulong1{};
    } else if constexpr (sizeof(data_type) * BATCH_SIZE == sizeof(uint1)) {
        *reinterpret_cast<uint *>(dst) = uint{};
    } else {
#pragma unroll
        for (int b = 0; b < BATCH_SIZE; ++b) {
            dst[b] = 0;
        }
    }
}

__device__ inline __half SILU(__half x) { return x * (__half(1.0f) / (__half(1.0f) + hexp(-x))); }

__device__ __inline__ void wave_barrier() { asm volatile("s_barrier" : : : "memory"); }

__device__ void inline async_buffer_load_fence(int cnt = 0) {
    asm volatile(R"(
        s_waitcnt vmcnt(%0); \n\t
    )" ::"n"(cnt)
                 : "memory");
}

template <typename data_type, int DST_Y, int DST_X, int PAD_X, int SRC_STRIDE, int BLOCK_SIZE, int BATCH_SIZE>
__device__ inline void load_input_moe_sync(data_type dst[DST_Y][DST_X + PAD_X], const data_type *src_begin,
                                           const int SRC_Y) {
    // we assume SRC_X always equals to DST_X

    static_assert(BATCH_SIZE > 0);
    static_assert((SRC_STRIDE % DST_X == 0));
    static_assert((DST_X % BATCH_SIZE == 0));
    static_assert(BATCH_SIZE <= DST_X && DST_X % BATCH_SIZE == 0);
    const int begin_idx = threadIdx.x * BATCH_SIZE;
    const constexpr int total_elements = DST_X * DST_Y;
    const constexpr int elements_per_step = BLOCK_SIZE * BATCH_SIZE;
// FIXME: loop unrolling
#pragma unroll
    for (int k = begin_idx; k < total_elements; k += elements_per_step) {
        int l_kx = k % DST_X;
        int l_ky = k / DST_X;
        auto *dst_flatten = &dst[l_ky][l_kx];

        if (l_ky < SRC_Y) {
            read_batch<data_type, BATCH_SIZE>(dst_flatten, src_begin + l_ky * SRC_STRIDE + l_kx);
        } else {
            zero_batch<data_type, BATCH_SIZE>(dst_flatten);
        }
    }
}

template <typename LoadVec, int VecSize, int BK, typename data_type, int STRIDE, int BLOCK_SIZE>
__device__ inline void gds2reg(LoadVec regs[VecSize], ck::int32x4_t gds_resource, int gds_offset) {
    constexpr int BATCH_SIZE = sizeof(LoadVec) / sizeof(data_type);
    const constexpr int elements_per_step = BLOCK_SIZE * BATCH_SIZE;
#pragma unroll
    for (int k = 0; k < VecSize; ++k) {
        int idx = (k * BLOCK_SIZE + threadIdx.x) * BATCH_SIZE;
        int i = idx / BK;
        int j = idx % BK;
        int v_offset = (i * STRIDE + j) * sizeof(data_type);
        using scalar_t = typename ck::scalar_type<LoadVec>::type;
        constexpr int vector_size = ck::scalar_type<LoadVec>::vector_size;
        regs[k] = ck::amd_buffer_load_impl<scalar_t, vector_size, ck::AmdBufferCoherenceEnum::DefaultCoherence>(
            gds_resource, v_offset, gds_offset);
    }
}

template <typename data_type, int BK, int PAD, typename LoadVec, int VecSize, int BLOCK_SIZE>
__device__ inline void reg2lds(data_type (*lds)[BK + PAD], LoadVec regs[VecSize]) {
    constexpr int BATCH_SIZE = sizeof(LoadVec) / sizeof(data_type);
    constexpr int load_vec_per_row = exact_div<BK * sizeof(data_type), sizeof(LoadVec)>();
    for (int k = 0; k < VecSize; ++k) {
        int idx = (k * BLOCK_SIZE + threadIdx.x) * BATCH_SIZE;
        int i = idx / BK;
        int j = idx % BK;
        *reinterpret_cast<LoadVec *>(&lds[i][j]) = regs[k];
    }
}

__device__ ck::int32x4_t inline make_wave_buffer_resource(const void *ptr, uint32_t size = 0xffffffff) {
    int32x4_t res;

    // Pack the 64-bit pointer into two 32-bit integers
    uint64_t ptr_val = reinterpret_cast<uint64_t>(ptr);
    res.x = static_cast<uint32_t>(ptr_val);
    res.y = static_cast<uint32_t>(ptr_val >> 32);

    // Set buffer size and format
    res.z = size;       // Buffer size in bytes
    res.w = 0x00020000; // hardcoded for gfx942

    res.x = __builtin_amdgcn_readfirstlane(res.x);
    res.y = __builtin_amdgcn_readfirstlane(res.y);
    res.z = __builtin_amdgcn_readfirstlane(res.z);
    res.w = __builtin_amdgcn_readfirstlane(res.w);
    return res;
}

DEVICE_CODE_BELOW
template <typename in_data_type, typename out_data_type, int N, int K, int A_STRIDE, int B_STRIDE, int C_STRIDE, int BM,
          int BN, int BK, int FRAG_M, int FRAG_N, int FRAG_K, int WMMA_M, int WMMA_N, int WMMA_K, int WARP_M,
          int WARP_N, int BLOCK_SIZE, int A_PAD = 8, int B_PAD = 8>
__device__ inline void gemm_pipeline_kernel(const in_data_type a[][A_STRIDE], const in_data_type b[][B_STRIDE],
                                            out_data_type c[][C_STRIDE], int TILE_M) {

    using FragA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, in_data_type, wmma::row_major>;
    using FragB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, in_data_type, wmma::col_major>;
    using FragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;
    using FragOut = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, out_data_type>;
    using LoadVec = ck::half8_t;

    __shared__ in_data_type s_a[BM][BK + A_PAD];
    __shared__ in_data_type s_b[BN][BK + B_PAD];
    LoadVec regs_a[exact_div<BM * BK * sizeof(in_data_type), BLOCK_SIZE * sizeof(LoadVec)>()];
    LoadVec regs_b[exact_div<BN * BK * sizeof(in_data_type), BLOCK_SIZE * sizeof(LoadVec)>()];
    FragA frag_a[FRAG_K][FRAG_M];
    FragB frag_b[FRAG_K][FRAG_N];
    FragC frag_r[FRAG_M][FRAG_N];
    auto wmma_load = [](const in_data_type s_a[BM][BK + A_PAD], const in_data_type s_b[BN][BK + B_PAD],
                        FragA frag_a[FRAG_K][FRAG_M], FragB frag_b[FRAG_K][FRAG_N]) {
        const int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
        const int comp_c_frag_m = __builtin_amdgcn_readfirstlane(warp_id / WARP_N);
        const int comp_c_frag_n = __builtin_amdgcn_readfirstlane(warp_id % WARP_N);
#pragma unroll
        for (int i = 0; i < FRAG_M; ++i) {
#pragma unroll
            for (int k = 0; k < FRAG_K; ++k) {
                int s_a_col = k * WMMA_K;
                int s_a_row = (comp_c_frag_m * FRAG_M + i) * WMMA_M;
                wmma::load_matrix_sync(frag_a[k][i], &s_a[s_a_row][s_a_col], BK + A_PAD);
            }
        }
#pragma unroll
        for (int j = 0; j < FRAG_N; ++j) {
#pragma unroll
            for (int k = 0; k < FRAG_K; ++k) {
                int s_b_col = k * WMMA_K;
                int s_b_row = (comp_c_frag_n * FRAG_N + j) * WMMA_N;
                wmma::load_matrix_sync(frag_b[k][j], &s_b[s_b_row][s_b_col], BK + B_PAD);
            }
        }
    };

    auto wmma_compute = [](FragA frag_a[FRAG_K][FRAG_M], FragB frag_b[FRAG_K][FRAG_N], FragC frag_r[FRAG_M][FRAG_N]) {
#pragma unroll
        for (int i = 0; i < FRAG_M; ++i) {
#pragma unroll
            for (int j = 0; j < FRAG_N; ++j) {
#pragma unroll
                for (int k = 0; k < FRAG_K; ++k) {
                    wmma::mma_sync(frag_r[i][j], frag_a[k][i], frag_b[k][j], frag_r[i][j]);
                }
            }
        }
    };

    auto wmma_store = [](out_data_type *c, ck::int32x4_t c_src, const FragC frag_r[FRAG_M][FRAG_N], int TILE_M) {
        const int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
        const int lane_id = __lane_id();
        const int comp_c_frag_m = warp_id / WARP_N;
        const int comp_c_frag_n = warp_id % WARP_N;
        // #pragma unroll
        for (int i = 0; i < FRAG_M; ++i) {
            // #pragma unroll
            for (int j = 0; j < FRAG_N; ++j) {
                int frag_m = comp_c_frag_m * FRAG_M + i;
                int frag_n = comp_c_frag_n * FRAG_N + j;
                int row = frag_m * WMMA_M;
                int col = frag_n * WMMA_N;
                // #pragma unroll
                // for (int k = 0; k < FragC::num_elements; k++) {
                //     // ck::half4_t data = {
                //     //     static_cast<ck::half_t>(frag_r[i][j].x[k + 0]),
                //     //     static_cast<ck::half_t>(frag_r[i][j].x[k + 1]),
                //     //     static_cast<ck::half_t>(frag_r[i][j].x[k + 2]),
                //     //     static_cast<ck::half_t>(frag_r[i][j].x[k + 3])
                //     // };
                //     // using scalar_t = typename ck::scalar_type<decltype(data)>::type;
                //     // constexpr int vector_size = ck::scalar_type<decltype(data)>::vector_size;
                //     int m = ((k >> 2) << 3) | ((lane_id >> 5) << 2) | (k & 3);
                //     int n = lane_id & 31;
                //     int threads_offset = (row * C_STRIDE + col + m * C_STRIDE + n) * sizeof(out_data_type);
                //     int wave_offset = 0;
                //     ck::amd_buffer_store_impl<ck::half_t, 1, ck::AmdBufferCoherenceEnum::DefaultCoherence>(
                //         static_cast<ck::half_t>(frag_r[i][j].x[k]), c_src, threads_offset, wave_offset);
                // }
                if (row < TILE_M) {
                    FragOut frag_out;
                    static_assert(FragOut::num_elements == FragC::num_elements);
                    for (int k = 0; k < FragOut::num_elements; ++k) {
                        frag_out.x[k] = frag_r[i][j].x[k];
                    }
                    if (row + WMMA_M <= TILE_M) {
                        wmma::store_matrix_sync(c + row * C_STRIDE + col, frag_out, C_STRIDE, wmma::mem_row_major);
                    } else {
                        for (int k = 0; k < FragC::num_elements; k++) {
                            int lane_id = threadIdx.x % WAVE_SIZE;
                            int m = ((k >> 2) << 3) | ((lane_id >> 5) << 2) | (k & 3);
                            int n = lane_id & 31;
                            if (row + m < TILE_M) {
                                c[row * C_STRIDE + col + m * C_STRIDE + n] = frag_r[i][j].x[k];
                            }
                        }
                    }
                }
            }
        }
    };

    constexpr int LOAD_BATCH_SIZE = (2 * sizeof(float4) / sizeof(in_data_type)) > 0
                                        ? (2 * sizeof(float4) / sizeof(in_data_type))
                                        : 1; // Ensure batch size > 0
    static_assert(LOAD_BATCH_SIZE > 0, "LOAD_BATCH_SIZE must be positive");

    for (int i = 0; i < FRAG_M; ++i) {
        for (int j = 0; j < FRAG_N; ++j) {
            wmma::fill_fragment(frag_r[i][j], {});
        }
    }

    auto a_src = make_wave_buffer_resource(a, TILE_M * A_STRIDE * sizeof(in_data_type));
    auto b_src = make_wave_buffer_resource(b, BN * B_STRIDE * sizeof(in_data_type));
    auto c_src = make_wave_buffer_resource(c, TILE_M * C_STRIDE * sizeof(out_data_type));

    int bk = __builtin_amdgcn_readfirstlane(0);
    int block_k = __builtin_amdgcn_readfirstlane(bk * BK);
    // global2lds_async(s_a[bk % 2], s_b[bk % 2], &a[0][block_k], b[block_k], TILE_M);
    __syncthreads();
    gds2reg<LoadVec, sizeof(regs_a) / sizeof(LoadVec), BK, in_data_type, A_STRIDE, BLOCK_SIZE>(
        regs_a, a_src, block_k * sizeof(in_data_type));
    gds2reg<LoadVec, sizeof(regs_b) / sizeof(LoadVec), BK, in_data_type, B_STRIDE, BLOCK_SIZE>(
        regs_b, b_src, block_k * sizeof(in_data_type));
    reg2lds<in_data_type, BK, A_PAD, LoadVec, sizeof(regs_a) / sizeof(LoadVec), BLOCK_SIZE>(s_a, regs_a);
    reg2lds<in_data_type, BK, B_PAD, LoadVec, sizeof(regs_b) / sizeof(LoadVec), BLOCK_SIZE>(s_b, regs_b);
    for (bk = __builtin_amdgcn_readfirstlane(1); bk < exact_div<K, BK>(); bk += 1) {
        block_k = bk * BK;
        gds2reg<LoadVec, sizeof(regs_a) / sizeof(LoadVec), BK, in_data_type, A_STRIDE, BLOCK_SIZE>(
            regs_a, a_src, block_k * sizeof(in_data_type));
        gds2reg<LoadVec, sizeof(regs_b) / sizeof(LoadVec), BK, in_data_type, B_STRIDE, BLOCK_SIZE>(
            regs_b, b_src, block_k * sizeof(in_data_type));
        __syncthreads();
        wmma_load(s_a, s_b, frag_a, frag_b);
        wmma_compute(frag_a, frag_b, frag_r);
        __syncthreads();
        reg2lds<in_data_type, BK, A_PAD, LoadVec, sizeof(regs_a) / sizeof(LoadVec), BLOCK_SIZE>(s_a, regs_a);
        reg2lds<in_data_type, BK, B_PAD, LoadVec, sizeof(regs_b) / sizeof(LoadVec), BLOCK_SIZE>(s_b, regs_b);
    }
    __syncthreads();
    wmma_load(s_a, s_b, frag_a, frag_b);
    wmma_compute(frag_a, frag_b, frag_r);
    wmma_store(reinterpret_cast<out_data_type *>(c), c_src, frag_r, TILE_M);
}
template <bool FUSE_DUAL_B, bool SLICED_A, typename in_data_type, typename out_data_type, int N_ROUTED_EXPERTS, int M_,
          int N, int K, int BM, int BN, int BK, int BLOCK_COUNT, int BLOCK_SIZE, int SPLITK_FACTOR, int WARP_M,
          int WARP_N>
__launch_bounds__(BLOCK_SIZE) __global__
    void moe_gemm_scheduler_entry(const in_data_type *a, in_data_type *B0, in_data_type *B1, const in_data_type *b0,
                                  const in_data_type *b1, out_data_type *ab0, out_data_type *ab1,
                                  const int first_token_offset[N_ROUTED_EXPERTS + 1]) {

    constexpr int WMMA_M = 32, WMMA_N = 32, WMMA_K = 32;

    static_assert(BLOCK_SIZE == WARP_M * WARP_N * WAVE_SIZE);
    constexpr int FRAG_M = exact_div<BM, WMMA_M * WARP_M>();
    constexpr int FRAG_N = exact_div<BN, WMMA_N * WARP_N>();
    constexpr int FRAG_K = exact_div<BK, WMMA_K>();

    static_assert(K % BK == 0);
    static_assert(N % BN == 0);
    static_assert(K % BN == 0);

    static_assert(K % SPLITK_FACTOR == 0);

    constexpr int TILE_K = K / SPLITK_FACTOR;

    static_assert(TILE_K % BK == 0);

    const int block_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    int tile_count = 0;

    auto compute_row_slice = [&](int a_row_begin, int a_row_end, const in_data_type *b0, const in_data_type *b1) {

#pragma unroll
        for (int splitk_idx = 0; splitk_idx < SPLITK_FACTOR; splitk_idx++) {
            for (int row = a_row_begin; row < a_row_end; row += BM) {
                for (int col = 0; col < N; col += BN) {
                    if ((tile_count++) % BLOCK_COUNT == block_id) {
                        int M_tile = min(BM, a_row_end - row);
                        int N_tile = BN;
                        const in_data_type *a_tile = a + row * K + splitk_idx * TILE_K;
                        const in_data_type *b0_tile = b0 + col * K + splitk_idx * TILE_K;
                        const in_data_type *b1_tile = nullptr;
                        out_data_type *ab0_tile = ab0 + splitk_idx * M_ * N + row * N + col;
                        out_data_type *ab1_tile = nullptr;
                        if constexpr (FUSE_DUAL_B) {
                            b1_tile = b1 + col * K + splitk_idx * TILE_K;
                            ab1_tile = ab1 + splitk_idx * M_ * N + row * N + col;
                        }

                        gemm_pipeline_kernel<in_data_type, out_data_type, N, TILE_K, K, K, N, BM, BN, BK, FRAG_M,
                                             FRAG_N, FRAG_K, WMMA_M, WMMA_N, WMMA_K, WARP_M, WARP_N, BLOCK_SIZE>(
                            reinterpret_cast<const in_data_type(*)[K]>(a_tile),
                            reinterpret_cast<const in_data_type(*)[K]>(b0_tile),
                            reinterpret_cast<out_data_type(*)[N]>(ab0_tile), M_tile);

                        if constexpr (FUSE_DUAL_B) {
                            gemm_pipeline_kernel<in_data_type, out_data_type, N, TILE_K, K, K, N, BM, BN, BK, FRAG_M,
                                                 FRAG_N, FRAG_K, WMMA_M, WMMA_N, WMMA_K, WARP_M, WARP_N, BLOCK_SIZE>(
                                reinterpret_cast<const in_data_type(*)[K]>(a_tile),
                                reinterpret_cast<const in_data_type(*)[K]>(b1_tile),
                                reinterpret_cast<out_data_type(*)[N]>(ab1_tile), M_tile);
                        }
                    }
                }
            }
        }
    };

    if constexpr (SLICED_A) {
#pragma unroll
        for (int expert_idx = 0; expert_idx < N_ROUTED_EXPERTS; expert_idx++) {
            const in_data_type *b0 = reinterpret_cast<const in_data_type *>(B0 + expert_idx * N * K);
            const in_data_type *b1;
            if constexpr (FUSE_DUAL_B) {
                b1 = reinterpret_cast<const in_data_type *>(B1 + expert_idx * N * K);
            }

            int token_offset_begin = first_token_offset[expert_idx];
            int token_offset_end = first_token_offset[expert_idx + 1];

            compute_row_slice(token_offset_begin, token_offset_end, b0, b1);
        }
    } else {
        compute_row_slice(0, M_, b0, b1);
    }
}

template <bool FUSE_DUAL_B, typename data_type, int M, int N, int ELEM_PER_LDG, typename PER_LDG_TYPE>
__global__ __launch_bounds__(512) void moe_gemm_reduce_ab(const data_type *ab0, const data_type *ab1, data_type *out) {

    int const thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    int const thread_pos = thread_id * ELEM_PER_LDG;

    data_type *dest_ptr = out + thread_pos;
    PER_LDG_TYPE *dest_vec = reinterpret_cast<PER_LDG_TYPE *>(dest_ptr);

    if (thread_pos >= M * N) {
        return;
    }

    if constexpr (FUSE_DUAL_B) {
#pragma unroll
        for (int i = 0; i < ELEM_PER_LDG; i++) {
            dest_ptr[i] = SILU(ab0[thread_pos + i]) * ab1[thread_pos + i];
        }
    } else {

#pragma unroll
        for (int i = 0; i < ELEM_PER_LDG; i++) {
            dest_ptr[i] = ab0[thread_pos + i];
        }

        // dest_vec[0] = *reinterpret_cast<const PER_LDG_TYPE *>(ab0 + thread_pos);
    }
}
