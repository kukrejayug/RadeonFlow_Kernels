#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#include "../include/gpu_libs.h"
#include "../include/gpu_types.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <hip/hip_runtime.h>
#include "../include/clangd_workaround.h"

#include <rocwmma/rocwmma.hpp>
constexpr int M = 512;
constexpr int N = 512;
constexpr int K = 512;

// constexpr int M = 2048;
// constexpr int N = 2048;
// constexpr int K = 2048;

// constexpr int M = 4096;
// constexpr int N = 4096;
// constexpr int K = 4096;

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_NUM = BLOCK_SIZE / 32;
constexpr int WARP_M = 4;
constexpr int WARP_N = 2;
static_assert(WARP_M * WARP_N == WARP_NUM, "WARP_M * WARP_N must equal WARP_NUM");
constexpr int QUANT_SIZE = 128;
constexpr int QM = M;
constexpr int QN = N / QUANT_SIZE;
constexpr int QK = K / QUANT_SIZE;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 32;
constexpr int BK = 32;
constexpr int BM = 256;
constexpr int BN = 128;

constexpr int FRAG_K = BK / WMMA_K;
// static_assert(FRAG_K == 2);
constexpr int FRAG_M = BM / WMMA_M / WARP_M;
static_assert(FRAG_M == 4);
constexpr int FRAG_N = BN / WMMA_N / WARP_N;
static_assert(FRAG_N == 4);
constexpr int APAD = 0;
constexpr int BPAD = 0; 

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// using fp8_type  = half;
// using fp16_type = half;
// using fp32_type = half;
using fp8_type  = __FP8_TYPE;
using fp16_type = __hip_bfloat16;
using fp32_type = float;
namespace wmma = rocwmma;
using FragA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, fp8_type, wmma::col_major>;
using FragB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, fp8_type, wmma::row_major>;
using FragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, fp32_type>;
using FragOut = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half>;


template <typename data_type, int BATCH_SIZE> 
__device__ inline void read_batch(data_type *dst, const data_type *src) {
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

template <typename data_type, int BATCH_SIZE> 
__device__ inline void zero_batch(data_type *dst) {
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


template <typename data_type, int DST_Y, int DST_X, int SRC_Y, int SRC_X, int BLOCK_DIM, int BATCH_SIZE>
__device__ inline void copy_to_shared_v2(data_type dst[DST_Y][DST_X], const data_type src[SRC_Y][SRC_X],
                                         const int begin_x, const int begin_y) {
  static_assert(BATCH_SIZE > 0);
  /**
    Consider (SRC_X % DST_X == 0) && (SRC_Y % DST_Y == 0)
    Step 1:
      [   ][***][   ][   ]
      [   ][   ][   ][   ]
      [   ][   ][   ][   ]
      [   ][   ][   ][   ]
    Step 2:
      [   ][   ][   ][   ]
      [   ][***][   ][   ]
      [   ][   ][   ][   ]
      [   ][   ][   ][   ]
  */
  static_assert((SRC_X % BATCH_SIZE == 0) && (SRC_Y % BATCH_SIZE == 0));
  static_assert((DST_X % BATCH_SIZE == 0) && (DST_Y % BATCH_SIZE == 0));
  static_assert(BATCH_SIZE <= DST_X && DST_X % BATCH_SIZE == 0);
  const int begin_idx = threadIdx.x * BATCH_SIZE;
  const constexpr int total_elements = DST_X * DST_Y;
  const constexpr int elements_per_step = BLOCK_DIM * BATCH_SIZE;
  #pragma unroll
  for (int k = begin_idx; k < total_elements; k += elements_per_step) {
    int l_kx = k % DST_X;
    int l_ky = k / DST_X;
    int g_kx = l_kx + begin_x;
    int g_ky = l_ky + begin_y;
    auto *dst_flatten = &dst[l_ky][l_kx];
    const auto *src_flatten = &src[g_ky][g_kx];
    read_batch<data_type, BATCH_SIZE>(dst_flatten, src_flatten);
    if (((SRC_X % DST_X == 0) || (g_kx < SRC_X)) && ((SRC_Y % DST_Y == 0) || (g_ky < SRC_Y))) {
      const auto *src_flatten = &src[g_ky][g_kx];
      read_batch<data_type, BATCH_SIZE>(dst_flatten, src_flatten);
    } else {
      zero_batch<data_type, BATCH_SIZE>(dst_flatten);
    }
  }
}

__device__ void compute_matrix_multiplication(
    fp8_type s_a[BK][BM], 
    fp8_type s_b[BK][BN], 
    FragA frag_a[FRAG_K][FRAG_M], 
    FragB frag_b[FRAG_K][FRAG_N], 
    FragC frag_c[FRAG_M][FRAG_N], 
    FragC frag_r[FRAG_M][FRAG_N],
    float s_s[BM],
    int comp_c_frag_m, 
    int comp_c_frag_n
) {
    #pragma unroll
    for (int i = 0; i < FRAG_M; i++) {
        #pragma unroll
        for (int j = 0; j < FRAG_N; j++) {
            wmma::fill_fragment(frag_c[i][j], 0.0F);
        }
    }

    // Use loops to load fragments instead of hardcoded calls
    for (int k = 0; k < FRAG_K; ++k) {
        #pragma unroll
        for (int i = 0; i < FRAG_M; ++i) {
            int s_a_row = k * WMMA_K;
            int s_a_col = (comp_c_frag_m * FRAG_M + i) * WMMA_M;
            wmma::load_matrix_sync(frag_a[k][i], &s_a[s_a_row][s_a_col], BM + APAD);
        }
        #pragma unroll
        for (int j = 0; j < FRAG_N; ++j) {
            int s_b_row = k * WMMA_K;
            int s_b_col = (comp_c_frag_n * FRAG_N + j) * WMMA_N;
            wmma::load_matrix_sync(frag_b[k][j], &s_b[s_b_row][s_b_col], BN + BPAD);
        }
    }

    // Modified multiplication pattern for better tensor core utilization
    #pragma unroll
    for (int i = 0; i < FRAG_M; i++) {
        #pragma unroll
        for (int j = 0; j < FRAG_N; j++) {
            #pragma unroll
            for (int k = 0; k < FRAG_K; ++k) { // Loop over K dimension fragments
                wmma::mma_sync(frag_c[i][j], frag_a[k][i], frag_b[k][j], frag_c[i][j]);
            }
        }
    }
    
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < 8; k++) {
        for (int i = 0; i < FRAG_M; i++) {
            int m = ((threadIdx.x % 32) >> 2) | ((k & 2) << 2) + (comp_c_frag_m * WARP_M + i) * WMMA_M;
            float scale = s_s[m];
            #pragma unroll
            for (int j = 0; j < FRAG_N; j++) {
                frag_r[i][j].x[k] += (fp32_type)scale * (fp32_type)frag_c[i][j].x[k];
            }
        }
    }
}

namespace wmmax {

__device__ inline void shfl_store_matrix(__hip_bfloat16 *data, FragC &frag) {
    int lane_id = threadIdx.x % 32;
    #pragma unroll
    for (int k = 0; k < 7; ++k) {
        int lane_mask = k + 1;
        int uk = k + (k >= lane_id);
        int target_id = (lane_id & 7) ^ lane_mask;
        float reg = frag.x[target_id];
        frag.x[target_id] = __shfl_xor(reg, lane_mask, 8);
    }
    __hip_bfloat16 val[8];
    #pragma unroll
    for (int k = 0; k < 8; k += 1) {
        val[k] = __hip_bfloat16(frag.x[k]);
    }

    static_assert(sizeof(val) == sizeof(ulong2));
    int start_pos = (((lane_id & 8) >> 3) | ((lane_id & 7) << 1) | (lane_id & 16)) * 8;
    int start_pos2 = (start_pos / WMMA_N) * N + (start_pos % WMMA_N);
    *reinterpret_cast<ulong2*>(&data[start_pos2]) = *reinterpret_cast<ulong2*>(val);
    
}
} // namespace wmmax

__device__ inline void store_result_to_global_memory(
    fp16_type c[M][N], 
    FragC frag_r[4][4], 
    int bx, 
    int by, 
    int comp_c_frag_m, 
    int comp_c_frag_n
) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int frag_m = comp_c_frag_m * 4 + i;
            int frag_n = comp_c_frag_n * 4 + j;
            int row = by * BM + frag_m * WMMA_M;
            int col = bx * BN + frag_n * WMMA_N;
            
            fp16_type *c_ptr = &c[row][col];
            // wmmax::shfl_store_matrix(c_ptr, frag_r[i][j]);
            // wmma::store_matrix_sync(c_ptr, frag_r[i][j], N, wmma::mem_row_major);
            FragOut frag_out;
            for (int k=0;k<8;++k) {
                // __buildin_memcpy(&frag_out.x[k], &frag_r[i][j].x[k], sizeof(half));
                __hip_bfloat16 reg = frag_r[i][j].x[k];
                frag_out.x[k] = *reinterpret_cast<half*>(&reg);
            }
            wmma::store_matrix_sync(reinterpret_cast<half*>(c_ptr), frag_out, N, wmma::mem_row_major);
        }
    }
}

__global__ void gemm_kernel(
    const fp8_type a[K][M],
    const fp8_type b[K][N],
    fp16_type c[M][N],
    const float sa[QK][QM],
    const float sb[QK][QN]
) {
    __shared__ fp8_type s_a[BK][BM];
    __shared__ fp8_type s_b[BK][BN];
    __shared__ float    s_s[BM];
    FragA frag_a[FRAG_K][FRAG_M];
    FragB frag_b[FRAG_K][FRAG_N];
    FragC frag_c[FRAG_M][FRAG_N];
    FragC frag_r[FRAG_M][FRAG_N];


    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;
    int wid = tid / 32;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            wmma::fill_fragment(frag_r[i][j], 0.0f);
        }
    }

    int load_a_smem_m = (tid >> 2) << 1;
    int load_a_smem_k = (tid &  3) << 3;
    int load_b_smem_k = (tid >> 5) << 2;
    int load_b_smem_n = (tid & 31) << 3;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_smem_k, K);
    int load_b_gmem_addr = OFFSET(load_b_smem_k, load_b_gmem_n, N);

    int comp_c_frag_m = wid / WARP_N; // Correct mapping based on WARP_M=4, WARP_N=2
    int comp_c_frag_n = wid % WARP_N; // Correct mapping based on WARP_M=4, WARP_N=2
    
    for (int bk = 0; bk < K / BK; bk++) {
        const int g_ax = bk * BK;
        const int g_ay = blockIdx.y * BM;
        const int g_bx = blockIdx.x * BN;
        const int g_by = bk * BK;
        copy_to_shared_v2<fp8_type, BK, BM, K, M, BLOCK_SIZE, 2 * sizeof(float4) / sizeof(fp8_type)>(
            s_a, a, g_ay, g_ax);
        copy_to_shared_v2<fp8_type, BK, BN, K, N, BLOCK_SIZE, 2 * sizeof(float4) / sizeof(fp8_type)>(
            s_b, b, g_bx, g_by);
            
        // Load scaling factors into shared memory
        for (int m = threadIdx.x; m < BM; m += BLOCK_SIZE) {
            s_s[m] = sa[g_ax / QUANT_SIZE][g_ay + m] * sb[g_by / QUANT_SIZE][g_bx / QUANT_SIZE];
            // printf("s_s[%d] = %f\n", m, s_s[m]);
        }
        __syncthreads();
        
        load_a_gmem_addr += BK;
        load_b_gmem_addr += BK * N;
        
        compute_matrix_multiplication(s_a, s_b, frag_a, frag_b, frag_c, frag_r, s_s, comp_c_frag_m, comp_c_frag_n);

        __syncthreads();
    }
    


    store_result_to_global_memory(c, frag_r, bx, by, comp_c_frag_m, comp_c_frag_n);
    
}

HOST_CODE_BELOW

#define CHECK_HIP_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != hipSuccess) {
        fprintf(stderr, "HIP Runtime Error at: %s:%d\n", file, line);
        fprintf(stderr, "%s %s\n", hipGetErrorString(err), func);
        exit(1);
    }
}

// Define a macro to check HIP errors
#define HIP_CALL(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

void cpu_gemm(const fp8_type a[K][M], const fp8_type b[K][N], fp16_type c[M][N], 
              const float sa[QK][QM], const float sb[QK][QN]) {
    float (*rc)[N] = new float[M][N];
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            rc[m][n] = 0.0f;
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n < N; ++n) {
                float scale = sa[k/QUANT_SIZE][m] * sb[k/QUANT_SIZE][n/QUANT_SIZE];
                // float scale = 1.0f;
                rc[m][n] += ((float)scale * (float)a[k][m] * (float)b[k][n]);
            }
        }
    }
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            c[m][n] = rc[m][n];
        }
    }
    delete[] rc;
}

int main() {
    fp8_type (*h_a)[M] = new fp8_type[K][M];
    fp8_type (*h_b)[N] = new fp8_type[K][N];
    fp16_type (*h_c)[N] = new fp16_type[M][N];
    fp16_type (*h_c_ref)[N] = new fp16_type[M][N];
    
    float (*h_sa)[QM] = new float[QK][QM];
    float (*h_sb)[QN] = new float[QK][QN];

    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < M; ++j) {
            h_a[i][j] = (fp8_type)((rand() % 10000) / 10000.0f);
        }
    }
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            h_b[i][j] = (fp8_type)((rand() % 10000) / 10000.0f);
            // if (i < 3 && j < 3) {
            //     h_b[i][j] = (fp8_type)(fp8_type)((rand() % 10000) / 10000.0f);;
            // } else {
            //     h_b[i][j] = 0;
            // }
            
        }
    }
    
    for (int i = 0; i < QK; ++i) {
        for (int j = 0; j < QM; ++j) {
            // h_sa[i][j] = rand() % 1000 / 1000.0f - 2;
            h_sa[i][j] = 1.0f; 
        }
    }
    for (int i = 0; i < QK; ++i) {
        for (int j = 0; j < QN; ++j) {
            // h_sb[i][j] = rand() % 1000 / 1000.0f - 2;
            h_sb[i][j] = 1.0f;
        }
    }

    fp8_type (*d_a)[M];
    fp8_type (*d_b)[N];
    fp16_type (*d_c)[N];
    float (*d_sa)[QM];
    float (*d_sb)[QN];
    
    CHECK_HIP_ERROR(hipMalloc(&d_a, K * M * sizeof(fp8_type)));
    CHECK_HIP_ERROR(hipMalloc(&d_b, K * N * sizeof(fp8_type)));
    CHECK_HIP_ERROR(hipMalloc(&d_c, M * N * sizeof(fp16_type)));
    CHECK_HIP_ERROR(hipMalloc(&d_sa, QK * QM * sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_sb, QK * QN * sizeof(float)));

    CHECK_HIP_ERROR(hipMemcpy(d_a, h_a, K * M * sizeof(fp8_type), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_b, h_b, K * N * sizeof(fp8_type), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_sa, h_sa, QK * QM * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_sb, h_sb, QK * QN * sizeof(float), hipMemcpyHostToDevice));

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(BLOCK_SIZE);
    
    if (BLOCK_SIZE % 32 != 0) {
        printf("Error: Block size must be a multiple of warp size (32)\n");
        return 1;
    }
    
    int deviceId;
    HIP_CALL(hipGetDevice(&deviceId));
    hipDeviceProp_t deviceProp;
    HIP_CALL(hipGetDeviceProperties(&deviceProp, deviceId));
    
    if (deviceProp.major < 7) {
        printf("Error: This kernel requires a GPU with compute capability 7.0 or higher\n");
        return 1;
    }
    
    printf("Running GEMM kernel with grid(%d,%d), block(%d)...\n", 
           grid.x, grid.y, block.x);
    
    // Query and print kernel and device information
    printf("Querying kernel and device information...\n");

    // Get device properties
    HIP_CALL(hipGetDeviceProperties(&deviceProp, deviceId));
    printf("Device Name: %s\n", deviceProp.name);
    printf("Total Global Memory: %lu bytes\n", deviceProp.totalGlobalMem);
    printf("Shared Memory per Block: %lu bytes\n", deviceProp.sharedMemPerBlock);
    printf("Registers per Block: %d\n", deviceProp.regsPerBlock);
    printf("Warp Size: %d\n", deviceProp.warpSize);
    printf("Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max Threads per Multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf("Number of Multiprocessors: %d\n", deviceProp.multiProcessorCount);

    // Query kernel attributes
    hipFuncAttributes funcAttr;
    HIP_CALL(hipFuncGetAttributes(&funcAttr, (const void*)gemm_kernel));
    printf("Kernel Attributes:\n");
    printf("  Shared Memory Size: %lu bytes\n", funcAttr.sharedSizeBytes);
    printf("  Number of Registers: %d\n", funcAttr.numRegs);
    printf("  Max Threads per Block: %d\n", funcAttr.maxThreadsPerBlock);
    printf("  Local Memory Size: %lu bytes\n", funcAttr.localSizeBytes);

    CHECK_HIP_ERROR(hipMemset(d_c, 0, M * N * sizeof(fp16_type)));
    
    printf("Performing warmup runs...\n");
    gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, d_sa, d_sb);
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, d_sa, d_sb);
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    
    hipEvent_t start, stop;
    HIP_CALL(hipEventCreate(&start));
    HIP_CALL(hipEventCreate(&stop));

    CHECK_HIP_ERROR(hipDeviceSynchronize());
    HIP_CALL(hipEventRecord(start));
    
    // Launch kernel
    printf("Launching kernel...\n");
    gemm_kernel<<<grid, block>>>(d_a, d_b, d_c, d_sa, d_sb);
    
    HIP_CALL(hipEventRecord(stop));
    
    HIP_CALL(hipEventSynchronize(stop));
    float milliseconds = 0;
    HIP_CALL(hipEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %f ms\n", milliseconds);
    
    CHECK_HIP_ERROR(hipGetLastError());
    
    double operations = 2.0 * M * N * K;
    double seconds = milliseconds / 1000.0;
    double tflops = (operations / seconds) / 1e12;
    printf("GPU Performance: %.2f TFLOPS\n", tflops);
    
    CHECK_HIP_ERROR(hipMemcpy(h_c, d_c, M * N * sizeof(fp16_type), hipMemcpyHostToHost));

    printf("Computing reference result on CPU...\n");
    cpu_gemm(h_a, h_b, h_c_ref, h_sa, h_sb);

    printf("Verifying results...\n");
    int errors = 0;
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    struct ErrorInfo {
        int row, col;
        float gpu_val, cpu_val, abs_diff, rel_diff;
    };
    ErrorInfo first_10_errors[10];
    ErrorInfo max_10_errors[10] = {};

    // Add a configurable variable for the number of errors to output
    int max_errors_to_output = 10; // You can modify this value as needed

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float gpu_val = (float)h_c[i][j];
            float cpu_val = (float)h_c_ref[i][j];
            float abs_diff;
            float rel_diff;

            if (std::isnan(gpu_val) || std::isnan(cpu_val)) {
                abs_diff = INFINITY;
                rel_diff = INFINITY;
            } else {
                abs_diff = abs(gpu_val - cpu_val);
                rel_diff = abs_diff / (abs(cpu_val) + FLT_EPSILON);
            }

            // Track max absolute and relative differences
            max_abs_diff = fmaxf(max_abs_diff, abs_diff);
            max_rel_diff = fmaxf(max_rel_diff, rel_diff);

            // Record first 10 errors
            if (errors < max_errors_to_output && (rel_diff > 1e-2 || abs_diff > 1e-3)) {
                first_10_errors[errors] = {i, j, gpu_val, cpu_val, abs_diff, rel_diff};
            }

            // Track top 10 largest errors
            if (rel_diff > 1e-2 || abs_diff > 1e-3) {
                errors++;
                for (int k = 0; k < max_errors_to_output; ++k) {
                    if (abs_diff > max_10_errors[k].abs_diff) {
                        for (int l = max_errors_to_output - 1; l > k; --l) {
                            max_10_errors[l] = max_10_errors[l - 1];
                        }
                        max_10_errors[k] = {i, j, gpu_val, cpu_val, abs_diff, rel_diff};
                        break;
                    }
                }
            }
        }
    }

    // Print first 10 errors
    printf("First %d errors:\n", max_errors_to_output);
    for (int i = 0; i < fmin(errors, max_errors_to_output); ++i) {
        printf("Error at [%d, %d]: GPU=%f, CPU=%f, AbsDiff=%f, RelDiff=%f\n",
               first_10_errors[i].row, first_10_errors[i].col,
               first_10_errors[i].gpu_val, first_10_errors[i].cpu_val,
               first_10_errors[i].abs_diff, first_10_errors[i].rel_diff);
    }

    // Print top 10 largest errors
    printf("Top %d largest errors:\n", max_errors_to_output);
    for (int i = 0; i < max_errors_to_output && max_10_errors[i].abs_diff > 0; ++i) {
        printf("Error at [%d, %d]: GPU=%f, CPU=%f, AbsDiff=%f, RelDiff=%f\n",
               max_10_errors[i].row, max_10_errors[i].col,
               max_10_errors[i].gpu_val, max_10_errors[i].cpu_val,
               max_10_errors[i].abs_diff, max_10_errors[i].rel_diff);
    }

    printf("Max abs_diff: %f, Max rel_diff: %f\n", max_abs_diff, max_rel_diff);
    if (errors == 0) {
        printf("Test PASSED!\n");
    } else {
        printf("Test FAILED with %d errors\n", errors);
    }

    double flops = 2.0 * M * N * K;
    double gflops = (flops * 1e-9) / (milliseconds * 1e-3);
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_c_ref;
    delete[] h_sa;
    delete[] h_sb;
    HIP_CALL(hipFree(d_a));
    HIP_CALL(hipFree(d_b));
    HIP_CALL(hipFree(d_c));
    HIP_CALL(hipFree(d_sa));
    HIP_CALL(hipFree(d_sb));
    HIP_CALL(hipEventDestroy(start));
    HIP_CALL(hipEventDestroy(stop));

    return 0;
}