#include "../include/gpu_libs.h"
#include "../include/gpu_types.h"
#include <cstdio>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/driver_types.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include "../include/clangd_workaround.h"

constexpr int WMMA_M = 32;
constexpr int WMMA_N = 32;
constexpr int WMMA_K = 32;
using data_type = half;
using FragC = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, data_type, wmma::row_major>;

__global__ void write_thread_idx(data_type thread_map[WMMA_M][WMMA_N]) {
    thread_map[threadIdx.x % WMMA_M][threadIdx.x / WMMA_M] = 777.0f;
    FragC fragC;
    printf("threadIdx.x = %d\n", threadIdx.x);
    wmma::fill_fragment(fragC, half{-1.0f});
    __syncthreads();
    for (int i = 0; i < fragC.num_elements; i++) {
        fragC.x[i] = static_cast<data_type>(threadIdx.x);
    }
    __syncthreads();
    wmma::store_matrix_sync(reinterpret_cast<data_type*>(thread_map), fragC, WMMA_N);
}

__global__ void write_frag_idx(data_type frag_idx_map[WMMA_M][WMMA_N]) {
    FragC fragC;
    wmma::fill_fragment(fragC, data_type{-1.0f});
    __syncthreads();
    for (int i = 0; i < FragC::num_elements; i++) {
        fragC.x[i] = static_cast<data_type>(i);
    }
    __syncthreads();
    wmma::store_matrix_sync(reinterpret_cast<data_type*>(frag_idx_map), fragC, WMMA_N);
}

#define HIP_CALL(cmd) do { \
    hipError_t e = cmd; \
    if(e != hipSuccess) { \
        fprintf(stderr, "HIP error: %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

HOST_CODE_BELOW

void validFragPos(const data_type thread_map[WMMA_M][WMMA_N], const data_type frag_idx_map[WMMA_M][WMMA_N]) {
    int success_count = 0;
    int total_count = WMMA_M * WMMA_N;

    for (int m = 0; m < WMMA_M; ++m) {
        for (int n = 0; n < WMMA_N; ++n) {
            int thread_idx = static_cast<int>(thread_map[m][n]);
            int frag_idx = static_cast<int>(frag_idx_map[m][n]);

            if (thread_idx < 0 || frag_idx < 0) {
                printf("Warning: Uninitialized value at [%d][%d]: thread_idx=%d, frag_idx=%d\n", m, n, thread_idx, frag_idx);
                continue;
            }

            int m1 = -1, n1 = -1;

            if constexpr (WAVE_SIZE == 32) {
                m1 = ((thread_idx & 16) >> 1) | (frag_idx & 7);
                n1 = (thread_idx & 15);
            } else if constexpr (WAVE_SIZE == 64 && WMMA_M == 32 && WMMA_N == 32) {
                // m1 = ((frag_idx >> 2) << 3) | ((thread_idx >> 5) << 2) | (frag_idx & 3);
                // n1 = thread_idx & 31;
                n1 = (frag_idx / 8 * 16 + thread_idx / 32 * 8 + frag_idx % 8); // col
                m1 = (thread_idx % 32); // row
                // m = 8 * (f // 4) + 4 * (t // 32) + (f % 4)
                //     n = t % 32
            } else if constexpr (WAVE_SIZE == 64) {
                // accumulator
                m1 = ((thread_idx & 48) >> 2) | frag_idx;
                n1 = (thread_idx & 15);
            
            } else {
                printf("Error: Unsupported WAVE_SIZE %d in validFragPos\n", WAVE_SIZE);
            }

            if (m1 == m && n1 == n) {
                success_count++;
            } else {
                if (m1 != -1 || n1 != -1) {
                    printf("Error: thread_map[%d][%d] = %d, frag_idx_map[%d][%d] = %d -> Calculated m1=%d, n1=%d (Expected m=%d, n=%d)\n",
                           m, n, thread_idx, m, n, frag_idx, m1, n1, m, n);
                }
            }
        }
    }

    float success_ratio = static_cast<float>(success_count) / total_count * 100.0f;
    printf("Validation Success ratio: %.2f%% (%d / %d)\n", success_ratio, success_count, total_count);
}

int main() {
    std::cout << "WAVE_SIZE1 = " << WAVE_SIZE << std::endl;
    data_type *d_thread_map, *d_frag_idx_map;
    data_type *h_thread_map, *h_frag_idx_map;
    h_thread_map = new data_type[WMMA_M * WMMA_N];
    h_frag_idx_map = new data_type[WMMA_M * WMMA_N];
    HIP_CALL(hipMalloc(&d_thread_map, WMMA_M * WMMA_N * sizeof(data_type)));
    HIP_CALL(hipMalloc(&d_frag_idx_map, WMMA_M * WMMA_N * sizeof(data_type)));
    HIP_CALL(hipMemset(d_thread_map, 0, WMMA_M * WMMA_N * sizeof(data_type)));
    HIP_CALL(hipMemset(d_frag_idx_map, 0, WMMA_M * WMMA_N * sizeof(data_type)));
    HIP_CALL(hipDeviceSynchronize());
    write_thread_idx<<<1, WAVE_SIZE>>>(reinterpret_cast<data_type(*)[WMMA_N]>(d_thread_map));
    write_frag_idx<<<1, WAVE_SIZE>>>(reinterpret_cast<data_type(*)[WMMA_N]>(d_frag_idx_map));
    HIP_CALL(hipDeviceSynchronize());
    HIP_CALL(hipMemcpy(h_thread_map, d_thread_map, WMMA_M * WMMA_N * sizeof(data_type), hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(h_frag_idx_map, d_frag_idx_map, WMMA_M * WMMA_N * sizeof(data_type), hipMemcpyDeviceToHost));
    HIP_CALL(hipDeviceSynchronize());
    printf("============== Thread Map ==============\n");
    for (int i = 0; i < WMMA_M * WMMA_N; ++i) {
        printf("%4d ", static_cast<int>(h_thread_map[i]));
        if ((i + 1) % WMMA_N == 0) printf("\n");
    }
    printf("============== Frag Index Map ==============\n");
    for (int i = 0; i < WMMA_M * WMMA_N; ++i) {
        printf("%4d ", static_cast<int>(h_frag_idx_map[i]));
        if ((i + 1) % WMMA_N == 0) printf("\n");
    }

    validFragPos(reinterpret_cast<const data_type(*)[WMMA_N]>(h_thread_map), 
                 reinterpret_cast<const data_type(*)[WMMA_N]>(h_frag_idx_map));

    HIP_CALL(hipFree(d_thread_map));
    HIP_CALL(hipFree(d_frag_idx_map));
    delete[] h_thread_map;
    delete[] h_frag_idx_map;

    return 0;
}