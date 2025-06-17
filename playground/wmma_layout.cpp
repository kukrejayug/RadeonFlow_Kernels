#include "../include/gpu_libs.h"
#include "../include/gpu_types.h"
#include <cstdio>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/driver_types.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include "../include/clangd_workaround.h"

namespace wmma = rocwmma;

#define HIP_CALL(cmd) do { \
    hipError_t e = cmd; \
    if(e != hipSuccess) { \
        fprintf(stderr, "HIP error: %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
using fp8_type = __half;
using fp16_type = __half;
using fp32_type = float;

using FragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

__device__ void shfl_frag_layout(FragC& frag) { // Signature changed to modify frag
    int warp_id = threadIdx.x / WAVE_SIZE;
    int lane_id = threadIdx.x % WAVE_SIZE;
    if (true) {
        for (int k = 0; k < 7; ++k) {
            int lane_mask = k + 1;
            int uk = k + (k >= lane_id);
            int target_id = (lane_id & 7) ^ lane_mask;
            float reg = frag.x[target_id];
            frag.x[target_id] = __shfl_xor(reg, lane_mask, 8);
            if (false) {
                printf("lane_id=%d, target_id=%d, k=%d, uk=%d, lane_mask=%d\n", lane_id, lane_id ^ lane_mask, k, uk, lane_mask);
            }
        }
    }
}

__global__ void test_shfl_frag_layout(float *data1, float *data2) {
    int elements_per_thread = WMMA_M * WMMA_N / WAVE_SIZE;
    __syncthreads();
    FragC fragC;
    wmma::load_matrix_sync(fragC, data1, WMMA_N, wmma::mem_row_major);
    shfl_frag_layout(fragC);
    wmma::store_matrix_sync(data1, fragC, WMMA_N, wmma::mem_row_major);
    int lane_id = threadIdx.x % WAVE_SIZE;
    int new_idx = (((lane_id & 8) >> 3) | ((lane_id & 7) << 1) | (lane_id & 16));
    for (int k = 0; k < elements_per_thread; ++k) {
        data2[new_idx * elements_per_thread + k] = fragC.x[k];
    }
}

__global__ void testFragCThreadIdxLayout(float* thread_map, float* frag_idx_map, int* elem_map) {
    FragC fragC;
    wmma::fill_fragment(fragC, -1.0f);

    // Each thread fills its own fragC
    for (int i = 0; i < fragC.num_elements; i++) {
        fragC.x[i] = static_cast<float>(i);
    }
    wmma::store_matrix_sync(frag_idx_map, fragC, WMMA_N, wmma::mem_row_major);
    __syncthreads();
    wmma::fill_fragment(fragC, -1.0f);
    for (int i = 0; i < fragC.num_elements; i++) {
        fragC.x[i] = static_cast<float>(threadIdx.x);
    }
    wmma::store_matrix_sync(thread_map, fragC, WMMA_N, wmma::mem_row_major);

    // New addition: record the linear index of A[m][n] elements that each thread is responsible for
    for (int i = 0; i < fragC.num_elements; i++) {
        // Calculate linear index of global A
        int row = (i / WMMA_N);
        int col = (i % WMMA_N);
        int idx = row * WMMA_N + col;
        elem_map[threadIdx.x * fragC.num_elements + i] = idx;
    }
    __syncthreads();
}

HOST_CODE_BELOW

void validFragPos(const float thread_map[WMMA_M][WMMA_N], const float frag_idx_map[WMMA_M][WMMA_N]) {
    int success_count = 0;
    int total_count = WMMA_M * WMMA_N;

    for (int m = 0; m < WMMA_M; ++m) {
        for (int n = 0; n < WMMA_N; ++n) {
            int thread_idx = static_cast<int>(thread_map[m][n]);
            int frag_idx = static_cast<int>(frag_idx_map[m][n]);

            // Check if thread_idx or frag_idx are -1 (uninitialized/error)
            if (thread_idx < 0 || frag_idx < 0) {
                printf("Warning: Uninitialized value at [%d][%d]: thread_idx=%d, frag_idx=%d\n", m, n, thread_idx, frag_idx);
                continue; // Skip validation for this element
            }

            int m1 = -1, n1 = -1; // Initialize to invalid values

            if constexpr (WAVE_SIZE == 32) {
                // Original RDNA formula (assuming it was correct for Wave32)
                m1 = ((thread_idx & 16) >> 1) | (frag_idx & 7); // frag_idx is 0-7
                n1 = (thread_idx & 15);
            } else if constexpr (WAVE_SIZE == 64) {
                // Common Wave64 mapping for 16x16 Acc
                int thread_group = thread_idx / 4; // Group of 4 threads
                int group_row_start = (thread_group % 4) * 4; // Starting row for the 4x16 block
                int group_col = thread_group / 4; // Column block index

                int lane_in_group = thread_idx % 4; // 0, 1, 2, 3 within the group

                // Map frag_idx (0-3) to 2x2 coordinates within the thread's responsibility
                m1 = (thread_idx >> 4) * 4 + (frag_idx & 1) * 2 + (frag_idx >> 1); // Row calculation
                n1 = (thread_idx & 0xF); // Column calculation
            } else {
                // Default or error case
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
    // Construct matrix with known content
    float *h_mat = new float[WMMA_M * WMMA_N];
    for (int i = 0; i < WMMA_M * WMMA_N; ++i) h_mat[i] = static_cast<float>(i);

    float *d_mat;
    HIP_CALL(hipMalloc(&d_mat, WMMA_M * WMMA_N * sizeof(float)));
    HIP_CALL(hipMemcpy(d_mat, h_mat, WMMA_M * WMMA_N * sizeof(float), hipMemcpyHostToDevice));

    // Allocate two blocks of output device memory
    float *d_thread_map, *d_frag_idx_map, *h_thread_map, *h_frag_idx_map;
    size_t map_size = WMMA_M * WMMA_N * sizeof(float);
    HIP_CALL(hipMalloc(&d_thread_map, map_size));
    HIP_CALL(hipMalloc(&d_frag_idx_map, map_size));
    // Initialize device memory to a known value (-2.0f)
    float init_val = -2.0f;
    HIP_CALL(hipMemset(d_thread_map, *reinterpret_cast<int*>(&init_val), map_size)); // Use reinterpret_cast for hipMemset byte value
    HIP_CALL(hipMemset(d_frag_idx_map, *reinterpret_cast<int*>(&init_val), map_size));
    h_thread_map = new float[WMMA_M * WMMA_N];
    h_frag_idx_map = new float[WMMA_M * WMMA_N];

    // New addition: allocate elem_map
    int *d_elem_map, *h_elem_map;
    size_t elem_map_size = WAVE_SIZE * FragC::num_elements * sizeof(int); // WAVE_SIZE threads, each thread has fragC.num_elements
    HIP_CALL(hipMalloc(&d_elem_map, elem_map_size));
    // Optionally initialize d_elem_map too if needed, e.g., with -1
    HIP_CALL(hipMemset(d_elem_map, -1, elem_map_size));
    h_elem_map = new int[WAVE_SIZE * FragC::num_elements];

    // Launch experiment kernel
    hipLaunchKernelGGL(testFragCThreadIdxLayout, dim3(1), dim3(WAVE_SIZE), 0, 0, d_thread_map, d_frag_idx_map, d_elem_map);
    HIP_CALL(hipDeviceSynchronize());

    HIP_CALL(hipMemcpy(h_thread_map, d_thread_map, map_size, hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(h_frag_idx_map, d_frag_idx_map, map_size, hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(h_elem_map, d_elem_map, elem_map_size, hipMemcpyDeviceToHost));

    // Print responsibility relationship
    printf("=== threadIdx.x responsible for each element ===\n");
    for (int i = 0; i < WMMA_M * WMMA_N; ++i) {
        printf("%4d ", (int)h_thread_map[i]);
        if ((i + 1) % WMMA_N == 0) printf("\n");
    }
    printf("\n");

    printf("=== fragC.x index for each element ===\n");
    for (int i = 0; i < WMMA_M * WMMA_N; ++i) {
        printf("%4d ", (int)h_frag_idx_map[i]);
        if ((i + 1) % WMMA_N == 0) printf("\n");
    }
    printf("\n");

    // Correction: count which elements of A each thread is responsible for
    printf("=== Elements of A held by each thread (corrected) ===\n");
    for (int t = 0; t < WAVE_SIZE; ++t) {
        printf("threadIdx.x = %2d:", t);
        for (int i = 0; i < WMMA_M * WMMA_N; ++i) {
            if ((int)h_thread_map[i] == t) {
                int m = i / WMMA_N;
                int n = i % WMMA_N;
                printf(" A[%d][%d]", m, n);
            }
        }
        printf("\n");
    }
    printf("\n");

    // Call validFragPos function
    validFragPos(reinterpret_cast<const float(*)[WMMA_N]>(h_thread_map), 
    reinterpret_cast<const float(*)[WMMA_N]>(h_frag_idx_map));

    // --- Start: Test shfl_frag_layout ---
    printf("\n=== Testing shfl_frag_layout ===\n");
    float *d_data1, *d_data2;
    float *h_data1, *h_data2;
    size_t data_size = WMMA_M * WMMA_N * sizeof(float);

    HIP_CALL(hipMalloc(&d_data1, data_size));
    HIP_CALL(hipMalloc(&d_data2, data_size));
    h_data1 = new float[WMMA_M * WMMA_N];
    for (int m=0; m < WMMA_M; ++m) {
        for (int n=0; n < WMMA_N; ++n) {
            *reinterpret_cast<int*>(&h_data1[m * WMMA_N + n]) = m * WMMA_N + n;
        }
    }
    h_data2 = new float[WMMA_M * WMMA_N];
    HIP_CALL(hipMemcpy(d_data1, h_data1, WMMA_M * WMMA_N * sizeof(float), hipMemcpyHostToDevice));
    printf("Original Data (h_data1):\n");
    for (int i = 0; i < WMMA_M * WMMA_N; ++i) {
        printf("[%2d,%2d] ", (int)reinterpret_cast<int*>(h_data1)[i] / WMMA_N, (int)reinterpret_cast<int*>(h_data1)[i] % WMMA_N);
        if ((i + 1) % WMMA_N == 0) printf("\n");
    }
    // Launch the shuffle test kernel
    hipLaunchKernelGGL(test_shfl_frag_layout, dim3(1), dim3(WAVE_SIZE), 0, 0, d_data1, d_data2);
    HIP_CALL(hipDeviceSynchronize());

    // Copy results back to host
    HIP_CALL(hipMemcpy(h_data1, d_data1, data_size, hipMemcpyDeviceToHost));
    HIP_CALL(hipMemcpy(h_data2, d_data2, data_size, hipMemcpyDeviceToHost));

    // Print original data (optional)
    printf("Update Data (h_data1):\n");
    for (int i = 0; i < WMMA_M * WMMA_N; ++i) {
        printf("[%2d,%2d] ", (int)reinterpret_cast<int*>(h_data1)[i] / WMMA_N, (int)reinterpret_cast<int*>(h_data1)[i] % WMMA_N);
        if ((i + 1) % WMMA_N == 0) printf("\n");
    }
    printf("\n");

    // Print shuffled data
    printf("Write Data (h_data2):\n");
    for (int i = 0; i < WMMA_M * WMMA_N; ++i) {
        printf("[%2d,%2d] ", (int)reinterpret_cast<int*>(h_data2)[i] / WMMA_N, (int)reinterpret_cast<int*>(h_data2)[i] % WMMA_N);
        if ((i + 1) % WMMA_N == 0) printf("\n");
    }
    printf("\n");

    // Free memory for shuffle test
    HIP_CALL(hipFree(d_data1));
    HIP_CALL(hipFree(d_data2));
    delete[] h_data1;
    delete[] h_data2;
    // --- End: Test shfl_frag_layout ---

    HIP_CALL(hipFree(d_thread_map));
    HIP_CALL(hipFree(d_frag_idx_map));
    HIP_CALL(hipFree(d_elem_map));
    delete[] h_thread_map;
    delete[] h_frag_idx_map;
    delete[] h_elem_map;

    HIP_CALL(hipFree(d_mat));
    delete[] h_mat;
    return 0;
}
