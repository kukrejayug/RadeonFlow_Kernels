#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
// #include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>
#include <numeric>
#include <algorithm>
#include "../include/clangd_workaround.h"

#define HIP_CHECK(command) { \
    hipError_t status = command; \
    if (status != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s at %s:%d\n", hipGetErrorString(status), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 32;
constexpr int WAVE_SIZE = 32; // Added wavefront size definition
using fp32_type = float;
namespace wmma = rocwmma;
using FragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, fp32_type>;

constexpr int BLOCK_SIZE = 256;
constexpr int FRAGS_PER_BLOCK = BLOCK_SIZE; // 1 thread per fragment for simplicity

// --- Kernel: global memory -> fragment -> global memory ---
__global__ void wmma_gmem_read_write_bandwidth(const fp32_type* g_in, fp32_type* g_out, size_t num_frags) {
    size_t frag_idx = (blockIdx.x * blockDim.x + threadIdx.x) / WAVE_SIZE;
    if (frag_idx >= num_frags) return;

    const fp32_type* src = g_in + frag_idx * WMMA_M * WMMA_N;
    fp32_type* dst = g_out + frag_idx * WMMA_M * WMMA_N;

    FragC frag;
    wmma::load_matrix_sync(frag, src, WMMA_N, wmma::mem_row_major);
    for(int i=0; i<frag.num_elements; ++i) frag.x[i] += 1.0f;
    wmma::store_matrix_sync(dst, frag, WMMA_N, wmma::mem_row_major);
}

// --- Kernel: global -> shared -> WMMA -> shared -> global ---
__global__ void wmma_gmem_smem_read_write_bandwidth(const fp32_type* g_in, fp32_type* g_out, size_t num_frags) {
    extern __shared__ fp32_type s_mem[];
    size_t frag_idx = (blockIdx.x * blockDim.x + threadIdx.x) / WAVE_SIZE;
    if (frag_idx >= num_frags) return;

    if (threadIdx.x < WMMA_M * WMMA_N) {
        size_t base = blockIdx.x * WMMA_M * WMMA_N;
        s_mem[threadIdx.x] = g_in[base + threadIdx.x];
    }
    __syncthreads();

    FragC frag;
    wmma::load_matrix_sync(frag, s_mem, WMMA_N, wmma::mem_row_major);
    for(int i=0; i<frag.num_elements; ++i) frag.x[i] += 1.0f;

    if (threadIdx.x < WMMA_M * WMMA_N) {
        wmma::store_matrix_sync(s_mem, frag, WMMA_N, wmma::mem_row_major);
    }
    __syncthreads();

    // Write back to global memory
    if (threadIdx.x < WMMA_M * WMMA_N) {
        size_t base = blockIdx.x * WMMA_M * WMMA_N;
        g_out[base + threadIdx.x] = s_mem[threadIdx.x];
    }
}

HOST_CODE_BELOW

// --- Host Benchmark Function ---
template <typename KernelFunc>
float run_benchmark(KernelFunc kernel, const fp32_type* d_in, fp32_type* d_out, size_t num_frags,
                    int block_size, int smem_bytes, int num_iter, int warmup_iter)
{
    size_t block_frags = block_size / WAVE_SIZE;
    size_t num_blocks = (num_frags + block_frags - 1) / block_frags;
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    for (int i = 0; i < warmup_iter; ++i) {
        kernel<<<num_blocks, block_size, smem_bytes>>>(d_in, d_out, num_frags);
    }
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(start, 0));
    for (int i = 0; i < num_iter; ++i) {
        kernel<<<num_blocks, block_size, smem_bytes>>>(d_in, d_out, num_frags);
    }
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    return ms / num_iter;
}

int main() {
    const size_t TOTAL_MB = 256;
    const size_t FRAG_SIZE = WMMA_M * WMMA_N * sizeof(fp32_type);
    const size_t TOTAL_BYTES = TOTAL_MB * 1024 * 1024;
    const size_t NUM_FRAGS = TOTAL_BYTES / FRAG_SIZE;
    const int NUM_ITER = 100, WARMUP_ITER = 10;
    const size_t NUM_ELEMENTS_PER_FRAG = WMMA_M * WMMA_N;
    const size_t TOTAL_ELEMENTS = NUM_FRAGS * NUM_ELEMENTS_PER_FRAG;

    printf("WMMA FragC Bandwidth Microbenchmark\n");
    printf("  Data Size: %zu MB (%zu fragments)\n", TOTAL_MB, NUM_FRAGS);
    printf("  Fragment: %dx%d fp32, %zu bytes each\n", WMMA_M, WMMA_N, FRAG_SIZE);
    printf("  Block Size: %d, Wave Size: %d, Iter: %d, Warmup: %d\n", BLOCK_SIZE, WAVE_SIZE, NUM_ITER, WARMUP_ITER);

    fp32_type* h_in = (fp32_type*)malloc(TOTAL_BYTES);
    for(size_t i = 0; i < TOTAL_ELEMENTS; ++i) {
        h_in[i] = static_cast<fp32_type>(i % 100) * 0.1f;
    }

    fp32_type *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, TOTAL_BYTES));
    HIP_CHECK(hipMalloc(&d_out, TOTAL_BYTES));
    HIP_CHECK(hipMemcpy(d_in, h_in, TOTAL_BYTES, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_out, 0, TOTAL_BYTES));

    printf("\n%-24s | %-10s | %-15s | %s\n", "Kernel", "Time (ms)", "Bandwidth (GB/s)", "Notes");
    printf("----------------------------------------------------------------------------------\n");

    double gbps;
    size_t num_blocks = (NUM_FRAGS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Unified test: global -> wmma -> global
    float ms = run_benchmark(wmma_gmem_read_write_bandwidth, d_in, d_out, NUM_FRAGS, BLOCK_SIZE, 0, NUM_ITER, WARMUP_ITER);
    gbps = (double)(2 * TOTAL_BYTES) / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    printf("%-24s | %10.4f | %15.2f | Read+Write Global\n", "Global->WMMA->Global", ms, gbps);

    // New test: global -> shared -> WMMA -> shared -> global
    size_t smem_bytes = WMMA_M * WMMA_N * sizeof(fp32_type);
    ms = run_benchmark(wmma_gmem_smem_read_write_bandwidth, d_in, d_out, NUM_FRAGS, BLOCK_SIZE, smem_bytes, NUM_ITER, WARMUP_ITER);
    gbps = (double)(2 * TOTAL_BYTES) / (ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    printf("%-24s | %10.4f | %15.2f | Read+Write Global (via Shared)\n", "Global->Shared->WMMA->Shared->Global", ms, gbps);

    printf("----------------------------------------------------------------------------------\n");

    free(h_in);
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
