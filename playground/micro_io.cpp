#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h> // For float1, float2, float4
#include <assert.h>
#include <numeric>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip> // For std::setw
#include "../include/clangd_workaround.h"

#define HIP_CHECK(command) { \
    hipError_t status = command; \
    if (status != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s at %s:%d\n", hipGetErrorString(status), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// --- Kernel Implementations ---

// --- Contiguous Access Kernels ---
// Each thread copies VECS_PER_THREAD contiguous VecType elements: global->reg->global
template <typename VecType, int BLOCK_SIZE, int FLOATS_PER_BLOCK>
__global__ void copy_contiguous_rw(const float* g_in, float* g_out, size_t N) {
    constexpr int VEC_SIZE = sizeof(VecType) / sizeof(float);
    constexpr int VECS_PER_BLOCK = FLOATS_PER_BLOCK / VEC_SIZE;
    constexpr int VECS_PER_THREAD = VECS_PER_BLOCK / BLOCK_SIZE;

    size_t block_start_float_idx = (size_t)blockIdx.x * FLOATS_PER_BLOCK;
    const VecType* g_in_vec = reinterpret_cast<const VecType*>(g_in);
    VecType* g_out_vec = reinterpret_cast<VecType*>(g_out);

    size_t thread_start_g_idx = block_start_float_idx / VEC_SIZE + threadIdx.x * VECS_PER_THREAD;

    VecType reg[VECS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < VECS_PER_THREAD; ++i) {
        size_t idx = thread_start_g_idx + i;
        if (block_start_float_idx + (threadIdx.x * VECS_PER_THREAD + i) * VEC_SIZE < N) {
            reg[i] = g_in_vec[idx];
        }
    }

    #pragma unroll
    for (int i = 0; i < VECS_PER_THREAD; ++i) {
        size_t idx = thread_start_g_idx + i;
        if (block_start_float_idx + (threadIdx.x * VECS_PER_THREAD + i) * VEC_SIZE < N) {
            g_out_vec[idx] = reg[i];
        }
    }
}

// --- Interleaved Access Kernels ---
// Each thread copies VECS_PER_THREAD VecType elements, interleaved by BLOCK_SIZE * VEC_SIZE floats: global->reg->global
template <typename VecType, int BLOCK_SIZE, int FLOATS_PER_BLOCK>
__global__ void copy_interleaved_rw(const float* g_in, float* g_out, size_t N) {
    constexpr int VEC_SIZE = sizeof(VecType) / sizeof(float);
    constexpr int VECS_PER_BLOCK = FLOATS_PER_BLOCK / VEC_SIZE;
    constexpr int FLOATS_PER_THREAD = FLOATS_PER_BLOCK / BLOCK_SIZE;
    constexpr int VECS_PER_THREAD = FLOATS_PER_THREAD / VEC_SIZE;

    size_t block_start_float_idx = (size_t)blockIdx.x * FLOATS_PER_BLOCK;
    const VecType* g_in_vec = reinterpret_cast<const VecType*>(g_in);
    VecType* g_out_vec = reinterpret_cast<VecType*>(g_out);

    VecType reg[VECS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < VECS_PER_THREAD; ++i) {
        size_t float_offset_in_block = threadIdx.x * VEC_SIZE + i * BLOCK_SIZE * VEC_SIZE;
        size_t g_vec_idx = (block_start_float_idx + float_offset_in_block) / VEC_SIZE;
        if (block_start_float_idx + float_offset_in_block < N) {
            reg[i] = g_in_vec[g_vec_idx];
        }
    }

    #pragma unroll
    for (int i = 0; i < VECS_PER_THREAD; ++i) {
        size_t float_offset_in_block = threadIdx.x * VEC_SIZE + i * BLOCK_SIZE * VEC_SIZE;
        size_t g_vec_idx = (block_start_float_idx + float_offset_in_block) / VEC_SIZE;
        if (block_start_float_idx + float_offset_in_block < N) {
            g_out_vec[g_vec_idx] = reg[i];
        }
    }
}

// --- Shared Memory Bandwidth Kernel ---
// global -> shared -> global
template <typename VecType, int BLOCK_SIZE, int FLOATS_PER_BLOCK>
__global__ void copy_shared_rw(const float* g_in, float* g_out, size_t N) {
    constexpr int VEC_SIZE = sizeof(VecType) / sizeof(float);
    constexpr int VECS_PER_BLOCK = FLOATS_PER_BLOCK / VEC_SIZE;
    constexpr int VECS_PER_THREAD = VECS_PER_BLOCK / BLOCK_SIZE;

    extern __shared__ char smem_raw[];
    VecType* s_data = reinterpret_cast<VecType*>(smem_raw);

    size_t block_start_float_idx = (size_t)blockIdx.x * FLOATS_PER_BLOCK;
    const VecType* g_in_vec = reinterpret_cast<const VecType*>(g_in);
    VecType* g_out_vec = reinterpret_cast<VecType*>(g_out);

    size_t thread_start_g_idx = block_start_float_idx / VEC_SIZE + threadIdx.x * VECS_PER_THREAD;
    size_t thread_start_s_idx = threadIdx.x * VECS_PER_THREAD;

    // global -> shared
    #pragma unroll
    for (int i = 0; i < VECS_PER_THREAD; ++i) {
        size_t idx = thread_start_g_idx + i;
        size_t sidx = thread_start_s_idx + i;
        if (block_start_float_idx + (threadIdx.x * VECS_PER_THREAD + i) * VEC_SIZE < N) {
            s_data[sidx] = g_in_vec[idx];
        }
    }
    __syncthreads();

    // shared -> global
    #pragma unroll
    for (int i = 0; i < VECS_PER_THREAD; ++i) {
        size_t idx = thread_start_g_idx + i;
        size_t sidx = thread_start_s_idx + i;
        if (block_start_float_idx + (threadIdx.x * VECS_PER_THREAD + i) * VEC_SIZE < N) {
            g_out_vec[idx] = s_data[sidx];
        }
    }
}

// --- Host Code ---

HOST_CODE_BELOW

// Function to run benchmark for a specific kernel (RW: read+write)
template <typename KernelFunc, typename VecType>
float run_benchmark_rw(KernelFunc kernel, const float* d_in, float* d_out, size_t N,
                    int block_size, int floats_per_block, int num_iterations, int warmup_iterations)
{
    size_t num_blocks = (N + floats_per_block - 1) / floats_per_block;
    dim3 grid(num_blocks);
    dim3 block(block_size);

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Warmup runs
    for (int i = 0; i < warmup_iterations; ++i) {
        kernel<<<grid, block>>>(d_in, d_out, N);
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Timed runs
    HIP_CHECK(hipEventRecord(start, 0));
    for (int i = 0; i < num_iterations; ++i) {
        kernel<<<grid, block>>>(d_in, d_out, N);
    }
    HIP_CHECK(hipEventRecord(stop, 0));
    HIP_CHECK(hipEventSynchronize(stop));

    float milliseconds = 0;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    return milliseconds / num_iterations;
}

int main() {
    const size_t COPY_SIZE_MB = 256;
    const int BLOCK_SIZE = 256;
    const int FLOATS_PER_BLOCK = 4096; // 16 KB per block (must be divisible by BLOCK_SIZE * VEC_SIZE_MAX=4)
    const int NUM_ITERATIONS = 100;
    const int WARMUP_ITERATIONS = 10;

    const size_t COPY_SIZE_BYTES = COPY_SIZE_MB * 1024 * 1024;
    const size_t COPY_SIZE_FLOATS = COPY_SIZE_BYTES / sizeof(float);

    // Ensure N is suitable for the block configuration and vector types
    if (COPY_SIZE_FLOATS % FLOATS_PER_BLOCK != 0) {
        fprintf(stderr, "Warning: COPY_SIZE_FLOATS is not perfectly divisible by FLOATS_PER_BLOCK. Adjusting N.\n");
        // Adjust N to be the largest multiple of FLOATS_PER_BLOCK less than or equal to the original N
        // size_t N_adjusted = (COPY_SIZE_FLOATS / FLOATS_PER_BLOCK) * FLOATS_PER_BLOCK;
        // This adjustment might not be necessary if kernels handle boundaries correctly,
        // but using a perfectly divisible size simplifies benchmarking.
        // For simplicity, let's require it for now.
         fprintf(stderr, "Error: COPY_SIZE_FLOATS (%zu) must be divisible by FLOATS_PER_BLOCK (%d).\n", COPY_SIZE_FLOATS, FLOATS_PER_BLOCK);
         exit(EXIT_FAILURE);
    }
     if (FLOATS_PER_BLOCK % (BLOCK_SIZE * 4) != 0) { // Check divisibility for float4
        fprintf(stderr, "Error: FLOATS_PER_BLOCK (%d) must be divisible by BLOCK_SIZE*4 (%d).\n", FLOATS_PER_BLOCK, BLOCK_SIZE*4);
        exit(EXIT_FAILURE);
    }


    printf("Benchmark Configuration:\n");
    printf("  Data Size: %zu MB (%zu floats)\n", COPY_SIZE_MB, COPY_SIZE_FLOATS);
    printf("  Block Size: %d threads\n", BLOCK_SIZE);
    printf("  Floats per Block: %d (%.1f KB shared mem per block)\n", FLOATS_PER_BLOCK, (float)FLOATS_PER_BLOCK * sizeof(float) / 1024.0);
    printf("  Iterations: %d (Warmup: %d)\n", NUM_ITERATIONS, WARMUP_ITERATIONS);
    printf("--------------------------------------------------\n");

    size_t num_blocks = (COPY_SIZE_FLOATS + FLOATS_PER_BLOCK - 1) / FLOATS_PER_BLOCK;
    printf("Total number of blocks: %zu\n", num_blocks);

    // Allocate host and device memory
    float* h_in = (float*)malloc(COPY_SIZE_BYTES);
    if (!h_in) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return 1;
    }
    // Initialize host data (optional, but good practice)
    std::iota(h_in, h_in + COPY_SIZE_FLOATS, 0.0f);

    float *d_in, *d_out;
    HIP_CHECK(hipMalloc(&d_in, COPY_SIZE_BYTES));
    HIP_CHECK(hipMalloc(&d_out, COPY_SIZE_BYTES)); // For dummy write

    // Copy data to device
    HIP_CHECK(hipMemcpy(d_in, h_in, COPY_SIZE_BYTES, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_out, 0, COPY_SIZE_BYTES)); // Clear output buffer

    // --- Run Benchmarks ---
    float avg_ms;
    double bandwidth_gbps;

    printf("%-24s | %-10s | %-15s\n", "Kernel", "Time (ms)", "Bandwidth (GB/s)");
    printf("--------------------------------------------------\n");

    // --- Contiguous (global->reg->global) ---
    printf("[Contiguous Access] (global->reg->global)\n");
    avg_ms = run_benchmark_rw<decltype(&copy_contiguous_rw<float4, BLOCK_SIZE, FLOATS_PER_BLOCK>), float4>(
        copy_contiguous_rw<float4, BLOCK_SIZE, FLOATS_PER_BLOCK>, d_in, d_out, COPY_SIZE_FLOATS,
        BLOCK_SIZE, FLOATS_PER_BLOCK, NUM_ITERATIONS, WARMUP_ITERATIONS);
    bandwidth_gbps = (double)COPY_SIZE_BYTES * 2 / (avg_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0); // read+write
    printf("%-24s | %10.4f | %15.2f\n", "Contig-RW-float4", avg_ms, bandwidth_gbps);

    avg_ms = run_benchmark_rw<decltype(&copy_contiguous_rw<float2, BLOCK_SIZE, FLOATS_PER_BLOCK>), float2>(
        copy_contiguous_rw<float2, BLOCK_SIZE, FLOATS_PER_BLOCK>, d_in, d_out, COPY_SIZE_FLOATS,
        BLOCK_SIZE, FLOATS_PER_BLOCK, NUM_ITERATIONS, WARMUP_ITERATIONS);
    bandwidth_gbps = (double)COPY_SIZE_BYTES * 2 / (avg_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    printf("%-24s | %10.4f | %15.2f\n", "Contig-RW-float2", avg_ms, bandwidth_gbps);

    avg_ms = run_benchmark_rw<decltype(&copy_contiguous_rw<float, BLOCK_SIZE, FLOATS_PER_BLOCK>), float>(
        copy_contiguous_rw<float, BLOCK_SIZE, FLOATS_PER_BLOCK>, d_in, d_out, COPY_SIZE_FLOATS,
        BLOCK_SIZE, FLOATS_PER_BLOCK, NUM_ITERATIONS, WARMUP_ITERATIONS);
    bandwidth_gbps = (double)COPY_SIZE_BYTES * 2 / (avg_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    printf("%-24s | %10.4f | %15.2f\n", "Contig-RW-float1", avg_ms, bandwidth_gbps);

    printf("--------------------------------------------------\n");

    // --- Interleaved (global->reg->global) ---
    printf("[Interleaved Access] (global->reg->global)\n");
    avg_ms = run_benchmark_rw<decltype(&copy_interleaved_rw<float4, BLOCK_SIZE, FLOATS_PER_BLOCK>), float4>(
        copy_interleaved_rw<float4, BLOCK_SIZE, FLOATS_PER_BLOCK>, d_in, d_out, COPY_SIZE_FLOATS,
        BLOCK_SIZE, FLOATS_PER_BLOCK, NUM_ITERATIONS, WARMUP_ITERATIONS);
    bandwidth_gbps = (double)COPY_SIZE_BYTES * 2 / (avg_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    printf("%-24s | %10.4f | %15.2f\n", "Inter-RW-float4", avg_ms, bandwidth_gbps);

    avg_ms = run_benchmark_rw<decltype(&copy_interleaved_rw<float2, BLOCK_SIZE, FLOATS_PER_BLOCK>), float2>(
        copy_interleaved_rw<float2, BLOCK_SIZE, FLOATS_PER_BLOCK>, d_in, d_out, COPY_SIZE_FLOATS,
        BLOCK_SIZE, FLOATS_PER_BLOCK, NUM_ITERATIONS, WARMUP_ITERATIONS);
    bandwidth_gbps = (double)COPY_SIZE_BYTES * 2 / (avg_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    printf("%-24s | %10.4f | %15.2f\n", "Inter-RW-float2", avg_ms, bandwidth_gbps);

    avg_ms = run_benchmark_rw<decltype(&copy_interleaved_rw<float, BLOCK_SIZE, FLOATS_PER_BLOCK>), float>(
        copy_interleaved_rw<float, BLOCK_SIZE, FLOATS_PER_BLOCK>, d_in, d_out, COPY_SIZE_FLOATS,
        BLOCK_SIZE, FLOATS_PER_BLOCK, NUM_ITERATIONS, WARMUP_ITERATIONS);
    bandwidth_gbps = (double)COPY_SIZE_BYTES * 2 / (avg_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    printf("%-24s | %10.4f | %15.2f\n", "Inter-RW-float1", avg_ms, bandwidth_gbps);

    printf("--------------------------------------------------\n");

    // --- Register (global->reg->global) ---
    printf("[Register Only] (global->reg->global, alias)\n");
    avg_ms = run_benchmark_rw<decltype(&copy_contiguous_rw<float4, BLOCK_SIZE, FLOATS_PER_BLOCK>), float4>(
        copy_contiguous_rw<float4, BLOCK_SIZE, FLOATS_PER_BLOCK>, d_in, d_out, COPY_SIZE_FLOATS,
        BLOCK_SIZE, FLOATS_PER_BLOCK, NUM_ITERATIONS, WARMUP_ITERATIONS);
    bandwidth_gbps = (double)COPY_SIZE_BYTES * 2 / (avg_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    printf("%-24s | %10.4f | %15.2f\n", "Reg-RW-Contig-float4", avg_ms, bandwidth_gbps);

    avg_ms = run_benchmark_rw<decltype(&copy_contiguous_rw<float2, BLOCK_SIZE, FLOATS_PER_BLOCK>), float2>(
        copy_contiguous_rw<float2, BLOCK_SIZE, FLOATS_PER_BLOCK>, d_in, d_out, COPY_SIZE_FLOATS,
        BLOCK_SIZE, FLOATS_PER_BLOCK, NUM_ITERATIONS, WARMUP_ITERATIONS);
    bandwidth_gbps = (double)COPY_SIZE_BYTES * 2 / (avg_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    printf("%-24s | %10.4f | %15.2f\n", "Reg-RW-Contig-float2", avg_ms, bandwidth_gbps);

    avg_ms = run_benchmark_rw<decltype(&copy_contiguous_rw<float, BLOCK_SIZE, FLOATS_PER_BLOCK>), float>(
        copy_contiguous_rw<float, BLOCK_SIZE, FLOATS_PER_BLOCK>, d_in, d_out, COPY_SIZE_FLOATS,
        BLOCK_SIZE, FLOATS_PER_BLOCK, NUM_ITERATIONS, WARMUP_ITERATIONS);
    bandwidth_gbps = (double)COPY_SIZE_BYTES * 2 / (avg_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    printf("%-24s | %10.4f | %15.2f\n", "Reg-RW-Contig-float1", avg_ms, bandwidth_gbps);

    printf("--------------------------------------------------\n");

    // --- Register Interleaved (global->reg->global) ---
    printf("[Register Only] (global->reg->global, interleaved)\n");
    avg_ms = run_benchmark_rw<decltype(&copy_interleaved_rw<float4, BLOCK_SIZE, FLOATS_PER_BLOCK>), float4>(
        copy_interleaved_rw<float4, BLOCK_SIZE, FLOATS_PER_BLOCK>, d_in, d_out, COPY_SIZE_FLOATS,
        BLOCK_SIZE, FLOATS_PER_BLOCK, NUM_ITERATIONS, WARMUP_ITERATIONS);
    bandwidth_gbps = (double)COPY_SIZE_BYTES * 2 / (avg_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    printf("%-24s | %10.4f | %15.2f\n", "Reg-RW-Inter-float4", avg_ms, bandwidth_gbps);

    avg_ms = run_benchmark_rw<decltype(&copy_interleaved_rw<float2, BLOCK_SIZE, FLOATS_PER_BLOCK>), float2>(
        copy_interleaved_rw<float2, BLOCK_SIZE, FLOATS_PER_BLOCK>, d_in, d_out, COPY_SIZE_FLOATS,
        BLOCK_SIZE, FLOATS_PER_BLOCK, NUM_ITERATIONS, WARMUP_ITERATIONS);
    bandwidth_gbps = (double)COPY_SIZE_BYTES * 2 / (avg_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    printf("%-24s | %10.4f | %15.2f\n", "Reg-RW-Inter-float2", avg_ms, bandwidth_gbps);

    avg_ms = run_benchmark_rw<decltype(&copy_interleaved_rw<float, BLOCK_SIZE, FLOATS_PER_BLOCK>), float>(
        copy_interleaved_rw<float, BLOCK_SIZE, FLOATS_PER_BLOCK>, d_in, d_out, COPY_SIZE_FLOATS,
        BLOCK_SIZE, FLOATS_PER_BLOCK, NUM_ITERATIONS, WARMUP_ITERATIONS);
    bandwidth_gbps = (double)COPY_SIZE_BYTES * 2 / (avg_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    printf("%-24s | %10.4f | %15.2f\n", "Reg-RW-Inter-float1", avg_ms, bandwidth_gbps);

    printf("--------------------------------------------------\n");

    // --- Shared (global->shared->global) Contiguous ---
    printf("[Shared Memory] (global->shared->global, contiguous)\n");
    avg_ms = run_benchmark_rw<decltype(&copy_shared_rw<float4, BLOCK_SIZE, FLOATS_PER_BLOCK>), float4>(
        copy_shared_rw<float4, BLOCK_SIZE, FLOATS_PER_BLOCK>, d_in, d_out, COPY_SIZE_FLOATS,
        BLOCK_SIZE, FLOATS_PER_BLOCK, NUM_ITERATIONS, WARMUP_ITERATIONS);
    bandwidth_gbps = (double)COPY_SIZE_BYTES * 2 / (avg_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    printf("%-24s | %10.4f | %15.2f\n", "Shared-RW-Contig-float4", avg_ms, bandwidth_gbps);

    avg_ms = run_benchmark_rw<decltype(&copy_shared_rw<float2, BLOCK_SIZE, FLOATS_PER_BLOCK>), float2>(
        copy_shared_rw<float2, BLOCK_SIZE, FLOATS_PER_BLOCK>, d_in, d_out, COPY_SIZE_FLOATS,
        BLOCK_SIZE, FLOATS_PER_BLOCK, NUM_ITERATIONS, WARMUP_ITERATIONS);
    bandwidth_gbps = (double)COPY_SIZE_BYTES * 2 / (avg_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    printf("%-24s | %10.4f | %15.2f\n", "Shared-RW-Contig-float2", avg_ms, bandwidth_gbps);

    avg_ms = run_benchmark_rw<decltype(&copy_shared_rw<float, BLOCK_SIZE, FLOATS_PER_BLOCK>), float>(
        copy_shared_rw<float, BLOCK_SIZE, FLOATS_PER_BLOCK>, d_in, d_out, COPY_SIZE_FLOATS,
        BLOCK_SIZE, FLOATS_PER_BLOCK, NUM_ITERATIONS, WARMUP_ITERATIONS);
    bandwidth_gbps = (double)COPY_SIZE_BYTES * 2 / (avg_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    printf("%-24s | %10.4f | %15.2f\n", "Shared-RW-Contig-float1", avg_ms, bandwidth_gbps);

    printf("--------------------------------------------------\n");

    // --- Shared (global->shared->global) Interleaved ---
    printf("[Shared Memory] (global->shared->global, interleaved)\n");
    avg_ms = run_benchmark_rw<decltype(&copy_shared_rw<float4, BLOCK_SIZE, FLOATS_PER_BLOCK>), float4>(
        copy_shared_rw<float4, BLOCK_SIZE, FLOATS_PER_BLOCK>, d_in, d_out, COPY_SIZE_FLOATS,
        BLOCK_SIZE, FLOATS_PER_BLOCK, NUM_ITERATIONS, WARMUP_ITERATIONS);
    bandwidth_gbps = (double)COPY_SIZE_BYTES * 2 / (avg_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    printf("%-24s | %10.4f | %15.2f\n", "Shared-RW-Inter-float4", avg_ms, bandwidth_gbps);

    avg_ms = run_benchmark_rw<decltype(&copy_shared_rw<float2, BLOCK_SIZE, FLOATS_PER_BLOCK>), float2>(
        copy_shared_rw<float2, BLOCK_SIZE, FLOATS_PER_BLOCK>, d_in, d_out, COPY_SIZE_FLOATS,
        BLOCK_SIZE, FLOATS_PER_BLOCK, NUM_ITERATIONS, WARMUP_ITERATIONS);
    bandwidth_gbps = (double)COPY_SIZE_BYTES * 2 / (avg_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    printf("%-24s | %10.4f | %15.2f\n", "Shared-RW-Inter-float2", avg_ms, bandwidth_gbps);

    avg_ms = run_benchmark_rw<decltype(&copy_shared_rw<float, BLOCK_SIZE, FLOATS_PER_BLOCK>), float>(
        copy_shared_rw<float, BLOCK_SIZE, FLOATS_PER_BLOCK>, d_in, d_out, COPY_SIZE_FLOATS,
        BLOCK_SIZE, FLOATS_PER_BLOCK, NUM_ITERATIONS, WARMUP_ITERATIONS);
    bandwidth_gbps = (double)COPY_SIZE_BYTES * 2 / (avg_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    printf("%-24s | %10.4f | %15.2f\n", "Shared-RW-Inter-float1", avg_ms, bandwidth_gbps);

    printf("--------------------------------------------------\n");

    // Cleanup
    free(h_in);
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    return 0;
}