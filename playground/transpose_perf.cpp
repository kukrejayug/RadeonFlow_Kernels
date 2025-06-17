#include <__clang_hip_runtime_wrapper.h>
#include <hip/amd_detail/amd_hip_runtime.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <ck/utility/data_type.hpp>
#include <cassert>
#include "../include/gpu_libs.h"
#include "../include/gpu_types.h"
#include "../include/clangd_workaround.h"

#define HIP_CHECK(command) { \
    hipError_t status = command; \
    if (status != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s at %s:%d\n", hipGetErrorString(status), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

template<typename Elem, int M, int N, int TILE_DIM, int BLOCK_SIZE, int VEC_SIZE>
__global__ void transpose_kernel2(Elem odata[N][M], const Elem idata[M][N]) {
        /**
        GDS, Elem = fp16
        [0 ] [1 ] [2 ] [3 ]
        [4 ] [5 ] [6 ] [7 ]
        [8 ] [9 ] [10] [11]
        [12] [13] [14] [15]

        Reg, WAVE_SIZE = 4 
        #0 [0 ] [1 ] [8 ] [9 ]
        #1 [2 ] [3 ] [10] [11]
        #2 [4 ] [5 ] [12] [13]
        #3 [6 ] [7 ] [14] [15]

        LDS
        [0 ] [1 ] [8 ] [9 ]
        [2 ] [3 ] [10] [11]
        [4 ] [5 ] [12] [13]
        [6 ] [7 ] [14] [15]

        GDS
        [0 ] [4 ] [8 ] [12]
        [1 ] [5 ] [9 ] [13]
        [2 ] [6 ] [10] [14]
        [3 ] [7 ] [11] [15]
    */

    constexpr int TBLOCK_X = TILE_DIM / VEC_SIZE;
    constexpr int TBLOCK_Y = BLOCK_SIZE / TBLOCK_X;
    constexpr int REG_BUF_SIZE = TILE_DIM * TILE_DIM / WAVE_SIZE;

    Elem reg_buf[REG_BUF_SIZE];
    __shared__ Elem lds_buf[TILE_DIM][TILE_DIM];

    int block_x = __builtin_amdgcn_readfirstlane(blockIdx.x) * TILE_DIM;
    int block_y = __builtin_amdgcn_readfirstlane(blockIdx.y) * TILE_DIM;
    int thread_x = threadIdx.x * VEC_SIZE;
    int thread_y = threadIdx.y;

    #pragma unroll
    for (int k = 0; k < REG_BUF_SIZE; ++k) {
        reg_buf[k] = idata[block_y + thread_y][block_x + thread_x + k];
    }
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < REG_BUF_SIZE; ++k) {
        lds_buf[thread_y][thread_x + k] = reg_buf[k];
    }

    #pragma unroll
    for (int k = 0; k < REG_BUF_SIZE; ++k) {
        odata[block_x + thread_x + k][block_y + thread_y] = lds_buf[thread_y][thread_x + k];
    }
}



template <typename Elem, int M, int N, int TILE_DIM, int BLOCK_SIZE, int VEC_SIZE>
__launch_bounds__(BLOCK_SIZE)
__global__ void transpose_kernel(Elem *odata, const Elem *idata) {

    constexpr auto TBLOCK_X = TILE_DIM / VEC_SIZE;
    constexpr auto TBLOCK_Y = BLOCK_SIZE / TBLOCK_X;

    // avoid read bank conflict
    // VEC_SIZE * (TILE_DIM + d) * sizeof(Elem) = TBLOCK_Y / (BLOCK_SIZE / WARP_SIZE) * sizeof(Elem) + 128k
    // each warp read row = TILE_DIM (in VEC_SIZE reads), col = TBLOCK_Y / (BLOCK_SIZE / WARP_SIZE)
    // warp 0                     warp 1
    // t0    t16    t32    t48    ...
    // ...
    // t1
    // ...
    // t15
    // don't know why padding to d as described above is not working, maybe gpu could merge contigious ds_read_u8 and
    // cause padding to be TBLOCK_Y / (BLOCK_SIZE / WARP_SIZE)
    constexpr auto PADDING = TBLOCK_Y / (BLOCK_SIZE / warpSize);
    __shared__ Elem tile[TILE_DIM][TILE_DIM + PADDING];

    int x = blockIdx.x * TILE_DIM + threadIdx.x * VEC_SIZE;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

// Load tile
#pragma unroll
    for (int i = 0; i < TILE_DIM; i += TBLOCK_Y) {
#pragma unroll
        for (int v = 0; v < VEC_SIZE; v++) {
            tile[threadIdx.y + i][threadIdx.x * VEC_SIZE + v] = idata[(y + i) * N + x + v];
        }
    }

    __syncthreads();

    // Transpose indices
    x = blockIdx.y * TILE_DIM + threadIdx.x * VEC_SIZE;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

// Write tile
#pragma unroll
    for (int i = 0; i < TILE_DIM; i += TBLOCK_Y) {
#pragma unroll
        for (int v = 0; v < VEC_SIZE; v++) {
            odata[(y + i) * M + x + v] = tile[threadIdx.x * VEC_SIZE + v][threadIdx.y + i];
        }
    }
}


template<typename Elem>
__global__ void naive_transpose_kernel(Elem* odata, const Elem* idata, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < N && idy < M) {
        odata[idx * M + idy] = idata[idy * N + idx];
    }
}

template<typename Elem, int M, int N, int TILE_DIM, int BLOCK_SIZE, int VEC_SIZE>
void launch_transpose(Elem *out, const Elem *in) {
    static_assert(TILE_DIM % VEC_SIZE == 0);
    constexpr auto TBLOCK_X = TILE_DIM / VEC_SIZE;
    static_assert(BLOCK_SIZE % TBLOCK_X == 0);
    constexpr auto TBLOCK_Y = BLOCK_SIZE / TBLOCK_X;
    
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    dim3 block(TBLOCK_X, TBLOCK_Y);
    
    transpose_kernel<Elem, M, N, TILE_DIM, BLOCK_SIZE, VEC_SIZE><<<grid, block>>>(out, in);
}

template<typename Elem, int M, int N, int TILE_DIM, int BLOCK_SIZE, int VEC_SIZE>
void launch_transpose2(Elem *out, const Elem *in) {
    static_assert(TILE_DIM % VEC_SIZE == 0);
    constexpr auto TBLOCK_X = TILE_DIM / VEC_SIZE;
    static_assert(BLOCK_SIZE % TBLOCK_X == 0);
    constexpr auto TBLOCK_Y = BLOCK_SIZE / TBLOCK_X;
    
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    dim3 block(TBLOCK_X, TBLOCK_Y);
    
    transpose_kernel2<Elem, M, N, TILE_DIM, BLOCK_SIZE, VEC_SIZE><<<grid, block>>>(reinterpret_cast<Elem(*)[M]>(out), 
                                                                                      reinterpret_cast<const Elem(*)[N]>(in));
}

template<typename Elem>
void launch_naive_transpose(Elem *out, const Elem *in, int M, int N) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    naive_transpose_kernel<<<grid, block>>>(out, in, M, N);
}

HOST_CODE_BELOW

template<typename T>
bool verify_transpose(const T* original, const T* transposed, int M, int N, float tolerance = 1e-5f) {
    bool passed = true;
    int error_count = 0;
    const int max_errors = 10;
    
    for (int i = 0; i < M && error_count < max_errors; ++i) {
        for (int j = 0; j < N && error_count < max_errors; ++j) {
            T orig_val = original[i * N + j];
            T trans_val = transposed[j * M + i];
            
            if (abs(static_cast<float>(orig_val - trans_val)) > tolerance) {
                if (error_count == 0) {
                    printf("Transpose verification failed!\n");
                }
                printf("Error at [%d, %d]: original=%f, transposed=%f\n", 
                       i, j, static_cast<float>(orig_val), static_cast<float>(trans_val));
                error_count++;
                passed = false;
            }
        }
    }
    
    if (passed) {
        printf("Transpose verification PASSED!\n");
    } else {
        printf("Transpose verification FAILED with %d+ errors!\n", error_count);
    }
    
    return passed;
}

template<typename T, typename LaunchFunc>
double benchmark_transpose_kernel(LaunchFunc launch_func, 
                                 T* d_out, const T* d_in, int M, int N,
                                 const std::string& kernel_name, int iterations = 100) {
    // Warmup
    for (int i = 0; i < 5; ++i) {
        launch_func(d_out, d_in, M, N);
    }
    HIP_CHECK(hipDeviceSynchronize());
    
    // Benchmark
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iterations; ++i) {
        launch_func(d_out, d_in, M, N);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    
    double avg_time_us = (ms * 1000.0) / iterations; // Convert to microseconds
    size_t bytes = 2ULL * M * N * sizeof(T); // Read + Write
    double bandwidth = (bytes / (avg_time_us / 1000000.0)) / (1024.0 * 1024.0 * 1024.0); // GB/s
    
    printf("%-20s: %8.2f us, %8.2f GB/s\n", kernel_name.c_str(), avg_time_us, bandwidth);
    
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    
    return bandwidth;
}

template<typename T, int M, int N>
void test_transpose_performance() {
    printf("\n=== Testing %dx%d transpose performance (element size: %zu bytes) ===\n", 
           M, N, sizeof(T));
    
    size_t size = M * N;
    size_t bytes = size * sizeof(T);
    
    // Allocate host memory
    std::vector<T> h_input(size);
    std::vector<T> h_output(size);
    
    // Initialize input data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < size; ++i) {
        h_input[i] = static_cast<T>(dis(gen));
    }
    
    // Allocate device memory
    T* d_input;
    T* d_output;
    HIP_CHECK(hipMalloc(&d_input, bytes));
    HIP_CHECK(hipMalloc(&d_output, bytes));
    
    // Copy data to device
    HIP_CHECK(hipMemcpy(d_input, h_input.data(), bytes, hipMemcpyHostToDevice));
    
    // Test optimized transpose kernel
    constexpr int TILE_DIM = 64;
    constexpr int BLOCK_SIZE = 512;
    constexpr int VEC_SIZE = 4;
    
    launch_transpose<T, M, N, TILE_DIM, BLOCK_SIZE, VEC_SIZE>(d_output, d_input);
    HIP_CHECK(hipDeviceSynchronize());
    
    // Verify correctness
    HIP_CHECK(hipMemcpy(h_output.data(), d_output, bytes, hipMemcpyDeviceToHost));
    bool correct = verify_transpose(h_input.data(), h_output.data(), M, N);
    
    if (correct) {
        // Benchmark different kernels
        printf("\nBenchmark results:\n");
        
        auto opt_bw = benchmark_transpose_kernel(
            [](T* out, const T* in, int, int) {
                launch_transpose<T, M, N, TILE_DIM, BLOCK_SIZE, VEC_SIZE>(out, in);
            },
            d_output, d_input, M, N, "Optimized");
            
        auto naive_bw = benchmark_transpose_kernel(
            launch_naive_transpose<T>,
            d_output, d_input, M, N, "Naive");
            
        printf("Speedup: %.2fx\n", opt_bw / naive_bw);
    }
    
    // Cleanup
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
}

int main() {
    printf("Transpose Kernel Performance Test\n");
    printf("=================================\n");
    
    // Test different matrix sizes and data types
    test_transpose_performance<__FP8_TYPE, 1024, 1024>();
    test_transpose_performance<__FP8_TYPE, 2048, 2048>();
    test_transpose_performance<__FP8_TYPE, 4096, 4096>();
    test_transpose_performance<__FP8_TYPE, 1024, 2048>();
    test_transpose_performance<__FP8_TYPE, 2048, 1024>();
    
    
    return 0;
}
