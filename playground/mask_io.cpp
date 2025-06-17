#include <hip/hip_runtime.h>
#include <iostream>
#include <ck/utility/data_type.hpp>
#include <vector>

// Macro for HIP API error checking
#define HIP_CALL(cmd) do {                                         \
    hipError_t hip_status = cmd;                                   \
    if (hip_status != hipSuccess) {                                \
        std::cerr << "HIP error: " << hipGetErrorString(hip_status)\
                  << " at " << __FILE__ << ":" << __LINE__         \
                  << " '" << #cmd << "'" << std::endl;             \
        std::abort();                                              \
    }                                                              \
} while(0)

using int32x4_t = ck::int32x4_t;

const constexpr int N = 512;
const constexpr int WARP_SIZE = 64;
/* Benchmark Config */
const constexpr int BENCHMARK_DATA = 32 * 1024 * 1024; // 32MB
const constexpr int BUFFER_SIZE = 16 * 1024; // 16KB
const constexpr int BLOCK_NUM = 512;
const constexpr int BLOCK_SIZE = 256;
const constexpr int ONE_ROUND_DATA = BUFFER_SIZE * BLOCK_NUM; // 16KB * 16 = 256KB
static_assert(BENCHMARK_DATA % ONE_ROUND_DATA == 0, "BENCHMARK_DATA must be divisible by ONE_ROUND_DATA");



__device__ int32x4_t inline make_wave_buffer_resource(const void* ptr, uint32_t size = 0xffffffff) {
    int32x4_t res;
    
    // Pack the 64-bit pointer into two 32-bit integers
    uint64_t ptr_val = reinterpret_cast<uint64_t>(ptr);
    res.x = static_cast<uint32_t>(ptr_val);
    res.y = static_cast<uint32_t>(ptr_val >> 32);
    
    // Set buffer size and format
    res.z = size;  // Buffer size in bytes
    res.w = 0x00020000;  // hardcoded for gfx942
    
    res.x = __builtin_amdgcn_readfirstlane(res.x);
    res.y = __builtin_amdgcn_readfirstlane(res.y);
    res.z = __builtin_amdgcn_readfirstlane(res.z);
    res.w = __builtin_amdgcn_readfirstlane(res.w);
    return res;
}

__device__ void inline async_buffer_load_fence(int cnt = 0) {
    asm volatile("s_waitcnt vmcnt(%0)" :: "n"(cnt) : "memory");
}

__device__ void inline async_buffer_load_dword_v(
    void *smem,
    int32x4_t rsrc,
    int v_offset
) {
    const auto lds_ptr_sgpr = __builtin_amdgcn_readfirstlane((reinterpret_cast<uintptr_t>(smem)));
    asm volatile(R"(
        s_mov_b32 m0, %0; \n\t
        buffer_load_dword %1, %2, 0 offen lds
    )" :: "s"(lds_ptr_sgpr), "v"(v_offset), "s"(rsrc)
        : "memory");
}


__global__ void test_async_buffer_load(int* global_data, int* result) {
    __shared__ int lds_data[N];
    lds_data[threadIdx.x] = -1;
    __syncthreads();
    int32x4_t global_data_desc = make_wave_buffer_resource(global_data + threadIdx.x / WARP_SIZE * WARP_SIZE);
    async_buffer_load_dword_v(lds_data + threadIdx.x / WARP_SIZE * WARP_SIZE, global_data_desc, (threadIdx.x % WARP_SIZE) * sizeof(int));
    async_buffer_load_fence();
    __syncthreads();
    result[threadIdx.x] = lds_data[threadIdx.x];
}

__global__ void benchmark_buffer_load(char* global_data) {
    __shared__ char buffer[BUFFER_SIZE];
    static_assert(BENCHMARK_DATA % (BUFFER_SIZE * BLOCK_NUM) == 0, "BENCHMARK_DATA must be divisible by (BUFFER_SIZE * BLOCK_NUM)");
    static_assert(BUFFER_SIZE % (BLOCK_SIZE * sizeof(int)) == 0, "BUFFER_SIZE must be divisible by (BLOCK_SIZE * sizeof(int))");
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    
    // Compute base pointer for this block's portion of global data
    char* block_data = global_data + bid * BUFFER_SIZE;
    
    for (int i = 0; i < BENCHMARK_DATA / (BLOCK_NUM * BUFFER_SIZE); i++) {
        // Each iteration accesses one chunk of data of size BUFFER_SIZE
        char* current_data = block_data + i * BLOCK_NUM * BUFFER_SIZE;
        
        // Process this chunk in smaller segments
        for (int j = 0; j < BUFFER_SIZE / (BLOCK_SIZE * sizeof(int)); j++) {
            // Calculate pointers for this warp
            char* warp_global_data = current_data + j * BLOCK_SIZE * sizeof(int) + wid * WARP_SIZE * sizeof(int);
            char* warp_buffer = buffer + j * BLOCK_SIZE * sizeof(int) + wid * WARP_SIZE * sizeof(int);
            
            // Create resource descriptor for global memory
            int32x4_t global_data_desc = make_wave_buffer_resource(warp_global_data);
            
            // Perform async load - each thread loads one int
            async_buffer_load_dword_v(warp_buffer, global_data_desc, lane_id * sizeof(int));
        }
        
        // Wait for all loads to complete before next iteration
        async_buffer_load_fence();
    }
}

__global__ void benchmark_buffer_load_double_buffer(char* global_data) {
    __shared__ char buffer[BUFFER_SIZE];
    static_assert(BENCHMARK_DATA % (BUFFER_SIZE * BLOCK_NUM) == 0, "BENCHMARK_DATA must be divisible by (BUFFER_SIZE * BLOCK_NUM)");
    static_assert(BUFFER_SIZE % (BLOCK_SIZE * sizeof(int)) == 0, "BUFFER_SIZE must be divisible by (BLOCK_SIZE * sizeof(int))");
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    
    // Compute base pointer for this block's portion of global data
    char* block_data = global_data + bid * BUFFER_SIZE;
    
    for (int i = 0; i < BENCHMARK_DATA / (BLOCK_NUM * BUFFER_SIZE); i++) {
        // Each iteration accesses one chunk of data of size BUFFER_SIZE
        char* current_data = block_data + i * BLOCK_NUM * BUFFER_SIZE;
        
        // Process this chunk in smaller segments
        int j = 0;
        for (; j < BUFFER_SIZE / (BLOCK_SIZE * sizeof(int)) / 2; j++) {
            // Calculate pointers for this warp
            char* warp_global_data = current_data + j * BLOCK_SIZE * sizeof(int) + wid * WARP_SIZE * sizeof(int);
            char* warp_buffer = buffer + j * BLOCK_SIZE * sizeof(int) + wid * WARP_SIZE * sizeof(int);
            
            // Create resource descriptor for global memory
            int32x4_t global_data_desc = make_wave_buffer_resource(warp_global_data);
            
            // Perform async load - each thread loads one int
            async_buffer_load_dword_v(warp_buffer, global_data_desc, lane_id * sizeof(int));
        }

        async_buffer_load_fence(BUFFER_SIZE / (BLOCK_SIZE * sizeof(int)) / 2);
        
        for (; j < BUFFER_SIZE / (BLOCK_SIZE * sizeof(int)); j++) {
            // Calculate pointers for this warp
            char* warp_global_data = current_data + j * BLOCK_SIZE * sizeof(int) + wid * WARP_SIZE * sizeof(int);
            char* warp_buffer = buffer + j * BLOCK_SIZE * sizeof(int) + wid * WARP_SIZE * sizeof(int);
            
            // Create resource descriptor for global memory
            int32x4_t global_data_desc = make_wave_buffer_resource(warp_global_data);
            
            // Perform async load - each thread loads one int
            async_buffer_load_dword_v(warp_buffer, global_data_desc, lane_id * sizeof(int));
        }
        // Wait for all loads to complete before next iteration
        async_buffer_load_fence(BUFFER_SIZE / (BLOCK_SIZE * sizeof(int)) / 2);
    }
    async_buffer_load_fence();
}

// Test memory bandwidth of buffer_load instructions
void test_benchmark_buffer_load() {
    std::cout << "===== Memory Bandwidth Test =====" << std::endl;
    std::cout << "Data size: " << BENCHMARK_DATA / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Buffer size: " << BUFFER_SIZE / 1024 << " KB" << std::endl;
    std::cout << "Block number: " << BLOCK_NUM << std::endl;
    std::cout << "Block size: " << BLOCK_SIZE << std::endl;
    
    // Allocate device memory
    int *d_global_data;
    HIP_CALL(hipMalloc(&d_global_data, BENCHMARK_DATA));
    
    // Initialize with some data
    std::vector<int> h_global_data(BENCHMARK_DATA / sizeof(int), 1);
    HIP_CALL(hipMemcpy(d_global_data, h_global_data.data(), BENCHMARK_DATA, hipMemcpyHostToDevice));
    
    // Create timing events
    hipEvent_t start, stop;
    HIP_CALL(hipEventCreate(&start));
    HIP_CALL(hipEventCreate(&stop));
    
    // Warmup run
    benchmark_buffer_load<<<BLOCK_NUM, BLOCK_SIZE>>>(reinterpret_cast<char*>(d_global_data));
    HIP_CALL(hipDeviceSynchronize());
    
    // Benchmark runs
    const int NUM_ITERATIONS = 10;
    float totalTime = 0.0f;
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        // Record start time
        HIP_CALL(hipEventRecord(start, 0));
        
        // Launch kernel
        benchmark_buffer_load<<<BLOCK_NUM, BLOCK_SIZE>>>(reinterpret_cast<char*>(d_global_data));
        
        // Record stop time
        HIP_CALL(hipEventRecord(stop, 0));
        HIP_CALL(hipEventSynchronize(stop));
        
        // Calculate elapsed time
        float milliseconds = 0.0f;
        HIP_CALL(hipEventElapsedTime(&milliseconds, start, stop));
        totalTime += milliseconds;
        
        // Calculate bandwidth for this iteration
        float bandwidth = (BENCHMARK_DATA / (1024.0f * 1024.0f * 1024.0f)) / (milliseconds / 1000.0f);
        std::cout << "Iteration " << iter + 1 << ": " << bandwidth << " GB/s" << std::endl;
    }
    
    // Calculate average bandwidth
    float avgTime = totalTime / NUM_ITERATIONS;
    float avgBandwidth = (BENCHMARK_DATA / (1024.0f * 1024.0f * 1024.0f)) / (avgTime / 1000.0f);
    
    std::cout << "Average bandwidth: " << avgBandwidth << " GB/s" << std::endl;
    std::cout << "Average time: " << avgTime << " ms" << std::endl;
    
    // Cleanup
    HIP_CALL(hipEventDestroy(start));
    HIP_CALL(hipEventDestroy(stop));
    HIP_CALL(hipFree(d_global_data));
}

int main() {
    // Allocate host memory
    std::vector<int> h_global_data(N);
    std::vector<int> h_result(N, 0);
    
    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_global_data[i] = i;
    }
    
    // Allocate device memory
    int *d_global_data, *d_result;
    HIP_CALL(hipMalloc(&d_global_data, N * sizeof(int)));
    HIP_CALL(hipMalloc(&d_result, N * sizeof(int)));
    
    // Copy input data from host to device
    HIP_CALL(hipMemcpy(d_global_data, h_global_data.data(), N * sizeof(int), hipMemcpyHostToDevice));
    
    // Launch kernel
    dim3 grid(1);
    dim3 block(N);
    test_async_buffer_load<<<grid, block>>>(d_global_data, d_result);
    
    // Wait for kernel completion
    HIP_CALL(hipDeviceSynchronize());
    
    // Copy results back from device to host
    HIP_CALL(hipMemcpy(h_result.data(), d_result, N * sizeof(int), hipMemcpyDeviceToHost));
    
    // Output input data
    std::cout << "Input data (all " << N << " elements):" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << h_global_data[i] << " ";
        if ((i + 1) % 32 == 0) std::cout << std::endl;
    }
    if (N % 32 != 0) std::cout << std::endl;
    
    // Output result data
    std::cout << "Output data (all " << N << " elements):" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << h_result[i] << " ";
        if ((i + 1) % 32 == 0) std::cout << std::endl;
    }
    if (N % 32 != 0) std::cout << std::endl;
    
    // Verify results
    bool test_passed = true;
    for (int i = 0; i < N; i++) {
        if (h_result[i] != h_global_data[i]) {
            std::cout << "Mismatch at position " << i << ": expected " 
                      << h_global_data[i] << ", got " << h_result[i] << std::endl;
            test_passed = false;
            break;
        }
    }
    
    if (test_passed) {
        std::cout << "Test PASSED!" << std::endl;
    } else {
        std::cout << "Test FAILED!" << std::endl;
    }
    
    // Free device memory
    HIP_CALL(hipFree(d_global_data));
    HIP_CALL(hipFree(d_result));
    
    // Run bandwidth test
    test_benchmark_buffer_load();
    
    return 0;
}