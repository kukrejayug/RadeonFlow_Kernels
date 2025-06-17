#include <hip/hip_runtime.h>
#include <iostream>
#include <ck/utility/data_type.hpp>
#include <vector>
#include <cstdio> 
#include "../include/clangd_workaround.h"

// Kacro for HIP API error checking
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

const constexpr int WARP_SIZE = 64;
const constexpr int BLOCK_SIZE = 64;
const constexpr int M = 512, K = 512;
const constexpr int BM = 8, BK = 8;
const constexpr int SCALE = 10;


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

__device__ void inline load_transpose_naive(float smem[K][M], const float gmem[M][K]) {
    auto rsrc = make_wave_buffer_resource(gmem);
    int x = threadIdx.x % K;
    int y = threadIdx.x / K;
    smem[x][y] = gmem[y][x];
}

__device__ void inline load_transpose_async(float smem[BK][BM], const float gmem[M][K]) {
    auto rsrc = make_wave_buffer_resource(gmem);
    int i = threadIdx.x % M;
    int j = threadIdx.x / M;
    int v_offset = (j * K + i) * sizeof(float);
    async_buffer_load_dword_v(smem, rsrc, v_offset);
    async_buffer_load_fence(0);
}

__device__ void inline store_direct(const float smem[BK][BM], float gmem[K][M]) {
    auto rsrc = make_wave_buffer_resource(gmem);
    int x = threadIdx.x % BM;
    int y = threadIdx.x / BM;
    gmem[y][x] = smem[y][x];
}

__global__ void ref_transpose_kernel(const float src[M][K], float dst[K][M]) {
    __shared__ float smem[BK][BM];
    static_assert(BK * BM == BLOCK_SIZE);
    load_transpose_async(smem, src);
    __syncthreads();
    store_direct(smem, dst);
}


HOST_CODE_BELOW

int main() {
    float h_src[M][K], h_dst[K][M];
    for (int i = 0; i < BM; ++i) {
        for (int j = 0; j < BK; ++j) {
            h_src[i][j] = static_cast<float>(i * SCALE + j);
        }
    }
    printf("==================== SRC ====================\n");
    for (int i = 0; i < BM; ++i) {
        for (int j = 0; j < BK; ++j) {
            printf("%2d ", static_cast<int>(h_src[i][j]));
        }
        printf("\n");
    }
    float (*d_src)[K];
    float (*d_dst)[M];
    HIP_CALL(hipMalloc(&d_src, M * K * sizeof(float)));
    HIP_CALL(hipMalloc(&d_dst, K * M * sizeof(float)));
    HIP_CALL(hipMemcpy(d_src, h_src, M * K * sizeof(float), hipMemcpyHostToDevice));
    ref_transpose_kernel<<<1, BLOCK_SIZE>>>(d_src, d_dst);
    HIP_CALL(hipMemcpy(h_dst, d_dst, K * M * sizeof(float), hipMemcpyDeviceToHost));
    printf("==================== DST ====================\n"); 
    for (int i = 0; i < BK; ++i) {
        for (int j = 0; j < BM; ++j) {
            printf("%2d ", static_cast<int>(h_dst[i][j]));
        }
        printf("\n");
    }
    
    HIP_CALL(hipFree(d_src));
    HIP_CALL(hipFree(d_dst));
    
    return 0;
}