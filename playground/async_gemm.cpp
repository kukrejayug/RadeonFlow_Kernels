#include "../include/gpu_types.h"
#include <hip/hip_runtime.h>
#include <iostream>
#include <ck/utility/data_type.hpp>
#include <vector>
using int32x4_t = ck::int32x4_t;

// struct ResourceFlags {
//     union {
//         struct {
//             uint32_t dst_sel_x : 3;         // 96-98
//             uint32_t dst_sel_y : 3;         // 99-101
//             uint32_t dst_sel_z : 3;         // 102-104
//             uint32_t dst_sel_w : 3;         // 105-107
//             uint32_t num_format : 3;        // 108-110
//             uint32_t data_format : 4;       // 111-114
//             uint32_t user_vm_enable : 1;    // 115
//             uint32_t user_vm_mode : 1;      // 116
//             uint32_t index_stride : 2;      // 117-118
//             uint32_t add_tid_enable : 1;    // 119
//             uint32_t rsvd_0 : 3;            // 120-122
//             uint32_t nv : 1;                // 123
//             uint32_t rsvd_1 : 2;            // 124-125
//             uint32_t type : 2;              // 126-127
//         };
//         uint32_t u32;
//     };
// };

// template<int stride>
// __device__ int32x4_t inline make_wave_buffer_resource(const void* ptr, uint32_t size = 0xffffffff) {
//     int32x4_t res;
    
//     uint64_t ptr_val = reinterpret_cast<uint64_t>(ptr);
//     res.x = static_cast<uint32_t>(ptr_val);
//     res.y = static_cast<uint32_t>(ptr_val >> 32);
//     res.z = static_cast<uint32_t>(size);
//     res.w = static_cast<uint32_t>(ResourceFlags{
//         .dst_sel_x = 0,          // Select X component from buffer data
//         .dst_sel_y = 0,          // Select Y component from buffer data
//         .dst_sel_z = 0,          // Select Z component from buffer data
//         .dst_sel_w = 0,          // Select W component from buffer data
//         .num_format = 0,         // Unformatted buffer
//         .data_format = 4,        // 32-bit word addressing
//         .user_vm_enable = 0,     // User VM not enabled
//         .user_vm_mode = 0,       // Not applicable since user VM is disabled
//         .index_stride = stride,   // Default stride
//         .add_tid_enable = stride > 0,     // Thread ID not added to address
//         .rsvd_0 = 0,             // Reserved bits
//         .nv = 0,                 // Non-volatile is disabled
//         .rsvd_1 = 0,             // Reserved bits
//         .type = 0                // Typed buffer resource
//     }.u32);
    
//     res.x = __builtin_amdgcn_readfirstlane(res.x);
//     res.y = __builtin_amdgcn_readfirstlane(res.y);
//     res.z = __builtin_amdgcn_readfirstlane(res.z);
//     res.w = __builtin_amdgcn_readfirstlane(res.w);
//     return res;
// }

// constexpr int M = 128;
// constexpr int N = 128;
// constexpr int K = 128;

// constexpr int WMMA_M = 32;
// constexpr int WMMA_N = 32;
// constexpr int WMMA_K = 32;
// using data_type = half;

// __global__ void gemm_kernel(data_type a[M][K], data_type b[K][N], data_type c[M][N]) {
//     __shared__ data_type s_a[WMMA_M][WMMA_K];
//     __shared__ data_type s_b[WMMA_K][WMMA_N];
//      const int warp_id = __builtin_amdgcn_readfirstlane(threadIdx.x / WAVE_SIZE);
//     __builtin_assume(warp_id >= 0 && warp_id < BLOCK_SIZE / WAVE_SIZE);
//     const int lane_id = threadIdx.x % WAVE_SIZE;
       
//     {
//         auto rsrc = make_wave_buffer_resource<K>(a);
//         auto lds_ptr_sgpr = __builtin_amdgcn_readfirstlane((reinterpret_cast<uintptr_t>(&)));
//         asm volatile(R"(
//                 s_mov_b32 m0, %0; \n\t
//         )" :: "s"(lds_ptr_sgpr) : "memory");
//     }
// }

int main() {

}