#include <cstdio>
#include <hip/amd_detail/amd_warp_functions.h>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"
#include "../../include/gpu_libs.h"
#include "../../include/gpu_types.h"
#include "../../include/clangd_workaround.h"
#include <cstdlib>
#include <cfloat>

template <typename Elem, int M, int N, int TILE_DIM, int BLOCK_SIZE, int VEC_SIZE>
__device__ void transpose_kernel(Elem *odata, const Elem *idata) {
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

template <typename Elem, int BATCH_SIZE, int M, int N, int TILE_DIM, int BLOCK_SIZE, int VEC_SIZE>
__global__ void batched_transpose_kernel(Elem *odata_batch, const Elem (*idata_batch[])[N]) {
    int mat_id = blockIdx.z;
    transpose_kernel<Elem, M, N, TILE_DIM, BLOCK_SIZE, VEC_SIZE>(odata_batch + mat_id * M * N,
                                                                 reinterpret_cast<const Elem *>(idata_batch[mat_id]));
}

template <typename Elem, int M, int N, int TILE_DIM, int BLOCK_SIZE, int VEC_SIZE>
__global__ void single_transpose_kernel(Elem (*odata)[M], const Elem (*idata)[N]) {
    transpose_kernel<Elem, M, N, TILE_DIM, BLOCK_SIZE, VEC_SIZE>(reinterpret_cast<Elem *>(odata),
                                                                 reinterpret_cast<const Elem *>(idata));
}

template <typename Elem, int BATCH_SIZE, int M, int N, int TILE_DIM, int BLOCK_SIZE, int VEC_SIZE>
void launch_batched_transpose(Elem *out, const Elem (*in[])[N]) {
    static_assert(TILE_DIM % VEC_SIZE == 0);
    constexpr auto TBLOCK_X = TILE_DIM / VEC_SIZE;
    static_assert(BLOCK_SIZE % TBLOCK_X == 0);
    constexpr auto TBLOCK_Y = BLOCK_SIZE / TBLOCK_X;
    static_assert(M % TILE_DIM == 0 && N % TILE_DIM == 0);
    batched_transpose_kernel<Elem, BATCH_SIZE, M, N, TILE_DIM, BLOCK_SIZE, VEC_SIZE>
        <<<dim3(N / TILE_DIM, M / TILE_DIM, BATCH_SIZE), dim3(TBLOCK_X, TBLOCK_Y)>>>(out, in);
}

template <typename Elem, int M, int N, int TILE_DIM, int BLOCK_SIZE, int VEC_SIZE>
void launch_single_transpose(Elem (*out)[M], const Elem (*in)[N]) {
    static_assert(TILE_DIM % VEC_SIZE == 0);
    constexpr auto TBLOCK_X = TILE_DIM / VEC_SIZE;
    static_assert(BLOCK_SIZE % TBLOCK_X == 0);
    constexpr auto TBLOCK_Y = BLOCK_SIZE / TBLOCK_X;
    static_assert(M % TILE_DIM == 0 && N % TILE_DIM == 0);
    single_transpose_kernel<Elem, M, N, TILE_DIM, BLOCK_SIZE, VEC_SIZE>
        <<<dim3(N / TILE_DIM, M / TILE_DIM), dim3(TBLOCK_X, TBLOCK_Y)>>>(out, in);
}