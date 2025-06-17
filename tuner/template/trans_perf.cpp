#include <hip/hip_runtime.h>
#include "../../include/timer.h"
#include "../../src/utils/timer.cpp"

template <typename T>
constexpr T ceil_div(T a, T b) {
    return (a + b - 1) / b;
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

// Begin parameterization
using Elem = float /* param Elem */;
constexpr int M = 4096 /* param M */, N = 4096 /* param N */;
constexpr int TILE_DIM = 32 /* param TILE_DIM */, BLOCK_SIZE = 256 /* param BLOCK_SIZE */, VEC_SIZE = 4 /* param VEC_SIZE */;
// End parameterization

template <typename Elem, int M, int N, int TILE_DIM, int BLOCK_SIZE, int VEC_SIZE>
void launch_transpose(Elem *out, const Elem *in, hipStream_t stream = 0) {
    static_assert(TILE_DIM % VEC_SIZE == 0);
    constexpr auto TBLOCK_X = TILE_DIM / VEC_SIZE;
    static_assert(BLOCK_SIZE % TBLOCK_X == 0);
    constexpr auto TBLOCK_Y = BLOCK_SIZE / TBLOCK_X;
    static_assert(M % TILE_DIM == 0 && N % TILE_DIM == 0);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(transpose_kernel<Elem, M, N, TILE_DIM, BLOCK_SIZE, VEC_SIZE>),
        dim3(N / TILE_DIM, M / TILE_DIM), dim3(TBLOCK_X, TBLOCK_Y), 0, stream,
        out, in);
}

void launch_kernel(Elem *out, const Elem *in, hipStream_t stream) {
    launch_transpose<Elem, M, N, TILE_DIM, BLOCK_SIZE, VEC_SIZE>(out, in, stream);
}

int main() {
    Elem *d_in, *d_out;
    hipStream_t stream0;
    LIB_CALL(hipStreamCreateWithFlags(&stream0, hipStreamNonBlocking));

    LIB_CALL(hipMalloc(&d_in, M * N * sizeof(Elem)));
    LIB_CALL(hipMalloc(&d_out, M * N * sizeof(Elem)));
    LIB_CALL(hipMemsetAsync(d_in, 0, M * N * sizeof(Elem), stream0));
    LIB_CALL(hipStreamSynchronize(stream0));

    for (int k = 0; k < 3; ++k) {
        // warmup
        launch_kernel(d_out, d_in, stream0);
    }
    LIB_CALL(hipStreamSynchronize(stream0));

    hipEvent_t start, stop;
    LIB_CALL(hipEventCreate(&start));
    LIB_CALL(hipEventCreate(&stop));
    LIB_CALL(hipEventRecord(start, stream0));
    constexpr int n_iterations = 10;
    for (int k = 0; k < n_iterations; ++k) {
        launch_kernel(d_out, d_in, stream0);
    }
    LIB_CALL(hipEventRecord(stop, stream0));
    LIB_CALL(hipEventSynchronize(stop));

    float elapsedTime;
    LIB_CALL(hipEventElapsedTime(&elapsedTime, start, stop));
    
    // Calculate memory bandwidth (read + write)
    double bytes_transferred = 2.0 * M * N * sizeof(Elem) * n_iterations;
    double bandwidth_gb_s = bytes_transferred / (elapsedTime / 1000.0) / 1e9;
    
    printf("Time: %f ms\n", elapsedTime / n_iterations);
    printf("Memory Bandwidth: %.2f GB/s\n", bandwidth_gb_s / n_iterations);
    fprintf(stderr, "%f\n", (double)elapsedTime / n_iterations);

    LIB_CALL(hipEventDestroy(start));
    LIB_CALL(hipEventDestroy(stop));
    LIB_CALL(hipStreamDestroy(stream0));
    LIB_CALL(hipFree(d_in));
    LIB_CALL(hipFree(d_out));
    
    return 0;
}