#include <hip/hip_runtime.h>
#include "../../src/gemm/gemm_launcher.cpp"
#include "../../include/timer.h"
#include "../../src/utils/timer.cpp"

using in_data_type = __FP8_TYPE;
using out_data_type = __hip_bfloat16;
using acc_data_type = float;
// Begin parameterization
constexpr int M = 1024 /* param M*/, N = 1536 /* param N*/ , K = 7168 /* param K */;
constexpr int BM = 256 /* param BM */, BN = 128 /* param BN */, BK = 32 /* param BK */;
constexpr int QUANT_SIZE = 128 /* param QUANT_SIZE */, BLOCK_SIZE = 256 /* param BLOCK_SIZE */;
constexpr int WARP_M = 2 /* param WARP_M */, WARP_N = 2 /* param WARP_N */;
constexpr int SPLITK_FACTOR = 2 /* param SPLITK_FACTOR */;
constexpr int LOAD_BATCH_SIZE = 16 /* param LOAD_BATCH_SIZE */;
// End parameterization

constexpr int QM = M;
constexpr int QN = ceil_div(N, QUANT_SIZE);
constexpr int QK = ceil_div(K, QUANT_SIZE);
void launch_kernel(
    in_data_type (*a)[M], 
    in_data_type (*b)[N], 
    out_data_type (*c)[N], 
    float (*as)[QM], 
    float (*bs)[QN],
    float *c_splitk,
    hipStream_t stream
) {
    launch_gemm<M, N, K, BM, BN, BK, WARP_M, WARP_N, BLOCK_SIZE, QUANT_SIZE, SPLITK_FACTOR, LOAD_BATCH_SIZE>(
        reinterpret_cast<const in_data_type*>(a), 
        reinterpret_cast<const in_data_type*>(b), 
        reinterpret_cast<out_data_type*>(c), 
        reinterpret_cast<const float*>(as), 
        reinterpret_cast<const float*>(bs),
        stream
    );
}

int main() {
    in_data_type (*d_a)[M];
    in_data_type (*d_b)[N];
    float (*d_sa)[QM];
    float (*d_sb)[QN];
    out_data_type (*d_c)[N];
    float *d_c_splitk;
    hipStream_t stream0;
    LIB_CALL(hipStreamCreateWithFlags(&stream0, hipStreamNonBlocking));

    LIB_CALL(hipMalloc(&d_a, K * M * sizeof(in_data_type)));
    LIB_CALL(hipMalloc(&d_b, K * N * sizeof(in_data_type)));
    LIB_CALL(hipMalloc(&d_c, M * N * sizeof(out_data_type)));
    LIB_CALL(hipMalloc(&d_sa, QK * QM * sizeof(float)));
    LIB_CALL(hipMalloc(&d_sb, QK * QN * sizeof(float)));
    LIB_CALL(hipMalloc(&d_c_splitk, SPLITK_FACTOR * M * N * sizeof(float)));
    LIB_CALL(hipMemsetAsync(d_a, 0, K * M * sizeof(in_data_type), stream0));
    LIB_CALL(hipMemsetAsync(d_b, 0, K * N * sizeof(in_data_type), stream0));
    LIB_CALL(hipMemsetAsync(d_sa, 0, QK * QM * sizeof(float), stream0));
    LIB_CALL(hipMemsetAsync(d_sb, 0, QK * QN * sizeof(float), stream0));
    LIB_CALL(hipStreamSynchronize(stream0));
    init_workspace();
    for (int k = 0; k < 3; ++k) {
        // warmup
        launch_kernel(d_a, d_b, d_c, d_sa, d_sb, d_c_splitk, stream0);
    }
    LIB_CALL(hipStreamSynchronize(stream0));
    hipEvent_t start, stop;
    LIB_CALL(hipEventCreate(&start));
    LIB_CALL(hipEventCreate(&stop));
    LIB_CALL(hipEventRecord(start, stream0));
    constexpr int n_iterations = 10;
    for (int k = 0; k < n_iterations; ++k) {
        launch_kernel(d_a, d_b, d_c, d_sa, d_sb, d_c_splitk, stream0);
    }
    LIB_CALL(hipEventRecord(stop, stream0));
    LIB_CALL(hipEventSynchronize(stop));
    float elapsedTime;
    LIB_CALL(hipEventElapsedTime(&elapsedTime, start, stop));
    printf("Time: %f ms\n", elapsedTime / n_iterations);
    printf("Throughput: %.2f TFLOPS\n", 2.0 * M * N * K / (elapsedTime / n_iterations) / 1e9);
    fprintf(stderr, "%f\n", (double)elapsedTime / 10);
    LIB_CALL(hipEventDestroy(start));
    LIB_CALL(hipEventDestroy(stop));
    LIB_CALL(hipStreamDestroy(stream0));
    LIB_CALL(hipFree(d_a));
    LIB_CALL(hipFree(d_b));
    LIB_CALL(hipFree(d_c));
    LIB_CALL(hipFree(d_sa));
    LIB_CALL(hipFree(d_sb));
    
}
