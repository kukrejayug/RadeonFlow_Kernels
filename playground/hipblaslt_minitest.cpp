#include "../include/gpu_libs.h"
#include "../include/gpu_types.h"
#include "../include/clangd_workaround.h"
#include <array>
#include <cstddef>

#include <hip/hip_runtime.h>
#include <hipblas-common/hipblas-common.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>

#define CHECK_HIPBLASLT_ERROR(...)                                                                                     \
    do {                                                                                                               \
        auto __status = __VA_ARGS__;                                                                                   \
        if (__status != HIPBLAS_STATUS_SUCCESS) {                                                                      \
            std::cerr << "HIPBLASLT error at " << __FILE__ << ":" << __LINE__ << std::endl;                            \
            exit(-1);                                                                                                  \
        }                                                                                                              \
    } while (0)

hipblasLtHandle_t hipblasLtHandle;
hipblaslt_ext::GemmPreferenceV2 gemmPref;
void *d_workspace = nullptr;
float *d_alpha;
float *d_beta;

template <int M, int N, int K, int n_expert>
hipblasStatus_t LaunchGroupGEMM(const half *d_A_, half *d_B_, float *d_C_, const int *h_IDX_, hipStream_t stream = 0) {

    float alpha = 1.0f;
    float beta = 0.0f;
    LIB_CALL(hipMalloc(&d_alpha, sizeof(float)));
    LIB_CALL(hipMalloc(&d_beta, sizeof(float)));
    LIB_CALL(hipMemcpy(d_alpha, &alpha, sizeof(float), hipMemcpyHostToDevice));
    LIB_CALL(hipMemcpy(d_beta, &beta, sizeof(float), hipMemcpyHostToDevice));

    size_t max_workspace_size = 512ll * 1024 * 1024; // 512 MB
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&hipblasLtHandle));
    gemmPref.setMaxWorkspaceBytes(max_workspace_size);
    LIB_CALL(hipMalloc(&d_workspace, max_workspace_size));

    half *d_A, *d_B;
    float *d_C;

    LIB_CALL(hipMalloc(&d_A, M * K * sizeof(half)));
    LIB_CALL(hipMalloc(&d_B, N * K * n_expert * sizeof(half)));
    LIB_CALL(hipMalloc(&d_C, M * N * sizeof(float)));
    // LIB_CALL(hipMemset(d_A, 0, M * K * sizeof(half)));
    // LIB_CALL(hipMemset(d_B, 0, N * K * n_expert * sizeof(half)));
    // LIB_CALL(hipMemset(d_C, 0, M * N * sizeof(float)));

    hipblaslt_ext::GroupedGemm grouped_gemm(hipblasLtHandle, HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_16F, HIP_R_16F,
                                            HIP_R_32F, HIP_R_32F, HIPBLAS_COMPUTE_32F);
    std::vector<int64_t> m(n_expert);
    std::vector<int64_t> n(n_expert, N);
    std::vector<int64_t> k(n_expert, K);
    std::vector<int64_t> batch_count(n_expert, 1);
    std::vector<hipblaslt_ext::GemmInputsV2> inputs(n_expert);
    std::vector<hipblaslt_ext::GemmEpilogueV2> epilogue(n_expert);
    const int h_IDX[] = {0,     1026,  2017,  3044,  4031,  5123,  6151,  7175,  8151,  9153,  10173,
                         11225, 12255, 13236, 14221, 15274, 16343, 17384, 18428, 19438, 20458, 21457,
                         22520, 23580, 24578, 25603, 26601, 27673, 28661, 29730, 30796, 31774, 32768};
    for (int i = 0; i < n_expert; i++) {
        m[i] = h_IDX[i + 1] - h_IDX[i];
        inputs[i].setB(d_A + h_IDX[i] * K);
        inputs[i].setA(d_B + i * N * K);
        inputs[i].setC(d_C + h_IDX[i] * N);
        inputs[i].setD(d_C + h_IDX[i] * N);
        inputs[i].setAlpha(d_alpha);
        inputs[i].setBeta(d_beta);
        epilogue[i].setMode(HIPBLASLT_EPILOGUE_DEFAULT);
    }
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults;
    grouped_gemm.setProblem(n, m, k, batch_count, epilogue, inputs);
    grouped_gemm.algoGetHeuristic(1, gemmPref, heuristicResults);
    if (heuristicResults.size() == 0) {
        std::cout << "No heuristic results found" << std::endl;
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }
    auto algo = heuristicResults[0].algo;
    CHECK_HIPBLASLT_ERROR(grouped_gemm.initialize(algo, d_workspace, false, stream));
    return grouped_gemm.run(stream);
}

int main() {
    LaunchGroupGEMM<8192 * 4, 7168, 2048, 32>(nullptr, nullptr, nullptr, nullptr);
}