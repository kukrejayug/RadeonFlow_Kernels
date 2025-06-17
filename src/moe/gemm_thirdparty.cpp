#include <hipblas-common/hipblas-common.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include "gpu_libs.h"
struct ExpertWeights {
    void *ptr[64];
};

#define CHECK_HIPBLASLT_ERROR(...)                                                                                     \
    do {                                                                                                               \
        auto __status = __VA_ARGS__;                                                                                   \
        if (__status != HIPBLAS_STATUS_SUCCESS) {                                                                      \
            std::cerr << "HIPBLASLT error at " << __FILE__ << ":" << __LINE__ << std::endl;                            \
            exit(-1);                                                                                                  \
        }                                                                                                              \
    } while (0)

template <int M, int N, int K, bool GROUPED> constexpr int best_algo_index() {

    // for benchmark case 1
    if (M == 8192 * 4 && N == 2048 && K == 7168 && GROUPED)
        return 0;
    if (M == 8192 * 4 && N == 7168 && K == 2048 && GROUPED)
        return 1;
    if (M == 8192 && N == 2048 && K == 7168 && !GROUPED)
        return 2;
    if (M == 8192 && N == 7168 && K == 2048 && !GROUPED)
        return 3;
    if (M == 8192 && N == 32 && K == 7168 && !GROUPED)
        return 8;

    // for benchmark case 0
    if (M == 8192 && N == 2048 && K == 7168 && GROUPED)
        return 4;
    if (M == 8192 && N == 7168 && K == 2048 && GROUPED)
        return 5;
    if (M == 2048 && N == 2048 && K == 7168 && !GROUPED)
        return 6;
    if (M == 2048 && N == 7168 && K == 2048 && !GROUPED)
        return 7;
    if (M == 2048 && N == 32 && K == 7168 && !GROUPED)
        return 9;

    // std::cout<< "Untuned hipBLAS param! Low performance expected!" << std::endl;

    if (GROUPED)
        return 0;
    else if (N <= 32)
        return 9;
    else
        return 2;
}

struct GemmThirdParty {
    hipblasLtHandle_t hipblasLtHandle;
    hipblaslt_ext::GemmPreferenceV2 gemmPref;
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristics;
    void *d_workspace = nullptr;
    float *d_alpha;
    float *d_beta;
};

template <bool CONT_LAYOUT, bool TRANSPOSED, bool SECOND_C, int M, int N, int K, int n_expert, typename out_data_t>
hipblasStatus_t LaunchGroupGEMM(GemmThirdParty &gemm_thirdparty, const half *d_A, half *d_B,
                                const ExpertWeights &expert_weights, out_data_t *d_C, out_data_t *d_C_second, const int *h_IDX,
                                int algo_idx = 0, hipStream_t stream = 0) {
    auto hip_out_data_t = std::is_same_v<out_data_t, float> ? HIP_R_32F : HIP_R_16F;
    auto hip_op_b = TRANSPOSED ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblaslt_ext::GroupedGemm grouped_gemm(gemm_thirdparty.hipblasLtHandle, hip_op_b, HIPBLAS_OP_N, HIP_R_16F,
                                            HIP_R_16F, hip_out_data_t, hip_out_data_t, HIPBLAS_COMPUTE_32F);
    std::vector<int64_t> m(SECOND_C ? 2 * n_expert : n_expert);
    std::vector<int64_t> n(SECOND_C ? 2 * n_expert : n_expert, N);
    std::vector<int64_t> k(SECOND_C ? 2 * n_expert : n_expert, K);
    std::vector<int64_t> batch_count(SECOND_C ? 2 * n_expert : n_expert, 1);
    std::vector<hipblaslt_ext::GemmInputsV2> inputs(SECOND_C ? 2 * n_expert : n_expert);
    std::vector<hipblaslt_ext::GemmEpilogueV2> epilogue(SECOND_C ? 2 * n_expert : n_expert);
    for (int i = 0; i < n_expert; i++) {
        m[i] = h_IDX[i + 1] - h_IDX[i];
        inputs[i].setB(d_A + h_IDX[i] * K);
        if constexpr (CONT_LAYOUT) {
            inputs[i].setA(d_B + i * N * K);
        } else {
            inputs[i].setA(expert_weights.ptr[i]);
        }
        inputs[i].setC(d_C + h_IDX[i] * N);
        inputs[i].setD(d_C + h_IDX[i] * N);
        inputs[i].setAlpha(gemm_thirdparty.d_alpha);
        inputs[i].setBeta(gemm_thirdparty.d_beta);
        epilogue[i].setMode(HIPBLASLT_EPILOGUE_DEFAULT);
    }

    if constexpr(SECOND_C) {
        for (int i = 0; i < n_expert; i++) {
            m[i + n_expert] = h_IDX[i + 1] - h_IDX[i];
            inputs[i + n_expert].setB(d_A + h_IDX[i] * K);
            if constexpr (CONT_LAYOUT) {
                static_assert(false, "CONT_LAYOUT is not supported for second C");
            } else {
                inputs[i + n_expert].setA(expert_weights.ptr[i + n_expert]);
            }
            inputs[i + n_expert].setC(d_C_second + h_IDX[i] * N);
            inputs[i + n_expert].setD(d_C_second + h_IDX[i] * N);
            inputs[i + n_expert].setAlpha(gemm_thirdparty.d_alpha);
            inputs[i + n_expert].setBeta(gemm_thirdparty.d_beta);
            epilogue[i + n_expert].setMode(HIPBLASLT_EPILOGUE_DEFAULT);
        }
    }
    grouped_gemm.setProblem(n, m, k, batch_count, epilogue, inputs);

    CHECK_HIPBLASLT_ERROR(
        grouped_gemm.initialize(gemm_thirdparty.heuristics[algo_idx].algo, gemm_thirdparty.d_workspace, false, stream));
    return grouped_gemm.run(stream);
}

template <bool CONT_LAYOUT, bool TRANSPOSED, int M, int N, int K, int n_expert, typename out_data_t>
hipblasStatus_t LaunchGroupGEMMBench(GemmThirdParty &gemm_thirdparty, const half *d_A, half *d_B,
                                     const half (*d_B_non_cont[])[N], out_data_t *d_C, const int *h_IDX,
                                     int algo_idx__ = 0, hipStream_t stream = 0) {
    LIB_CALL(hipDeviceSynchronize());
    auto hip_out_data_t = std::is_same_v<out_data_t, float> ? HIP_R_32F : HIP_R_16F;
    auto hip_op_b = TRANSPOSED ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblaslt_ext::GroupedGemm grouped_gemm(gemm_thirdparty.hipblasLtHandle, hip_op_b, HIPBLAS_OP_N, HIP_R_16F,
                                            HIP_R_16F, hip_out_data_t, hip_out_data_t, HIPBLAS_COMPUTE_32F);

    std::vector<int64_t> m(n_expert);
    std::vector<int64_t> n(n_expert, N);
    std::vector<int64_t> k(n_expert, K);
    std::vector<int64_t> batch_count(n_expert, 1);
    std::vector<hipblaslt_ext::GemmInputsV2> inputs(n_expert);
    std::vector<hipblaslt_ext::GemmEpilogueV2> epilogue(n_expert);
    for (int i = 0; i < n_expert; i++) {
        m[i] = h_IDX[i + 1] - h_IDX[i];

        inputs[i].setB(d_A + h_IDX[i] * K);
        if constexpr (CONT_LAYOUT) {
            inputs[i].setA(d_B + i * N * K);
        } else {
            inputs[i].setA(d_B_non_cont[i]);
        }
        inputs[i].setC(d_C + h_IDX[i] * N);
        inputs[i].setD(d_C + h_IDX[i] * N);
        inputs[i].setAlpha(gemm_thirdparty.d_alpha);
        inputs[i].setBeta(gemm_thirdparty.d_beta);
        epilogue[i].setMode(HIPBLASLT_EPILOGUE_DEFAULT);
    }
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults;
    grouped_gemm.setProblem(n, m, k, batch_count, epilogue, inputs);
    CHECK_HIPBLASLT_ERROR(hipblaslt_ext::getAllAlgos(
        gemm_thirdparty.hipblasLtHandle, hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM, hip_op_b, HIPBLAS_OP_N,
        HIP_R_16F, HIP_R_16F, hip_out_data_t, hip_out_data_t, HIPBLAS_COMPUTE_32F, heuristicResults));

    // std::cout << "heuristicResults.size(): " << heuristicResults.size() << std::endl;

    hipEvent_t start, stop;
    LIB_CALL(hipEventCreate(&start));
    LIB_CALL(hipEventCreate(&stop));
    float min_time = std::numeric_limits<float>::max();
    int best_algo_index = -1;
    int best_algo_raw_index = -1;

    for (size_t i = 0; i < heuristicResults.size(); i++) {
        auto &result = heuristicResults[i];
        float min_iter_time = std::numeric_limits<float>::max();
        for (int iter = 0; iter < 10; iter++) {
            CHECK_HIPBLASLT_ERROR(grouped_gemm.initialize(result.algo, gemm_thirdparty.d_workspace, false, stream));
            LIB_CALL(hipEventRecord(start, stream));
            CHECK_HIPBLASLT_ERROR(grouped_gemm.run(stream));
            LIB_CALL(hipEventRecord(stop, stream));
            LIB_CALL(hipEventSynchronize(stop));
            float time;
            LIB_CALL(hipEventElapsedTime(&time, start, stop));
            min_iter_time = std::min(min_iter_time, time);
        }
        int algo_index = hipblaslt_ext::getIndexFromAlgo(result.algo);
        int algo_raw_index = i;
        // std::cout << "Algo index: " << algo_index << " min time: " << min_iter_time << " ms" << std::endl;

        if (min_iter_time < min_time) {
            min_time = min_iter_time;
            best_algo_index = algo_index;
            best_algo_raw_index = algo_raw_index;
        }
    }

    LIB_CALL(hipEventDestroy(start));
    LIB_CALL(hipEventDestroy(stop));

    std::cout << "Best algo index: " << best_algo_index << " with time: " << min_time << " ms" << std::endl;

    auto best_algo = heuristicResults[best_algo_raw_index].algo;
    CHECK_HIPBLASLT_ERROR(grouped_gemm.initialize(best_algo, gemm_thirdparty.d_workspace, false, stream));
    return grouped_gemm.run(stream);
}

template <int DUMMY = 1> void initialize_gemm_thirdparty(GemmThirdParty &gemm_thirdparty) {
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults;

    size_t max_workspace_size = 512ll * 1024 * 1024; // 512 MB
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&gemm_thirdparty.hipblasLtHandle));
    gemm_thirdparty.gemmPref.setMaxWorkspaceBytes(max_workspace_size);
    LIB_CALL(hipMalloc(&gemm_thirdparty.d_workspace, max_workspace_size));
    LIB_CALL(hipMalloc(&gemm_thirdparty.d_alpha, sizeof(float)));
    LIB_CALL(hipMalloc(&gemm_thirdparty.d_beta, sizeof(float)));
    float alpha = 1.0f;
    float beta = 0.0f;
    LIB_CALL(hipMemcpy(gemm_thirdparty.d_alpha, &alpha, sizeof(float), hipMemcpyHostToDevice));
    LIB_CALL(hipMemcpy(gemm_thirdparty.d_beta, &beta, sizeof(float), hipMemcpyHostToDevice));

    std::vector<int> algo_index = {166966};
    hipblaslt_ext::getAlgosFromIndex(gemm_thirdparty.hipblasLtHandle, algo_index, heuristicResults);
    gemm_thirdparty.heuristics.push_back(heuristicResults[0]);

    algo_index = {167403};
    heuristicResults.clear();
    hipblaslt_ext::getAlgosFromIndex(gemm_thirdparty.hipblasLtHandle, algo_index, heuristicResults);
    gemm_thirdparty.heuristics.push_back(heuristicResults[0]);

    algo_index = {177690};
    heuristicResults.clear();
    hipblaslt_ext::getAlgosFromIndex(gemm_thirdparty.hipblasLtHandle, algo_index, heuristicResults);
    gemm_thirdparty.heuristics.push_back(heuristicResults[0]);

    algo_index = {195760};
    heuristicResults.clear();
    hipblaslt_ext::getAlgosFromIndex(gemm_thirdparty.hipblasLtHandle, algo_index, heuristicResults);
    gemm_thirdparty.heuristics.push_back(heuristicResults[0]);

    algo_index = {167317};
    heuristicResults.clear();
    hipblaslt_ext::getAlgosFromIndex(gemm_thirdparty.hipblasLtHandle, algo_index, heuristicResults);
    gemm_thirdparty.heuristics.push_back(heuristicResults[0]);

    algo_index = {167317};
    heuristicResults.clear();
    hipblaslt_ext::getAlgosFromIndex(gemm_thirdparty.hipblasLtHandle, algo_index, heuristicResults);
    gemm_thirdparty.heuristics.push_back(heuristicResults[0]);

    algo_index = {177678};
    heuristicResults.clear();
    hipblaslt_ext::getAlgosFromIndex(gemm_thirdparty.hipblasLtHandle, algo_index, heuristicResults);
    gemm_thirdparty.heuristics.push_back(heuristicResults[0]);

    algo_index = {177681};
    heuristicResults.clear();
    hipblaslt_ext::getAlgosFromIndex(gemm_thirdparty.hipblasLtHandle, algo_index, heuristicResults);
    gemm_thirdparty.heuristics.push_back(heuristicResults[0]);

    algo_index = {214978};
    heuristicResults.clear();
    hipblaslt_ext::getAlgosFromIndex(gemm_thirdparty.hipblasLtHandle, algo_index, heuristicResults);
    gemm_thirdparty.heuristics.push_back(heuristicResults[0]);

    algo_index = {213349};
    heuristicResults.clear();
    hipblaslt_ext::getAlgosFromIndex(gemm_thirdparty.hipblasLtHandle, algo_index, heuristicResults);
    gemm_thirdparty.heuristics.push_back(heuristicResults[0]);
}

template <bool TRANSPOSED, int M, int N, int K, typename out_data_t>
hipblasStatus_t LaunchGEMM(GemmThirdParty &gemm_thirdparty, const half *d_A, const half *d_B, out_data_t *d_C,
                           int algo_idx = 0, hipStream_t stream = 0) {
    auto hip_out_data_t = std::is_same_v<out_data_t, float> ? HIP_R_32F : HIP_R_16F;
    auto hip_op_a = HIPBLAS_OP_N;
    auto hip_op_b = TRANSPOSED ? HIPBLAS_OP_T : HIPBLAS_OP_N;

    hipblaslt_ext::Gemm gemm(gemm_thirdparty.hipblasLtHandle, hip_op_b, hip_op_a, HIP_R_16F, HIP_R_16F, hip_out_data_t,
                             hip_out_data_t, HIPBLAS_COMPUTE_32F);

    hipblaslt_ext::GemmEpilogueV2 epilogue;
    hipblaslt_ext::GemmInputsV2 inputs;

    epilogue.setMode(HIPBLASLT_EPILOGUE_DEFAULT);

    inputs.setA(d_B);
    inputs.setB(d_A);
    inputs.setC(d_C);
    inputs.setD(d_C);
    inputs.setAlpha(gemm_thirdparty.d_alpha);
    inputs.setBeta(gemm_thirdparty.d_beta);

    gemm.setProblem(N, M, K, 1, epilogue, inputs);

    CHECK_HIPBLASLT_ERROR(
        gemm.initialize(gemm_thirdparty.heuristics[algo_idx].algo, gemm_thirdparty.d_workspace, false, stream));
    return gemm.run(stream);
}

template <bool TRANSPOSED, int M, int N, int K, typename out_data_t>
hipblasStatus_t LaunchGEMMBench(GemmThirdParty &gemm_thirdparty, const half *d_A, const half *d_B, out_data_t *d_C,
                                int algo_idx__ = 0, hipStream_t stream = 0) {

    LIB_CALL(hipDeviceSynchronize());
    auto hip_out_data_t = std::is_same_v<out_data_t, float> ? HIP_R_32F : HIP_R_16F;
    auto hip_op_a = HIPBLAS_OP_N;
    auto hip_op_b = TRANSPOSED ? HIPBLAS_OP_T : HIPBLAS_OP_N;

    hipblaslt_ext::Gemm gemm(gemm_thirdparty.hipblasLtHandle, hip_op_b, hip_op_a, HIP_R_16F, HIP_R_16F, hip_out_data_t,
                             hip_out_data_t, HIPBLAS_COMPUTE_32F);

    hipblaslt_ext::GemmEpilogueV2 epilogue;
    hipblaslt_ext::GemmInputsV2 inputs;

    epilogue.setMode(HIPBLASLT_EPILOGUE_DEFAULT);

    inputs.setA(d_B);
    inputs.setB(d_A);
    inputs.setC(d_C);
    inputs.setD(d_C);
    inputs.setAlpha(gemm_thirdparty.d_alpha);
    inputs.setBeta(gemm_thirdparty.d_beta);

    gemm.setProblem(N, M, K, 1, epilogue, inputs);

    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults;
    CHECK_HIPBLASLT_ERROR(hipblaslt_ext::getAllAlgos(
        gemm_thirdparty.hipblasLtHandle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM, hip_op_b, hip_op_a, HIP_R_16F,
        HIP_R_16F, hip_out_data_t, hip_out_data_t, HIPBLAS_COMPUTE_32F, heuristicResults));

    if (heuristicResults.empty()) {
        std::cerr << "No suitable algorithms found for GEMM operation!" << std::endl;
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    std::cout << "heuristicResults.size(): " << heuristicResults.size() << std::endl;

    hipEvent_t start, stop;
    LIB_CALL(hipEventCreate(&start));
    LIB_CALL(hipEventCreate(&stop));
    float min_time = std::numeric_limits<float>::max();
    int best_algo_index = -1;
    int best_algo_raw_index = -1;

    for (size_t i = 0; i < heuristicResults.size(); i++) {
        auto &result = heuristicResults[i];

        size_t workspaceSizeInBytes = 0;
        if (gemm.isAlgoSupported(result.algo, workspaceSizeInBytes) != HIPBLAS_STATUS_SUCCESS) {
            // std::cout << "Algorithm " << i << " does not support the problem" << std::endl;
            continue;
        }

        if (workspaceSizeInBytes > gemm_thirdparty.gemmPref.getMaxWorkspaceBytes()) {
            // std::cout << "Algorithm " << i << " requires too much workspace: " << workspaceSizeInBytes << " bytes"
            //   << std::endl;
            continue;
        }

        float min_iter_time = std::numeric_limits<float>::max();
        for (int iter = 0; iter < 10; iter++) {
            CHECK_HIPBLASLT_ERROR(gemm.initialize(result.algo, gemm_thirdparty.d_workspace, false, stream));
            LIB_CALL(hipEventRecord(start, stream));
            CHECK_HIPBLASLT_ERROR(gemm.run(stream));
            LIB_CALL(hipEventRecord(stop, stream));
            LIB_CALL(hipEventSynchronize(stop));
            float time;
            LIB_CALL(hipEventElapsedTime(&time, start, stop));
            min_iter_time = std::min(min_iter_time, time);
        }
        int algo_index = hipblaslt_ext::getIndexFromAlgo(result.algo);
        int algo_raw_index = i;
        // std::cout << "Algo index: " << algo_index << " min time: " << min_iter_time << " ms" << std::endl;

        if (min_iter_time < min_time) {
            min_time = min_iter_time;
            best_algo_index = algo_index;
            best_algo_raw_index = algo_raw_index;
        }
    }

    LIB_CALL(hipEventDestroy(start));
    LIB_CALL(hipEventDestroy(stop));

    std::cout << "Best algo index: " << best_algo_index << " with time: " << min_time << " ms" << std::endl;

    auto best_algo = heuristicResults[best_algo_raw_index].algo;
    CHECK_HIPBLASLT_ERROR(gemm.initialize(best_algo, gemm_thirdparty.d_workspace, false, stream));
    return gemm.run(stream);
}