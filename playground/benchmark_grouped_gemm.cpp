#include "../include/gpu_libs.h"
#include "../include/gpu_types.h"
#include "../include/clangd_workaround.h"
#include <array>
#include <cstddef>

// Define this macro to enable LibTorch tests
// #define ENABLE_TORCH_TESTS

#ifdef ENABLE_TORCH_TESTS
#include <ATen/core/TensorBody.h>
#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPGuard.h>
#include "torch/torch.h"
#endif

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

#define CHECK_HIP_ERROR(func)                                                                                          \
    do {                                                                                                               \
        auto __status = func;                                                                                          \
        if (__status != hipSuccess) {                                                                                  \
            std::cerr << "HIP error: " << hipGetErrorString(__status) << " at " << __FILE__ << ":" << __LINE__         \
                      << std::endl;                                                                                    \
            exit(-1);                                                                                                  \
        }                                                                                                              \
    } while (0)

#define CHECK_HIPBLASLT_ERROR(...)                                                                                     \
    do {                                                                                                               \
        auto __status = __VA_ARGS__;                                                                                   \
        if (__status != HIPBLAS_STATUS_SUCCESS) {                                                                      \
            std::cerr << "HIPBLASLT error at " << __FILE__ << ":" << __LINE__ << std::endl;                            \
            exit(-1);                                                                                                  \
        }                                                                                                              \
    } while (0)

HOST_CODE_BELOW

// Global constants for matrix dimensions
constexpr int MATRIX_M = 8192 * 4;
constexpr int MATRIX_N = 7168;
constexpr int MATRIX_K = 2048;
constexpr int NUM_EXPERTS = 32;

// constexpr int MATRIX_M = 1024;
// constexpr int MATRIX_N = 1024;
// constexpr int MATRIX_K = 1024;
// constexpr int NUM_EXPERTS = 2;

// constexpr int MATRIX_M = 4;
// constexpr int MATRIX_N = 4;
// constexpr int MATRIX_K = 4;
// constexpr int NUM_EXPERTS = 1;

hipblasLtHandle_t hipblasLtHandle;
hipblaslt_ext::GemmPreferenceV2 gemmPref;
void *d_workspace = nullptr;
float *d_alpha;
float *d_beta;

void InitWorkspace() {
    if (d_workspace)
        return;                                       // Already initialized
    size_t max_workspace_size = 8192ll * 1024 * 1024; // 512 MB
    CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&hipblasLtHandle));
    gemmPref.setMaxWorkspaceBytes(max_workspace_size);
    CHECK_HIP_ERROR(hipMalloc(&d_workspace, max_workspace_size));
    CHECK_HIP_ERROR(hipMalloc(&d_alpha, sizeof(float)));
    CHECK_HIP_ERROR(hipMalloc(&d_beta, sizeof(float)));
    float alpha = 1.0f;
    float beta = 0.0f;
    CHECK_HIP_ERROR(hipMemcpy(d_alpha, &alpha, sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_beta, &beta, sizeof(float), hipMemcpyHostToDevice));
}

template <int M, int N, int K, int n_expert>
hipblasStatus_t LaunchGroupGEMM(const half *d_A, const half *d_B, float *d_C,
                                const std::array<size_t, n_expert + 1> &h_IDX, hipblasLtMatmulAlgo_t algo,
                                hipStream_t stream = 0) {
    hipblaslt_ext::GroupedGemm grouped_gemm(hipblasLtHandle, HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_16F, HIP_R_16F,
                                            HIP_R_32F, HIP_R_32F, HIPBLAS_COMPUTE_32F);
    std::vector<int64_t> m(n_expert);
    std::vector<int64_t> n(n_expert, N);
    std::vector<int64_t> k(n_expert, K);
    std::vector<int64_t> batch_count(n_expert, 1);
    std::vector<hipblaslt_ext::GemmInputsV2> inputs(n_expert);
    std::vector<hipblaslt_ext::GemmEpilogueV2> epilogue(n_expert);
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
    grouped_gemm.setProblem(n, m, k, batch_count, epilogue, inputs);
    CHECK_HIPBLASLT_ERROR(grouped_gemm.initialize(algo, d_workspace, false, stream));
    return grouped_gemm.run(stream);
}

template <int M, int N, int K>
hipblasStatus_t LaunchBatchedGEMM(const half *d_A, const half *d_B, float *d_C, hipblasLtMatmulAlgo_t algo,
                                  hipStream_t stream = 0) {

    static_assert(M == 1024 + 128, "M must be 1024 + 128");

    hipblaslt_ext::Gemm gemm(hipblasLtHandle, HIPBLAS_OP_N, HIPBLAS_OP_N, HIP_R_16F, HIP_R_16F, HIP_R_32F, HIP_R_32F,
                             HIPBLAS_COMPUTE_32F);

    hipblaslt_ext::GemmEpilogueV2 epilogue;
    hipblaslt_ext::GemmInputsV2 inputs;
    hipblaslt_ext::GemmProblemTypeV2 problem_type(HIPBLAS_OP_N, HIPBLAS_OP_N, HIP_R_16F, HIP_R_16F, HIP_R_32F,
                                                  HIP_R_32F, HIPBLAS_COMPUTE_32F);

    epilogue.setMode(HIPBLASLT_EPILOGUE_DEFAULT);

    inputs.setA(d_B);
    inputs.setB(d_A);
    inputs.setC(d_C);
    inputs.setD(d_C);
    inputs.setAlpha(d_alpha);
    inputs.setBeta(d_beta);

    gemm.setProblem(N, M, K, NUM_EXPERTS, epilogue, inputs);

    size_t workspaceSizeInBytes = 512ll*1024*1024;
    if (gemm.isAlgoSupported(algo, workspaceSizeInBytes) != HIPBLAS_STATUS_SUCCESS) {
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    if (workspaceSizeInBytes > gemmPref.getMaxWorkspaceBytes()) {
        std::cout << "Algorithm " << " requires too much workspace: " << workspaceSizeInBytes << " bytes"
                  << std::endl;
        return HIPBLAS_STATUS_NOT_SUPPORTED;
    }

    CHECK_HIPBLASLT_ERROR(gemm.initialize(algo, d_workspace, false, stream));
    return gemm.run(stream);
}

#ifdef ENABLE_TORCH_TESTS
template <int M, int N, int K, int n_expert>
void LaunchTorchRef(const half *d_A, // [M, K]
                    const half *d_B, // [L, N, K]
                    float *d_C,      // [M, N]
                    const std::array<size_t, n_expert + 1> &h_IDX, hipStream_t stream = 0) {
    torch::Tensor A = torch::from_blob((void *)d_A, {M, K}, torch::dtype(torch::kFloat16).device(torch::kCUDA));
    torch::Tensor B =
        torch::from_blob((void *)d_B, {n_expert, N, K}, torch::dtype(torch::kFloat16).device(torch::kCUDA));
    torch::Tensor C = torch::from_blob((void *)d_C, {M, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    // at::Stream torch_stream(at::Stream::UNSAFE, torch::kHIP, reinterpret_cast<int64_t>(stream));
    // at::hip::HIPStreamGuard guard(torch_stream);
    for (int i = 0; i < n_expert; i++) {
        int start_idx = h_IDX[i];
        int end_idx = h_IDX[i + 1];
        int rows = end_idx - start_idx;
        auto A_slice = A.slice(0, start_idx, end_idx);                                  // [rows, K]
        auto B_slice = B[i];                                                            // [K, N]
        auto C_slice = torch::mm(A_slice, B_slice.transpose(0, 1)).to(torch::kFloat32); // [rows, N]
        C.slice(0, start_idx, end_idx).copy_(C_slice);
    }
}
#endif

bool TestGroupGEMM() {
    InitWorkspace();
    std::array<size_t, NUM_EXPERTS + 1> h_IDX = {0,     1026,  2017,  3044,  4031,  5123,  6151,  7175,  8151,
                                                 9153,  10173, 11225, 12255, 13236, 14221, 15274, 16343, 17384,
                                                 18428, 19438, 20458, 21457, 22520, 23580, 24578, 25603, 26601,
                                                 27673, 28661, 29730, 30796, 31774, 32768};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    std::vector<half> h_A(MATRIX_K * MATRIX_M);
    std::vector<half> h_B(NUM_EXPERTS * MATRIX_K * MATRIX_N);
    std::vector<half> h_C(MATRIX_M * MATRIX_N);
    for (auto &v : h_A)
        v = __float2half(dist(gen));
    for (auto &v : h_B)
        v = __float2half(dist(gen));

    half *d_A, *d_B;
    float *d_C;
    CHECK_HIP_ERROR(hipMalloc(&d_A, h_A.size() * sizeof(half)));
    CHECK_HIP_ERROR(hipMalloc(&d_B, h_B.size() * sizeof(half)));
    CHECK_HIP_ERROR(hipMalloc(&d_C, h_C.size() * sizeof(float)));
    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A.data(), h_A.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, h_B.data(), h_B.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(d_C, 0, h_C.size() * sizeof(float)));
    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_results;
    CHECK_HIPBLASLT_ERROR(hipblaslt_ext::getAllAlgos(hipblasLtHandle, hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM,
                                                     HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_16F, HIP_R_16F, HIP_R_32F,
                                                     HIP_R_32F, HIPBLAS_COMPUTE_32F, heuristic_results));
    if (heuristic_results.empty()) {
        std::cerr << "No valid solution found!" << std::endl;
        return false;
    }
    auto group_gemm_status = LaunchGroupGEMM<MATRIX_M, MATRIX_N, MATRIX_K, NUM_EXPERTS>(
        d_A, d_B, d_C, h_IDX, heuristic_results[0].algo, stream);
    CHECK_HIPBLASLT_ERROR(group_gemm_status);
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    return true;
}

bool BenchmarkGroupGEMM() {
    InitWorkspace();
    std::array<size_t, NUM_EXPERTS + 1> h_IDX = {0,     1026,  2017,  3044,  4031,  5123,  6151,  7175,  8151,
                                                 9153,  10173, 11225, 12255, 13236, 14221, 15274, 16343, 17384,
                                                 18428, 19438, 20458, 21457, 22520, 23580, 24578, 25603, 26601,
                                                 27673, 28661, 29730, 30796, 31774, 32768};

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    std::vector<half> h_A(MATRIX_K * MATRIX_M);
    std::vector<half> h_B(NUM_EXPERTS * MATRIX_K * MATRIX_N);
    std::vector<float> h_C(MATRIX_M * MATRIX_N);
    for (auto &v : h_A)
        v = __float2half(dist(gen));
    for (auto &v : h_B)
        v = __float2half(dist(gen));

    half *d_A, *d_B;
    float *d_C;
    CHECK_HIP_ERROR(hipMalloc(&d_A, h_A.size() * sizeof(half)));
    CHECK_HIP_ERROR(hipMalloc(&d_B, h_B.size() * sizeof(half)));
    CHECK_HIP_ERROR(hipMalloc(&d_C, h_C.size() * sizeof(float)));
    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A.data(), h_A.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, h_B.data(), h_B.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(d_C, 0, h_C.size() * sizeof(float)));

    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    hipEvent_t start_event, stop_event;
    CHECK_HIP_ERROR(hipEventCreate(&start_event));
    CHECK_HIP_ERROR(hipEventCreate(&stop_event));

    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_results;
    CHECK_HIPBLASLT_ERROR(hipblaslt_ext::getAllAlgos(hipblasLtHandle, hipblaslt_ext::GemmType::HIPBLASLT_GROUPED_GEMM,
                                                     HIPBLAS_OP_T, HIPBLAS_OP_N, HIP_R_16F, HIP_R_16F, HIP_R_32F,
                                                     HIP_R_32F, HIPBLAS_COMPUTE_32F, heuristic_results));

    if (heuristic_results.empty()) {
        std::cerr << "No valid solution found!" << std::endl;
        return false;
    }
    constexpr int num_iterations = 3;
    struct AlgoPerf {
        int algo_id;
        float avg_time_ms;
        float min_time_ms;
        float max_time_ms;
        float std_dev_ms;
        float tflops;
    };
    std::vector<AlgoPerf> perf_results;
    for (size_t algo_idx = 0; algo_idx < heuristic_results.size(); ++algo_idx) {
        auto algo = heuristic_results[algo_idx].algo;
        std::vector<float> iteration_times(num_iterations);
        for (int i = 0; i < 2; i++) {
            CHECK_HIPBLASLT_ERROR(
                LaunchGroupGEMM<MATRIX_M, MATRIX_N, MATRIX_K, NUM_EXPERTS>(d_A, d_B, d_C, h_IDX, algo, stream));
            CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        }
        for (int i = 0; i < num_iterations; i++) {
            CHECK_HIP_ERROR(hipEventRecord(start_event, stream));
            CHECK_HIPBLASLT_ERROR(
                LaunchGroupGEMM<MATRIX_M, MATRIX_N, MATRIX_K, NUM_EXPERTS>(d_A, d_B, d_C, h_IDX, algo, stream));
            CHECK_HIP_ERROR(hipEventRecord(stop_event, stream));
            CHECK_HIP_ERROR(hipEventSynchronize(stop_event));
            CHECK_HIP_ERROR(hipEventElapsedTime(&iteration_times[i], start_event, stop_event));
        }
        double total_time_ms = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0);
        double avg_time_ms = total_time_ms / num_iterations;
        auto min_max = std::minmax_element(iteration_times.begin(), iteration_times.end());
        float min_time_ms = *min_max.first;
        float max_time_ms = *min_max.second;
        double variance = 0.0;
        for (float time : iteration_times)
            variance += (time - avg_time_ms) * (time - avg_time_ms);
        variance /= num_iterations;
        double std_dev_ms = std::sqrt(variance);
        double total_flops = 2.0 * MATRIX_M * MATRIX_N * MATRIX_K;
        double tflops = (total_flops / 1e12) / (avg_time_ms / 1000.0);
        perf_results.push_back({(int)hipblaslt_ext::getIndexFromAlgo(algo), (float)avg_time_ms, min_time_ms,
                                max_time_ms, (float)std_dev_ms, (float)tflops});
    }
    std::cout << "\nAlgo Performance Comparison:\n";
    std::cout << "Idx\tAvg(ms)\tMin(ms)\tMax(ms)\tStd(ms)\tTFLOPS\n";
    std::sort(perf_results.begin(), perf_results.end(),
              [](const AlgoPerf &a, const AlgoPerf &b) { return a.tflops > b.tflops; });
    for (size_t i = 0; i < std::min(perf_results.size(), size_t(20)); ++i) {
        const auto &perf = perf_results[i];
        std::cout << perf.algo_id << "\t" << perf.avg_time_ms << "\t" << perf.min_time_ms << "\t" << perf.max_time_ms
                  << "\t" << perf.std_dev_ms << "\t" << perf.tflops << std::endl;
    }
    CHECK_HIP_ERROR(hipEventDestroy(start_event));
    CHECK_HIP_ERROR(hipEventDestroy(stop_event));
    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_B));
    CHECK_HIP_ERROR(hipFree(d_C));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
    return true;
}

bool BenchmarkGEMM() {
    InitWorkspace();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    std::vector<half> h_A(MATRIX_K * (1024 + 128) * NUM_EXPERTS);
    std::vector<half> h_B(NUM_EXPERTS * MATRIX_K * MATRIX_N);
    std::vector<float> h_C((1024 + 128) * NUM_EXPERTS * MATRIX_N);
    for (auto &v : h_A)
        v = __float2half(dist(gen));
    for (auto &v : h_B)
        v = __float2half(dist(gen));

    half *d_A, *d_B;
    float *d_C;
    CHECK_HIP_ERROR(hipMalloc(&d_A, h_A.size() * sizeof(half)));
    CHECK_HIP_ERROR(hipMalloc(&d_B, h_B.size() * sizeof(half)));
    CHECK_HIP_ERROR(hipMalloc(&d_C, h_C.size() * sizeof(float)));
    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A.data(), h_A.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, h_B.data(), h_B.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(d_C, 0, h_C.size() * sizeof(float)));

    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    hipEvent_t start_event, stop_event;
    CHECK_HIP_ERROR(hipEventCreate(&start_event));
    CHECK_HIP_ERROR(hipEventCreate(&stop_event));

    std::vector<hipblasLtMatmulHeuristicResult_t> heuristic_results;
    CHECK_HIPBLASLT_ERROR(hipblaslt_ext::getAllAlgos(hipblasLtHandle, hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                                     HIPBLAS_OP_N, HIPBLAS_OP_N, HIP_R_16F, HIP_R_16F, HIP_R_32F,
                                                     HIP_R_32F, HIPBLAS_COMPUTE_32F, heuristic_results));

    if (heuristic_results.empty()) {
        std::cerr << "No valid solution found!" << std::endl;
        return false;
    }
    constexpr int num_iterations = 3;
    struct AlgoPerf {
        int algo_id;
        float avg_time_ms;
        float min_time_ms;
        float max_time_ms;
        float std_dev_ms;
        float tflops;
    };
    std::vector<AlgoPerf> perf_results;
    for (size_t algo_idx = 0; algo_idx < heuristic_results.size(); ++algo_idx) {
        auto algo = heuristic_results[algo_idx].algo;
        std::vector<float> iteration_times(num_iterations);
        bool algo_failed = false;
        
        for (int i = 0; i < 2; i++) {
            auto status = LaunchBatchedGEMM<1024 + 128, MATRIX_N, MATRIX_K>(d_A, d_B, d_C, algo, stream);
            if (status != HIPBLAS_STATUS_SUCCESS) {
                algo_failed = true;
                break;
            }
            CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        }
        
        if (algo_failed) continue;
        
        for (int i = 0; i < num_iterations; i++) {
            CHECK_HIP_ERROR(hipEventRecord(start_event, stream));
            auto status = LaunchBatchedGEMM<1024 + 128, MATRIX_N, MATRIX_K>(d_A, d_B, d_C, algo, stream);
            if (status != HIPBLAS_STATUS_SUCCESS) {
                algo_failed = true;
                break;
            }
            CHECK_HIP_ERROR(hipEventRecord(stop_event, stream));
            CHECK_HIP_ERROR(hipEventSynchronize(stop_event));
            CHECK_HIP_ERROR(hipEventElapsedTime(&iteration_times[i], start_event, stop_event));
        }
        
        if (algo_failed) continue;
        
        double total_time_ms = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0);
        double avg_time_ms = total_time_ms / num_iterations;
        auto min_max = std::minmax_element(iteration_times.begin(), iteration_times.end());
        float min_time_ms = *min_max.first;
        float max_time_ms = *min_max.second;
        double variance = 0.0;
        for (float time : iteration_times)
            variance += (time - avg_time_ms) * (time - avg_time_ms);
        variance /= num_iterations;
        double std_dev_ms = std::sqrt(variance);
        double total_flops = 2.0 * MATRIX_M * MATRIX_N * MATRIX_K;
        double tflops = (total_flops / 1e12) / (avg_time_ms / 1000.0);
        perf_results.push_back({(int)hipblaslt_ext::getIndexFromAlgo(algo), (float)avg_time_ms, min_time_ms,
                                max_time_ms, (float)std_dev_ms, (float)tflops});
    }
    std::cout << "\nAlgo Performance Comparison:\n";
    std::cout << "Idx\tAvg(ms)\tMin(ms)\tMax(ms)\tStd(ms)\tTFLOPS\n";
    std::sort(perf_results.begin(), perf_results.end(),
              [](const AlgoPerf &a, const AlgoPerf &b) { return a.tflops > b.tflops; });
    for (size_t i = 0; i < std::min(perf_results.size(), size_t(20)); ++i) {
        const auto &perf = perf_results[i];
        std::cout << perf.algo_id << "\t" << perf.avg_time_ms << "\t" << perf.min_time_ms << "\t" << perf.max_time_ms
                  << "\t" << perf.std_dev_ms << "\t" << perf.tflops << std::endl;
    }
    CHECK_HIP_ERROR(hipEventDestroy(start_event));
    CHECK_HIP_ERROR(hipEventDestroy(stop_event));
    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_B));
    CHECK_HIP_ERROR(hipFree(d_C));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));
    return true;
}

#ifdef ENABLE_TORCH_TESTS
bool TestLibTorchGroupGEMM() {
    std::array<size_t, NUM_EXPERTS + 1> h_IDX = {0,     1026,  2017,  3044,  4031,  5123,  6151,  7175,  8151,
                                                 9153,  10173, 11225, 12255, 13236, 14221, 15274, 16343, 17384,
                                                 18428, 19438, 20458, 21457, 22520, 23580, 24578, 25603, 26601,
                                                 27673, 28661, 29730, 30796, 31774, 32768};

    std::vector<half> h_A(MATRIX_K * MATRIX_M);
    std::vector<half> h_B(NUM_EXPERTS * MATRIX_K * MATRIX_N);
    std::vector<float> h_C(MATRIX_M * MATRIX_N);

    half *d_A, *d_B;
    float *d_C;
    CHECK_HIP_ERROR(hipMalloc(&d_A, h_A.size() * sizeof(half)));
    CHECK_HIP_ERROR(hipMalloc(&d_B, h_B.size() * sizeof(half)));
    CHECK_HIP_ERROR(hipMalloc(&d_C, h_C.size() * sizeof(float)));
    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A.data(), h_A.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, h_B.data(), h_B.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(d_C, 0, h_C.size() * sizeof(float)));

    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    std::cout << "Running LibTorch grouped GEMM..." << std::endl;

    LaunchTorchRef<MATRIX_M, MATRIX_N, MATRIX_K, NUM_EXPERTS>(d_A, d_B, d_C, h_IDX, stream);
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));

    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_B));
    CHECK_HIP_ERROR(hipFree(d_C));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));

    std::cout << "LibTorch grouped GEMM completed successfully." << std::endl;
    return true;
}

bool BenchmarkLibTorchGroupGEMM() {
    std::array<size_t, NUM_EXPERTS + 1> h_IDX = {0,     1026,  2017,  3044,  4031,  5123,  6151,  7175,  8151,
                                                 9153,  10173, 11225, 12255, 13236, 14221, 15274, 16343, 17384,
                                                 18428, 19438, 20458, 21457, 22520, 23580, 24578, 25603, 26601,
                                                 27673, 28661, 29730, 30796, 31774, 32768};

    std::vector<half> h_A(MATRIX_K * MATRIX_M);
    std::vector<half> h_B(NUM_EXPERTS * MATRIX_K * MATRIX_N);
    std::vector<half> h_C(MATRIX_M * MATRIX_N);

    half *d_A, *d_B;
    float *d_C;
    CHECK_HIP_ERROR(hipMalloc(&d_A, h_A.size() * sizeof(half)));
    CHECK_HIP_ERROR(hipMalloc(&d_B, h_B.size() * sizeof(half)));
    CHECK_HIP_ERROR(hipMalloc(&d_C, h_C.size() * sizeof(float)));
    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A.data(), h_A.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_B, h_B.data(), h_B.size() * sizeof(half), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemset(d_C, 0, h_C.size() * sizeof(float)));

    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreate(&stream));

    // Warmup runs (2 times)
    std::cout << "Performing LibTorch warmup runs..." << std::endl;
    for (int warmup = 0; warmup < 2; warmup++) {
        LaunchTorchRef<MATRIX_M, MATRIX_N, MATRIX_K, NUM_EXPERTS>(d_A, d_B, d_C, h_IDX, stream);
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    }

    // Performance measurement
    constexpr int num_iterations = 10;
    double total_time_ms = 0.0;

    std::cout << "Running LibTorch performance test..." << std::endl;
    for (int iter = 0; iter < num_iterations; iter++) {
        auto start = std::chrono::high_resolution_clock::now();

        LaunchTorchRef<MATRIX_M, MATRIX_N, MATRIX_K, NUM_EXPERTS>(d_A, d_B, d_C, h_IDX, stream);
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        total_time_ms += elapsed.count();

        std::cout << "  Iteration " << iter + 1 << "/" << num_iterations << ": " << elapsed.count() << " ms"
                  << std::endl;
    }

    // Calculate average time and TFLOPS
    double avg_time_ms = total_time_ms / num_iterations;

    // Calculate total FLOPs
    double total_flops = 2.0 * MATRIX_M * MATRIX_N * MATRIX_K;
    double tflops = (total_flops / 1e12) / (avg_time_ms / 1000.0);

    std::cout << "\nLibTorch Performance Results:" << std::endl;
    std::cout << "  Matrix dimensions: M=" << MATRIX_M << ", N=" << MATRIX_N << ", K=" << MATRIX_K
              << ", experts=" << NUM_EXPERTS << std::endl;
    std::cout << "  Total FLOPs: " << total_flops << std::endl;
    std::cout << "  Average execution time: " << avg_time_ms << " ms" << std::endl;
    std::cout << "  Performance: " << tflops << " TFLOPS" << std::endl;

    // Cleanup
    CHECK_HIP_ERROR(hipFree(d_A));
    CHECK_HIP_ERROR(hipFree(d_B));
    CHECK_HIP_ERROR(hipFree(d_C));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));

    return true;
}

bool ValidateResults() {

    std::array<size_t, NUM_EXPERTS + 1> h_IDX = {0,     1026,  2017,  3044,  4031,  5123,  6151,  7175,  8151,
                                                 9153,  10173, 11225, 12255, 13236, 14221, 15274, 16343, 17384,
                                                 18428, 19438, 20458, 21457, 22520, 23580, 24578, 25603, 26601,
                                                 27673, 28661, 29730, 30796, 31774, 32768};
    // std::array<size_t, NUM_EXPERTS + 1> h_IDX = {0, MATRIX_M};
    // Create same input data for both implementations
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducible results
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    std::vector<half> h_A(MATRIX_K * MATRIX_M);
    std::vector<half> h_B(NUM_EXPERTS * MATRIX_K * MATRIX_N);
    std::generate(h_A.begin(), h_A.end(), [&]() { return __float2half(dist(gen)); });
    std::generate(h_B.begin(), h_B.end(), [&]() { return __float2half(dist(gen)); });

    // HipBLAS computation
    std::vector<float> h_C_hipblas(MATRIX_M * MATRIX_N);
    {
        InitWorkspace();
        half *d_A, *d_B;
        float *d_C;
        CHECK_HIP_ERROR(hipMalloc(&d_A, h_A.size() * sizeof(half)));
        CHECK_HIP_ERROR(hipMalloc(&d_B, h_B.size() * sizeof(half)));
        CHECK_HIP_ERROR(hipMalloc(&d_C, h_C_hipblas.size() * sizeof(float)));
        CHECK_HIP_ERROR(hipMemcpy(d_A, h_A.data(), h_A.size() * sizeof(half), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_B, h_B.data(), h_B.size() * sizeof(half), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemset(d_C, 0, h_C_hipblas.size() * sizeof(float)));

        hipStream_t stream;
        CHECK_HIP_ERROR(hipStreamCreate(&stream));

        std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResults;
        std::vector<int> algo_index = {189706};
        hipblaslt_ext::getAlgosFromIndex(hipblasLtHandle, algo_index, heuristicResults);
        auto algo = heuristicResults[0].algo;

        auto status = LaunchGroupGEMM<MATRIX_M, MATRIX_N, MATRIX_K, NUM_EXPERTS>(d_A, d_B, d_C, h_IDX, algo);
        CHECK_HIPBLASLT_ERROR(status);
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));

        CHECK_HIP_ERROR(hipMemcpy(h_C_hipblas.data(), d_C, h_C_hipblas.size() * sizeof(float), hipMemcpyDeviceToHost));

        CHECK_HIP_ERROR(hipFree(d_A));
        CHECK_HIP_ERROR(hipFree(d_B));
        CHECK_HIP_ERROR(hipFree(d_C));
        CHECK_HIP_ERROR(hipStreamDestroy(stream));
    }

    // LibTorch computation
    std::vector<float> h_C_torch(MATRIX_M * MATRIX_N);
    {
        half *d_A, *d_B;
        float *d_C;
        CHECK_HIP_ERROR(hipMalloc(&d_A, h_A.size() * sizeof(half)));
        CHECK_HIP_ERROR(hipMalloc(&d_B, h_B.size() * sizeof(half)));
        CHECK_HIP_ERROR(hipMalloc(&d_C, h_C_torch.size() * sizeof(float)));
        CHECK_HIP_ERROR(hipMemcpy(d_A, h_A.data(), h_A.size() * sizeof(half), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_B, h_B.data(), h_B.size() * sizeof(half), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemset(d_C, 0, h_C_torch.size() * sizeof(float)));

        hipStream_t stream;
        CHECK_HIP_ERROR(hipStreamCreate(&stream));
        LaunchTorchRef<MATRIX_M, MATRIX_N, MATRIX_K, NUM_EXPERTS>(d_A, d_B, d_C, h_IDX, stream);
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));

        CHECK_HIP_ERROR(hipMemcpy(h_C_torch.data(), d_C, h_C_torch.size() * sizeof(float), hipMemcpyDeviceToHost));

        CHECK_HIP_ERROR(hipFree(d_A));
        CHECK_HIP_ERROR(hipFree(d_B));
        CHECK_HIP_ERROR(hipFree(d_C));
        CHECK_HIP_ERROR(hipStreamDestroy(stream));
    }

    // Use PyTorch's allclose for comparison
    constexpr double atol = 0.1;  // Absolute tolerance
    constexpr double rtol = 0.01; // Relative tolerance

    // Convert results to PyTorch tensors
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor C_hipblas = torch::from_blob(h_C_hipblas.data(), {MATRIX_M, MATRIX_N}, options).clone();
    torch::Tensor C_torch = torch::from_blob(h_C_torch.data(), {MATRIX_M, MATRIX_N}, options).clone();

    // Check if tensors are close
    bool validation_passed = torch::allclose(C_hipblas, C_torch, rtol, atol);

    // Calculate statistics for reporting
    torch::Tensor diff = torch::abs(C_hipblas - C_torch);
    double max_diff = torch::max(diff).item<double>();
    double avg_diff = torch::mean(diff).item<double>();

    // Count elements outside tolerance
    torch::Tensor abs_diff = torch::abs(C_hipblas - C_torch);
    torch::Tensor rel_diff = abs_diff / (torch::abs(C_torch) + 1e-8);
    torch::Tensor exceeds_tolerance = (abs_diff > atol) & (rel_diff > rtol);
    int num_mismatches = torch::sum(exceeds_tolerance).item<int>();

    std::cout << "\nNumerical Validation Results:" << std::endl;
    std::cout << "  Total elements: " << h_C_hipblas.size() << std::endl;
    std::cout << "  Absolute tolerance: " << atol << std::endl;
    std::cout << "  Relative tolerance: " << rtol << std::endl;
    std::cout << "  Maximum difference: " << max_diff << std::endl;
    std::cout << "  Average absolute difference: " << avg_diff << std::endl;
    std::cout << "  Elements outside tolerance: " << num_mismatches << std::endl;
    std::cout << "  Percentage mismatched: " << (100.0 * num_mismatches / h_C_hipblas.size()) << "%" << std::endl;
    std::cout << "  Validation (torch.allclose): " << (validation_passed ? "PASSED" : "FAILED") << std::endl;

    return validation_passed;
}

#endif

int main(int argc, char *argv[]) {
    std::cout << "Testing GroupGEMM function..." << std::endl;
    bool test_passed = true;

    if (test_passed) {
#ifdef ENABLE_TORCH_TESTS
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "Testing LibTorch GroupGEMM function..." << std::endl;
        bool torch_test_passed = TestLibTorchGroupGEMM();

        if (torch_test_passed) {
            std::cout << "\n" << std::string(50, '=') << std::endl;
            std::cout << "Validating numerical consistency..." << std::endl;
            bool validation_passed = ValidateResults();

            if (validation_passed) {
                std::cout << "\n" << std::string(50, '=') << std::endl;
                std::cout << "Starting GroupGEMM performance benchmark..." << std::endl;
                BenchmarkGroupGEMM();

                std::cout << "\n" << std::string(50, '=') << std::endl;
                std::cout << "Starting LibTorch GroupGEMM performance benchmark..." << std::endl;
                BenchmarkLibTorchGroupGEMM();
            } else {
                std::cerr << "Numerical validation failed, skipping performance benchmarks." << std::endl;
                return 1;
            }
        }
#else
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "LibTorch tests disabled. Starting GroupGEMM performance benchmark..." << std::endl;
        BenchmarkGEMM();
#endif
    } else {
        std::cerr << "Functional test failed, skipping all other tests." << std::endl;
        return 1;
    }

    return 0;
}
