#include "../include/gpu_libs.h"
#include "../include/gpu_types.h"
#include "../include/clangd_workaround.h"
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <iostream>
#include <chrono>
#include <vector>

#define CHECK_HIP_ERROR(error) \
    if(error != hipSuccess) { \
        std::cerr << "Hip error: " << hipGetErrorString(error) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(-1); \
    }

#define CHECK_HIPBLASLT_ERROR(error) \
    if(error != HIPBLAS_STATUS_SUCCESS) { \
        std::cerr << "hipBLASLt error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(-1); \
    }

class HipBLASLtBenchmark {
private:
    hipblasLtHandle_t handle;
    void* d_A;
    void* d_B; 
    void* d_C;
    void* d_workspace;
    int64_t max_workspace_size;
    hipStream_t stream;
    
public:
    HipBLASLtBenchmark() : max_workspace_size(32 * 1024 * 1024) {
        CHECK_HIPBLASLT_ERROR(hipblasLtCreate(&handle));
        CHECK_HIP_ERROR(hipStreamCreate(&stream));
        CHECK_HIP_ERROR(hipMalloc(&d_workspace, max_workspace_size));
    }
    
    ~HipBLASLtBenchmark() {
        if(d_A) CHECK_HIP_ERROR(hipFree(d_A));
        if(d_B) CHECK_HIP_ERROR(hipFree(d_B));
        if(d_C) CHECK_HIP_ERROR(hipFree(d_C));
        if(d_workspace) CHECK_HIP_ERROR(hipFree(d_workspace));
        CHECK_HIP_ERROR(hipStreamDestroy(stream));
        hipblasLtDestroy(handle);
    }
    
    void allocateMemory(int64_t M, int64_t N, int64_t K) {
        size_t sizeA = M * K * sizeof(half);
        size_t sizeB = K * N * sizeof(half);
        size_t sizeC = M * N * sizeof(half);
        
        CHECK_HIP_ERROR(hipMalloc(&d_A, sizeA));
        CHECK_HIP_ERROR(hipMalloc(&d_B, sizeB));
        CHECK_HIP_ERROR(hipMalloc(&d_C, sizeC));
        
        // Initialize with dummy data
        std::vector<half> hostA(M * K), hostB(K * N);
        for(int i = 0; i < M * K; i++) hostA[i] = __float2half(0.5f);
        for(int i = 0; i < K * N; i++) hostB[i] = __float2half(0.5f);
        
        CHECK_HIP_ERROR(hipMemcpy(d_A, hostA.data(), sizeA, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_B, hostB.data(), sizeB, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemset(d_C, 0, sizeC));
    }
    
    void runGemm(int64_t M, int64_t N, int64_t K, float& alpha, float& beta) {
        // Use hipblaslt extension API
        hipblaslt_ext::GemmPreferenceV2 gemmPref;
        gemmPref.setMaxWorkspaceBytes(max_workspace_size);
        
        // Create Gemm object with transposed A and normal B
        hipblaslt_ext::Gemm gemm(handle, 
                                HIPBLAS_OP_T,    // A is transposed
                                HIPBLAS_OP_N,    // B is normal
                                HIP_R_16F,       // A type
                                HIP_R_16F,       // B type  
                                HIP_R_16F,       // C type
                                HIP_R_16F,       // D type
                                HIPBLAS_COMPUTE_32F); // Compute type (float accumulation)
        
        hipblaslt_ext::GemmEpilogueV2 epilogue; // Default epilogue
        epilogue.setMode(HIPBLASLT_EPILOGUE_DEFAULT); // Default epilogue
        hipblaslt_ext::GemmInputsV2 inputs;
        inputs.setA(d_A);
        inputs.setB(d_B);
        inputs.setC(d_C);
        inputs.setD(d_C);
        inputs.setAlpha(&alpha);
        inputs.setBeta(&beta);
        
        gemm.setProblem(M, N, K, 1, epilogue, inputs);
        
        // Get heuristic solutions
        const int request_solutions = 1;
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
        CHECK_HIPBLASLT_ERROR(gemm.algoGetHeuristic(request_solutions, gemmPref, heuristicResult));
        
        if(heuristicResult.empty()) {
            std::cerr << "No valid solution found!" << std::endl;
            return;
        }
        
        // Initialize and run
        CHECK_HIPBLASLT_ERROR(gemm.initialize(heuristicResult[0].algo, d_workspace));
        CHECK_HIPBLASLT_ERROR(gemm.run(stream));
    }
    
    double benchmarkMatmul(int64_t M, int64_t N, int64_t K, int iterations = 100) {
        allocateMemory(M, N, K);
        
        float alpha = 1.0f, beta = 0.0f;
        
        // Warm up
        for(int i = 0; i < 10; i++) {
            runGemm(M, N, K, alpha, beta);
        }
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < iterations; i++) {
            runGemm(M, N, K, alpha, beta);
        }
        CHECK_HIP_ERROR(hipStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        return elapsed / iterations;
    }
};

int main() {
    HipBLASLtBenchmark benchmark;
    
    // Test irregular M values with different N,K combinations
    std::vector<int64_t> M_values = {1023, 1025, 2047, 2049, 4095, 4097, 8191};
    std::vector<std::pair<int64_t, int64_t>> NK_pairs = {{7168, 7168}, {7168, 2048}, {2048, 7168}, {2048, 2048}};
    
    std::cout << "Testing half precision with transposed A, float accumulation using hipBLASLt Extension API\n";
    std::cout << "M\tN\tK\tTime(ms)\tTFLOPS\n";
    std::cout << "================================================\n";
    
    for(auto M : M_values) {
        for(auto [N, K] : NK_pairs) {
            try {
                double avgTime = benchmark.benchmarkMatmul(M, N, K);
                double flops = 2.0 * M * N * K;
                double tflops = (flops / (avgTime * 1e-3)) / 1e12;
                
                std::cout << M << "\t" << N << "\t" << K << "\t" 
                          << avgTime << "\t\t" << tflops << std::endl;
            } catch (const std::exception& e) {
                std::cout << M << "\t" << N << "\t" << K << "\t" 
                          << "ERROR" << std::endl;
            }
        }
    }
    
    return 0;
}
