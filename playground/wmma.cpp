// #pragma clang diagnostic push
// #pragma clang diagnostic ignored "-Wunknown-attributes"
// #include "../include/gpu_libs.h"
// #include "../include/gpu_types.h"
// #include <random>
// #include <ctime>
// #include <cmath> // For std::abs


// // Helper macro for HIP error checking
// #define HIP_CHECK(command)                                                                                             \
//   {                                                                                                                    \
//     hipError_t status = command;                                                                                       \
//     if (status != hipSuccess) {                                                                                        \
//       std::cerr << "HIP Error: " << hipGetErrorString(status) << " at line " << __LINE__ << std::endl;                 \
//       exit(EXIT_FAILURE);                                                                                              \
//     }                                                                                                                  \
//   }

// constexpr int WMMA_M = 16;
// constexpr int WMMA_N = 16;
// constexpr int WMMA_K = 32;
// constexpr int BLOCK_SIZE = 64;

// template <typename data_type, int BK, int BM, int BN, int BLOCK_SIZE>
// __global__ void compute_tile_wmma(float r_c[BM][BN], const data_type s_a[BK][BM], const data_type s_b[BK][BN]) {
//   static_assert(BM % WMMA_M == 0);
//   static_assert(BN % WMMA_N == 0);
//   static_assert(BK % WMMA_K == 0);
//   static_assert(BLOCK_SIZE % WAVE_SIZE == 0);
//   constexpr int warp_num = BLOCK_SIZE / WAVE_SIZE;
//   int tid = threadIdx.x;

//   int warp_id = tid / WAVE_SIZE;
//   int lane_id = tid % WAVE_SIZE;
//   int tile_xs = BN / WMMA_N;
//   int tile_ys = BM / WMMA_M;
//   int tile_num = tile_xs * tile_ys;
//   for (int tile_idx = warp_id; tile_idx < tile_num; tile_idx += warp_num) {
//     int tile_idx_y = tile_idx / tile_xs;
//     int tile_idx_x = tile_idx % tile_xs;
//     int tile_idx_a_xi = tile_idx_y * WMMA_M;
//     int tile_idx_b_xi = tile_idx_x * WMMA_N;
//     rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_acc_total;
//     rocwmma::fill_fragment(frag_acc_total, 0.0f);
//     for (int tile_k = 0; tile_k < BK / WMMA_K; tile_k++) {
//       int tile_idx_a = tile_idx_a_xi + tile_k * BM * WMMA_K;
//       int tile_idx_b = tile_idx_b_xi + tile_k * BN * WMMA_K;
//       rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, data_type, rocwmma::col_major> frag_a;
//       rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, data_type, rocwmma::row_major> frag_b;

//       rocwmma::load_matrix_sync(frag_a, reinterpret_cast<const data_type *>(s_a) + tile_idx_a, BM);
//       rocwmma::load_matrix_sync(frag_b, reinterpret_cast<const data_type *>(s_b) + tile_idx_b, BN);

//       rocwmma::mma_sync(frag_acc_total, frag_a, frag_b, frag_acc_total);
//     }
//     rocwmma::store_matrix_sync(reinterpret_cast<float *>(r_c) + tile_idx_x * WMMA_N + tile_idx_y * WMMA_M * BN,
//                                frag_acc_total, BN, rocwmma::mem_row_major);
//   }
// }

// int main() {
//   // Define matrix dimensions
//   const int M = 160;
//   const int N = 160;
//   const int K = 320;
//   // Host matrices
//   float h_a[K][M];
//   float h_b[K][N];
//   float h_c[M][N] = {0};

//   // Initialize random number generator
//   std::mt19937 gen(std::time(0));
//   std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

//   // Initialize matrices with random values
//   for (int i = 0; i < K; i++) {
//     for (int j = 0; j < M; j++) {
//       h_a[i][j] = dist(gen); // Assign random float
//     }
//   }
//   for (int i = 0; i < K; i++) {
//     for (int j = 0; j < N; j++) {
//       h_b[i][j] = dist(gen); // Assign random float
//     }
//   }

//   // Convert host float matrices to FP8 for both CPU reference and GPU
//   __FP8_TYPE *h_a_fp8 = new __FP8_TYPE[K * M];
//   __FP8_TYPE *h_b_fp8 = new __FP8_TYPE[K * N];

//   for (int i = 0; i < K; i++) {
//     for (int j = 0; j < M; j++) {
//       h_a_fp8[i * M + j] = __FP8_TYPE(h_a[i][j]); // Assuming KxM row-major storage for h_a_fp8
//     }
//   }
//   for (int i = 0; i < K; i++) {
//     for (int j = 0; j < N; j++) {
//       h_b_fp8[i * N + j] = __FP8_TYPE(h_b[i][j]); // Assuming KxN row-major storage for h_b_fp8
//     }
//   }

//   // Calculate reference result on CPU using FP8 values converted back to float
//   float *h_cpu_result = new float[M * N];
//   for (int i = 0; i < M; ++i) {
//     for (int j = 0; j < N; ++j) {
//       float sum = 0.0f;
//       for (int k_idx = 0; k_idx < K; ++k_idx) {
//         // Need to access h_a_fp8 and h_b_fp8 correctly based on C = A^T * B ? No, C=A*B where A is KxM, B is KxN
//         // The kernel seems to treat A as col_major (MxK effective) and B as row_major (KxN effective)
//         // Original CPU loop implies C[i][j] = sum(A[k][i] * B[k][j]) -> A is KxM, B is KxN
//         // Let's match this interpretation. h_a[k][i] -> h_a_fp8[k * M + i]. h_b[k][j] -> h_b_fp8[k * N + j]
//         sum += float(h_a_fp8[k_idx * M + i]) * float(h_b_fp8[k_idx * N + j]);
//       }
//       h_cpu_result[i * N + j] = sum;
//     }
//   }

//   // Device pointers
//   __FP8_TYPE *d_a, *d_b;
//   float *d_c;
//   // Allocate device memory
//   hipError_t err = hipMalloc(&d_a, K * M * sizeof(__FP8_TYPE));
//   if (err != hipSuccess)
//     return 1;
//   err = hipMalloc(&d_b, K * N * sizeof(__FP8_TYPE));
//   if (err != hipSuccess)
//     return 1;
//   err = hipMalloc(&d_c, M * N * sizeof(float));
//   if (err != hipSuccess)
//     return 1;

//   // Copy FP8 data to device (already converted)
//   err = hipMemcpy(d_a, h_a_fp8, K * M * sizeof(__FP8_TYPE), hipMemcpyHostToDevice);
//   if (err != hipSuccess) {
//     delete[] h_a_fp8;
//     delete[] h_b_fp8;
//     delete[] h_cpu_result;
//     return 1;
//   }
//   err = hipMemcpy(d_b, h_b_fp8, K * N * sizeof(__FP8_TYPE), hipMemcpyHostToDevice);
//   if (err != hipSuccess) {
//     delete[] h_a_fp8;
//     delete[] h_b_fp8;
//     delete[] h_cpu_result;
//     return 1;
//   }
//   // Copy initial C data to device (host h_c is all zeros)
//   err = hipMemcpy(d_c, h_c, M * N * sizeof(float), hipMemcpyHostToDevice);
//   if (err != hipSuccess) {
//     delete[] h_a_fp8;
//     delete[] h_b_fp8;
//     delete[] h_cpu_result;
//     return 1;
//   }

//   // Free host FP8 arrays now they are on device and used for CPU calc
//   delete[] h_a_fp8;
//   delete[] h_b_fp8;

//   // Here you would run your computation kernel
//   compute_tile_wmma<__FP8_TYPE, K, M, N, BLOCK_SIZE><<<1, BLOCK_SIZE>>>(
//       reinterpret_cast<float(*)[N]>(d_c), reinterpret_cast<__FP8_TYPE(*)[M]>(d_a), reinterpret_cast<__FP8_TYPE(*)[N]>(d_b));

//   HIP_CHECK(hipGetLastError());      // Check for kernel launch errors
//   HIP_CHECK(hipDeviceSynchronize()); // Wait for kernel to complete

//   // Allocate host memory to store the result from device
//   float *h_result = new float[M * N];
//   // Copy result from device to host
//   err = hipMemcpy(h_result, d_c, M * N * sizeof(float), hipMemcpyDeviceToHost);
//   if (err != hipSuccess) {
//     delete[] h_result;
//     delete[] h_cpu_result;
//     return 1;
//   }

//   // Compare GPU result with CPU result
//   bool error = false;
//   float epsilon = 1e-5f; // Tolerance for floating point comparison
//   for (int i = 0; i < M; i++) {
//     for (int j = 0; j < N; j++) {
//       if (std::abs(h_result[i * N + j] - h_cpu_result[i * N + j]) > epsilon) {
//         std::cerr << "Error: Mismatch at index (" << i << ", " << j << "). "
//                   << "GPU result: " << h_result[i * N + j] << ", CPU result: " << h_cpu_result[i * N + j] << std::endl;
//         error = true;
//         // break; // Optional: Stop checking after first error
//       }
//     }
//     // if (error) break; // Optional: Stop checking after first error in a row
//   }

//   if (!error) {
//     std::cerr << "Validation Passed: GPU results match CPU results." << std::endl;
//   } else {
//     std::cerr << "Validation Failed: GPU results do not match CPU results." << std::endl;
//   }

//   // Free host memory for results
//   delete[] h_result;
//   delete[] h_cpu_result;

//   // Free device memory
//   err = hipFree(d_a);
//   if (err != hipSuccess)
//     return 1;
//   err = hipFree(d_b);
//   if (err != hipSuccess)
//     return 1;
//   err = hipFree(d_c);
//   if (err != hipSuccess)
//     return 1;
//   std::cerr << "run completed" << std::endl;
//   return 0;
// }

int main() {
  
}