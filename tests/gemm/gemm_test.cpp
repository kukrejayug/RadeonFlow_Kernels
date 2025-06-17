#include "checker.h"
#include <ATen/ops/ones.h>
#include <dlfcn.h>
#include <c10/core/ScalarType.h>
#include "../../tests/checker/metrics.h"
#include <hip/hip_runtime.h>

using OutputData = torch::Tensor;
struct Shape {
  int m;
  int n;
  int k;
  int seed;
};

// ========= Shape Config =========

Shape shapes[] = {
    // {1024, 1536, 7168, 8135},
    // Tests
    // {64, 64, 128, 6635},
    // {64, 1536, 7168, 6635},
    // {64, 3072, 1536, 1236},
    // {64, 576, 7168, 542},
    // {96, 7168, 256, 1234},
    // {96, 7168, 2048, 4153},
    // {96, 4608, 7168, 412},
    // {128, 7168, 2304, 624},
    // {128, 512, 7168, 2514},
    // {512, 4096, 512, 543},
    // {512, 1536, 7168, 12341},
    // {4096, 4096, 4096, 0},
    // Benchmarks
  {1024, 1536, 7168, 8135},  {1024, 3072, 1536, 6251},  {1024, 576, 7168, 12346},  {1024, 7168, 256, 5364},
    {1024, 7168, 2048, 6132},  {1024, 4608, 7168, 7531},  {1024, 7168, 2304, 12345}, {1024, 512, 7168, 6563},
    {1024, 4096, 512, 17512},  {6144, 1536, 7168, 6543},  {6144, 3072, 1536, 234},   {6144, 576, 7168, 9863},
    {6144, 7168, 256, 764243}, {6144, 7168, 2048, 76547}, {6144, 4608, 7168, 65436}, {6144, 7168, 2304, 452345},
    {6144, 512, 7168, 12341},  {6144, 4096, 512, 45245}
    };

// ========= Shape Config =========

const std::pair<int, int> block_shape = {128, 128};
struct BlockwiseMatmulInputs {
  torch::Tensor a;       // float8_e4m3fnuz [m, k]
  torch::Tensor b;       // float8_e4m3fnuz [n, k]
  torch::Tensor a_scale; // float32 [m, k//128]
  torch::Tensor b_scale; // float32 [n//128, k//128]
  torch::Tensor c;       // bfloat16 [m, n]
  torch::Tensor c_ref;   // bfloat16 [m, n]
  int m;
  int n;
  int k;
};

#ifdef TEST_ON_CUDA
#define INPUT_TENSOR_TYPE torch::kBFloat16
#elif TEST_ON_RDNA4
#define INPUT_TENSOR_TYPE torch::kFloat8_e4m3fn
#else
#define INPUT_TENSOR_TYPE torch::kFloat8_e4m3fnuz
#endif

BlockwiseMatmulInputs generate_input(int m, int n, int k, int seed) {
  // Set the random seed for reproducibility
  torch::manual_seed(seed);

  // Calculate scale dimensions
  int block_shape_n = block_shape.first;
  int block_shape_k = block_shape.second;
  int scale_n = (n + block_shape_n - 1) / block_shape_n;
  int scale_k = (k + block_shape_k - 1) / block_shape_k;

  // Create options for each tensor type and device
  auto cuda_options = torch::TensorOptions().device(torch::kCUDA);
  auto fp8_options = cuda_options.dtype(INPUT_TENSOR_TYPE);
  auto bf16_options = cuda_options.dtype(torch::kBFloat16);
  auto fp32_options = cuda_options.dtype(torch::kFloat32);

  // Generate random inputs with FP8 quantization
  // First generate as BF16, then convert to FP8
  auto a_bf16 = torch::randn({k, m}, bf16_options);
  auto b_bf16 = torch::randn({k, n}, bf16_options);

  // Convert to FP8 format
  auto a = a_bf16.to(INPUT_TENSOR_TYPE).transpose(0, 1);
  auto b = b_bf16.to(INPUT_TENSOR_TYPE).transpose(0, 1);

  // Generate scaling factors with FP32
  auto a_scale = torch::randn({scale_k, m}, fp32_options).transpose(0, 1);
  // auto a_scale = torch::ones({scale_k, m}, fp32_options).transpose(0, 1);
  auto b_scale = torch::randn({scale_k, scale_n}, fp32_options).transpose(0, 1);
  // auto b_scale = torch::ones({scale_k, scale_n}, fp32_options).transpose(0, 1);

  // Initialize output tensor with zeros
  auto c = torch::zeros({m, n}, bf16_options);
  auto c_ref = torch::zeros({m, n}, bf16_options);
  return {a, b, a_scale, b_scale, c, c_ref, m, n, k};
}
typedef void (*run_t)(void *, void *, void *, void *, void *, int, int, int, PerfMetrics*, hipStream_t);
typedef void (*init_t)(void);
run_t run_func;
init_t init_func;
void *handle = nullptr;

void ref_kernel(const BlockwiseMatmulInputs &data) {
  // Unpack input data
  auto a = data.a.contiguous();
  auto b = data.b.contiguous();
  auto a_scale = data.a_scale.contiguous();
  auto b_scale = data.b_scale.contiguous();
  auto c = data.c_ref; // Pre-allocated memory

  // Constants
  int m = a.size(0);
  int n = b.size(0);
  int k = a.size(1);
  int block_shape_n = 128;
  int block_shape_k = 128;
  int scale_n = b_scale.size(0);
  int scale_k = b_scale.size(1);

  // Apply blockwise scaling to input 'a'
  // Equivalent to: a_scale = a_scale.unsqueeze(-1).repeat(1, 1, block_shape_k)
  auto a_scale_expanded = a_scale.unsqueeze(-1).repeat({1, 1, block_shape_k});

  // Equivalent to: a_scale = a_scale.reshape(m, scale_k * block_shape_k)
  a_scale_expanded = a_scale_expanded.reshape({m, scale_k * block_shape_k});

  // Equivalent to: a_scale = a_scale[:, :k]
  a_scale_expanded = a_scale_expanded.slice(1, 0, k);

  // Dequantize 'a'
  // Equivalent to: a = a.to(a_scale.dtype) * a_scale
  auto a_dequantized = a.to(a_scale_expanded.dtype()) * a_scale_expanded;

  // Apply blockwise scaling to input 'b'
  // First reshape b_scale to a 1D tensor and repeat
  auto b_scale_flat = b_scale.view({-1, 1});
  b_scale_flat = b_scale_flat.repeat({1, block_shape_n * block_shape_k});

  // Reshape to 4D tensor
  auto b_scale_4d = b_scale_flat.view({scale_n, scale_k, block_shape_n, block_shape_k});

  // Permute dimensions: [scale_n, scale_k, block_shape_n, block_shape_k] ->
  // [scale_n, block_shape_n, scale_k, block_shape_k]
  b_scale_4d = b_scale_4d.permute({0, 2, 1, 3});

  // Reshape to 2D tensor
  auto b_scale_2d = b_scale_4d.reshape({scale_n * block_shape_n, scale_k * block_shape_k});

  // Slice to match dimensions
  auto b_scale_final = b_scale_2d.slice(0, 0, n).slice(1, 0, k);

  // Dequantize 'b'
  auto b_dequantized = b.to(b_scale_final.dtype()) * b_scale_final;

  // Compute GEMM: c = a @ b.T
  // First, compute the matrix multiplication
  auto result = torch::matmul(a_dequantized, b_dequantized.transpose(0, 1));

  // Convert to bfloat16 and store in pre-allocated output tensor
  c.copy_(result.to(torch::kBFloat16));
}

void case_initialize() {
  // Load the symbol
  void *handle = dlopen("libgemm.so", RTLD_LAZY);
  if (!handle) {
    std::cerr << "Cannot open library: " << dlerror() << std::endl;
    abort();
  }
  // init_func = (init_t)dlsym(handle, "init_workspace");
  run_func = (run_t)dlsym(handle, "run");
  // init_func();
  if (!run_func) {
    std::cerr << "Cannot load symbol 'run': " << dlerror() << std::endl;
    dlclose(handle);
    abort();
  }
}
int get_params_count() { return sizeof(shapes) / sizeof(shapes[0]); }
void *case_get_input(int index) {
  auto shape = shapes[index];
  BlockwiseMatmulInputs *input = new BlockwiseMatmulInputs();
  *input = generate_input(shape.m, shape.n, shape.k, shape.seed);
  return (void *)input;
}
std::vector<Checkee> case_run_kernel(void *input, PerfMetrics* metrics) {
  BlockwiseMatmulInputs *data = (BlockwiseMatmulInputs *)input;
  run_func(data->a.data_ptr(), data->b.data_ptr(), data->a_scale.data_ptr(), data->b_scale.data_ptr(),
           data->c.data_ptr(), data->m, data->n, data->k, metrics, 0);
  hipDeviceSynchronize();
  return {Checkee{&data->c, CheckerMode::kElementWise}};
}
std::vector<Checkee> case_run_ref_kernel(void *input) {
  BlockwiseMatmulInputs *data = (BlockwiseMatmulInputs *)input;
  ref_kernel(*data);
  return {Checkee{&data->c_ref, CheckerMode::kElementWise}};
}
const char *case_get_name() { return "GEMM"; }
void get_error_tolerance(float *rtol, float *atol) {
  *rtol = 2e-2;
  *atol = 1e-3;
}
void case_destroy(void *input) {
  BlockwiseMatmulInputs *data = (BlockwiseMatmulInputs *)input;
  delete data;
}
CheckerMode get_checker_mode() { return CheckerMode::kElementWise; }