#include "ck/ck.hpp"
#include "ck/utility/data_type.hpp"

#include <hip/hip_runtime.h>
#include <ck/ck.hpp>
#include <hipcub/hipcub.hpp>
#include <iostream>
#include <rocblas/internal/rocblas-types.h>
#include <rocblas/rocblas.h>
#include "../include/clangd_workaround.h"

#define PARAMETERIZE_LIBRARY
#include "../src/moe/moe_gemm_kernel.cpp"

HOST_CODE_BELOW

#define HIP_CHECK(status)                                                                                              \
  if (status != hipSuccess) {                                                                                          \
    std::cerr << "Error at line " << __LINE__ << ": " << hipGetErrorString(status) << std::endl;                       \
    exit(EXIT_FAILURE);                                                                                                \
  }

constexpr int experts_count = 5;
int SEQ_LEN_EXPANDED = 1 + 2 + 4 + 7 + 8;
int first_token_offset[experts_count + 1] = {0, 1, 3, 7, 14, 22};
constexpr int D_HIDDEN = 7168;
constexpr int D_EXPERT = 2048;

ck::half_t *A, *C, *A_d, *C_d;
ck::half_t **experts_ptr, **experts_device_ptr, **experts_ptr_d;
int *first_token_offset_d;

template <ck::index_t... Is> using S = ck::Sequence<Is...>;
void prepare_data() {

  A = (ck::half_t *)malloc(SEQ_LEN_EXPANDED * D_HIDDEN * sizeof(ck::half_t));
  HIP_CHECK(hipMalloc(&A_d, SEQ_LEN_EXPANDED * D_HIDDEN * sizeof(ck::half_t)));

  experts_ptr = (ck::half_t **)malloc(experts_count * sizeof(ck::half_t *));
  experts_device_ptr = (ck::half_t **)malloc(experts_count * sizeof(ck::half_t *));

  for (int i = 0; i < experts_count; i++) {
    experts_ptr[i] = (ck::half_t *)malloc(D_EXPERT * D_HIDDEN * sizeof(ck::half_t));
    HIP_CHECK(hipMalloc(&experts_device_ptr[i], D_EXPERT * D_HIDDEN * sizeof(ck::half_t)));
  }

  HIP_CHECK(hipMalloc(&experts_ptr_d, experts_count * sizeof(ck::half_t *)));
  HIP_CHECK(hipMemcpy(experts_ptr_d, experts_device_ptr, experts_count * sizeof(ck::half_t *), hipMemcpyHostToDevice));

  ck::half_t *C = (ck::half_t *)malloc(SEQ_LEN_EXPANDED * D_EXPERT * sizeof(ck::half_t));

  for (int expert_idx = 0; expert_idx < experts_count; expert_idx++) {
    for (int i = first_token_offset[expert_idx]; i < first_token_offset[expert_idx + 1]; i++) {
      for (int j = 0; j < D_HIDDEN; j++) {
        A[i * D_HIDDEN + j] = ck::half_t(expert_idx + 1);
      }
    }
    for (int i = 0; i < D_EXPERT; i++) {
      for (int j = 0; j < D_HIDDEN; j++) {
        experts_ptr[expert_idx][i * D_HIDDEN + j] = ck::half_t(expert_idx + 1);
      }
    }
    HIP_CHECK(hipMemcpy(experts_device_ptr[expert_idx], experts_ptr[expert_idx],
                        D_EXPERT * D_HIDDEN * sizeof(ck::half_t), hipMemcpyHostToDevice));
  }

  HIP_CHECK(hipMemcpy(A_d, A, SEQ_LEN_EXPANDED * D_HIDDEN * sizeof(ck::half_t), hipMemcpyHostToDevice));

  memset(C, 0, SEQ_LEN_EXPANDED * D_EXPERT * sizeof(ck::half_t));
  HIP_CHECK(hipMalloc(&C_d, SEQ_LEN_EXPANDED * D_EXPERT * sizeof(ck::half_t)));
  HIP_CHECK(hipMemcpy(C_d, C, SEQ_LEN_EXPANDED * D_EXPERT * sizeof(ck::half_t), hipMemcpyHostToDevice));

  HIP_CHECK(hipMalloc(&first_token_offset_d, (experts_count + 1) * sizeof(int)));
  HIP_CHECK(
      hipMemcpy(first_token_offset_d, first_token_offset, (experts_count + 1) * sizeof(int), hipMemcpyHostToDevice));
}

void output_tensor(ck::half_t *tensor, int rows, int cols, std::pair<int, int> row_range,
                   std::pair<int, int> col_range) {
  ck::half_t *host_tensor = (ck::half_t *)malloc(rows * cols * sizeof(ck::half_t));
  HIP_CHECK(hipMemcpy(host_tensor, tensor, rows * cols * sizeof(ck::half_t), hipMemcpyDeviceToHost));

  std::cout << "Tensor output (" << rows << "x" << cols << "), showing rows " << row_range.first << "-"
            << row_range.second << ", cols " << col_range.first << "-" << col_range.second << ":" << std::endl;

  for (int i = row_range.first; i < std::min(row_range.second, rows); i++) {
    for (int j = col_range.first; j < std::min(col_range.second, cols); j++) {
      std::cout << float(host_tensor[i * cols + j]) << " ";
    }
    std::cout << std::endl;
  }

  free(host_tensor);
}

void do_compute() {

  // auto target_ptr =
  //     const_cast<const ck::half_t(**)[D_EXPERT]>(reinterpret_cast<ck::half_t(**)[D_EXPERT]>(experts_ptr_d));

  // moe_gemm_scheduler_entry<ck::half_t, ck::half_t, experts_count, D_HIDDEN, D_EXPERT, 64, 64, 32, 1, 256>
  //     <<<1, 256>>>(A_d, target_ptr, C_d, first_token_offset_d);
}

int main() {
  prepare_data();
  do_compute();
  output_tensor(C_d, SEQ_LEN_EXPANDED, D_EXPERT, {0, 10}, {0, 10});

  return 0;
}
