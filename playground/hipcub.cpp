#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <iostream>
#include <random>
#include <vector>
#include "../include/clangd_workaround.h"

HOST_CODE_BELOW

#define HIP_CHECK(status)                                                                                              \
  if (status != hipSuccess) {                                                                                          \
    std::cerr << "Error at line " << __LINE__ << ": " << hipGetErrorString(status) << std::endl;                       \
    exit(EXIT_FAILURE);                                                                                                \
  }

#define HOST_TYPE(func) hip##func

void minitest() {
  {
    size_t required_temp_storage_bytes = 0;

    int *debug_permuted_rows_d, *debug_permuted_experts_d, *debug_permuted_rows_h, *debug_permuted_experts_h,
        *debug_seq_experts_index_h, *debug_seq_experts_index_d, *permuted_rows_d, *source_rows_d;
    HIP_CHECK(HOST_TYPE(Malloc)(&debug_permuted_rows_d, 2048 * sizeof(int)));
    HIP_CHECK(HOST_TYPE(Malloc)(&debug_permuted_experts_d, 2048 * sizeof(int)));
    HIP_CHECK(HOST_TYPE(Malloc)(&debug_seq_experts_index_d, 2048 * sizeof(int)));
    HIP_CHECK(HOST_TYPE(Malloc)(&permuted_rows_d, 2048 * sizeof(int)));
    HIP_CHECK(HOST_TYPE(Malloc)(&source_rows_d, 2048 * sizeof(int)));

    int p[2048];
    for (int i = 0; i < 2048; i++) {
      p[i] = i;
    }
    HIP_CHECK(HOST_TYPE(Memcpy)(debug_seq_experts_index_d, p, 2048 * sizeof(int), hipMemcpyHostToDevice));
    debug_permuted_rows_h = (int *)malloc(2048 * sizeof(int));
    debug_permuted_experts_h = (int *)malloc(2048 * sizeof(int));
    debug_seq_experts_index_h = (int *)malloc(2048 * sizeof(int));
    // LIB_CALL(hipcub::DeviceRadixSort::SortPairs(temp_storage, required_temp_storage_bytes, debug_seq_experts_index_d,
    //                                             debug_permuted_experts_d, source_rows, debug_permuted_rows_d, 2048,
    //                                             0)); // TODO: specify bits to speed up.
    void *temp_storage = nullptr;
    HIP_CHECK(hipcub::DeviceRadixSort::SortPairs(temp_storage, required_temp_storage_bytes, debug_seq_experts_index_d,
                                                debug_permuted_experts_d, source_rows_d, permuted_rows_d, 2048,
                                                0)); // TODO: specify bits to speed up.
    HIP_CHECK(HOST_TYPE(Malloc)(&temp_storage, required_temp_storage_bytes));
    HIP_CHECK(hipcub::DeviceRadixSort::SortPairs(temp_storage, required_temp_storage_bytes, debug_seq_experts_index_d,
                                                debug_permuted_experts_d, source_rows_d, permuted_rows_d, 2048,
                                                0)); // TODO: specify bits to speed up.
    HIP_CHECK(hipMemcpy(debug_permuted_rows_h, debug_permuted_rows_d, 2048 * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(debug_permuted_experts_h, debug_permuted_experts_d, 2048 * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(debug_seq_experts_index_h, debug_seq_experts_index_d, 2048 * sizeof(int), hipMemcpyDeviceToHost));
    std::cout << "-------" << std::endl;
    std::cout << std::endl;
    for (int i = 0; i < 2048; i++) {
      std::cout << debug_permuted_experts_h[i] << ", ";
    }
  }
}

int main() {
  minitest();

  return 0;
}
