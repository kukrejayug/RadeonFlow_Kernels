#include <hip/hip_runtime.h>
#include <iostream>
#include <stdexcept>
#include "../include/clangd_workaround.h"

HOST_CODE_BELOW

#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t error = cmd;                                                    \
    if (error != hipSuccess) {                                                 \
      std::cerr << "HIP Error: " << hipGetErrorString(error) << " at line "    \
                << __LINE__ << " in file " << __FILE__ << std::endl;           \
      throw std::runtime_error("HIP error");                                   \
    }                                                                          \
  } while (0)

int main() {
  int deviceCount = 0;
  HIP_CHECK(hipGetDeviceCount(&deviceCount));

  if (deviceCount == 0) {
    std::cerr << "No HIP devices found!" << std::endl;
    return 1;
  }

  std::cout << "Found " << deviceCount << " HIP device(s)." << std::endl;

  for (int i = 0; i < deviceCount; ++i) {
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, i));

    // Query additional attributes needed early
    int wavefrontSize = 0;
    HIP_CHECK(hipDeviceGetAttribute(&wavefrontSize, hipDeviceAttributeWarpSize, i));

    int maxSharedMemoryPerMultiProcessor = 0;
    HIP_CHECK(hipDeviceGetAttribute(&maxSharedMemoryPerMultiProcessor, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, i));

    std::cout << "\n--- Device " << i << ": " << props.name << " ---"
              << std::endl;

    // Grouped memory and core properties
    std::cout << "  Total global memory:         "
              << props.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Shared memory per block:     "
              << props.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "  Max shared memory per MP:    " << maxSharedMemoryPerMultiProcessor / 1024 << " KB" << std::endl;
    std::cout << "  L2 cache size:               " << props.l2CacheSize / 1024
              << " KB" << std::endl;
    std::cout << "  Registers per block:         " << props.regsPerBlock
              << std::endl;
    std::cout << "  Warp/Wavefront size:         " << wavefrontSize << std::endl;

    // Remaining properties
    std::cout << "  Warp size (from props):      " << props.warpSize
              << std::endl;
    std::cout << "  Max threads per block:       " << props.maxThreadsPerBlock
              << std::endl;
    std::cout << "  Max threads dimensions:      (" << props.maxThreadsDim[0]
              << ", " << props.maxThreadsDim[1] << ", "
              << props.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "  Max grid size:               (" << props.maxGridSize[0]
              << ", " << props.maxGridSize[1] << ", " << props.maxGridSize[2]
              << ")" << std::endl;
    std::cout << "  Clock rate:                  " << props.clockRate / 1000
              << " MHz" << std::endl;
    std::cout << "  Memory clock rate:           "
              << props.memoryClockRate / 1000 << " MHz" << std::endl;
    std::cout << "  Memory bus width:            " << props.memoryBusWidth
              << " bits" << std::endl;
    std::cout << "  Total constant memory:       " << props.totalConstMem / 1024
              << " KB" << std::endl;
    std::cout << "  Compute capability (major.minor): " << props.major << "."
              << props.minor << std::endl;
    std::cout << "  Multiprocessor count:        " << props.multiProcessorCount
              << std::endl;
    std::cout << "  Max threads per multiprocessor: "
              << props.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  Compute mode:                " << props.computeMode
              << std::endl;
    std::cout << "  Concurrent kernels:          "
              << (props.concurrentKernels ? "Yes" : "No") << std::endl;
    std::cout << "  PCI bus ID:                  " << props.pciBusID
              << std::endl;
    std::cout << "  PCI device ID:               " << props.pciDeviceID
              << std::endl;
    std::cout << "  PCI domain ID:               " << props.pciDomainID
              << std::endl;
    std::cout << "  Is integrated:               "
              << (props.integrated ? "Yes" : "No") << std::endl;
    std::cout << "  Can map host memory:         "
              << (props.canMapHostMemory ? "Yes" : "No") << std::endl;
    std::cout << "  GCN Arch Name:               " << props.gcnArchName
              << std::endl;

    int asyncEngineCount = 0;
    HIP_CHECK(hipDeviceGetAttribute(&asyncEngineCount, hipDeviceAttributeAsyncEngineCount, i));
    std::cout << "  Async engine count:          " << asyncEngineCount << std::endl;

    int cooperativeLaunch = 0;
    HIP_CHECK(hipDeviceGetAttribute(&cooperativeLaunch, hipDeviceAttributeCooperativeLaunch, i));
    std::cout << "  Cooperative launch support:  " << (cooperativeLaunch ? "Yes" : "No") << std::endl;

    int cooperativeMultiDeviceLaunch = 0;
    HIP_CHECK(hipDeviceGetAttribute(&cooperativeMultiDeviceLaunch, hipDeviceAttributeCooperativeMultiDeviceLaunch, i));
    std::cout << "  Cooperative multi-device launch: " << (cooperativeMultiDeviceLaunch ? "Yes" : "No") << std::endl;

  }

  return 0;
}
