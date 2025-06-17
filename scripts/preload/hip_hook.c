#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <stdint.h>

// HIP types definitions
typedef struct dim3 {
    uint32_t x;
    uint32_t y;
    uint32_t z;
} dim3;

typedef void *hipStream_t;

// Function pointer type for the original hipLaunchKernel
typedef int (*hipLaunchKernel_func_t)(const void *function_address, dim3 numBlocks, dim3 dimBlocks, void **args,
                                      size_t sharedMemBytes, hipStream_t stream);

// Global variable to store the original function pointer
static hipLaunchKernel_func_t original_hipLaunchKernel = NULL;
static void *hip_handle = NULL;

// Initialize the hook - load the original library and get function pointer
static void init_hook() {
    if (original_hipLaunchKernel != NULL) {
        return; // Already initialized
    }

    // Load the original HIP library
    hip_handle = dlopen("/opt/rocm/lib/libamdhip64.so", RTLD_LAZY);
    if (!hip_handle) {
        fprintf(stderr, "Error loading libamdhip64.so: %s\n", dlerror());
        exit(1);
    }

    // Get the original function pointer
    original_hipLaunchKernel = (hipLaunchKernel_func_t)dlsym(hip_handle, "hipLaunchKernel");
    if (!original_hipLaunchKernel) {
        fprintf(stderr, "Error finding hipLaunchKernel symbol: %s\n", dlerror());
        dlclose(hip_handle);
        exit(1);
    }

    printf("[HIP HOOK] Successfully loaded original hipLaunchKernel function\n");
}

// Print function arguments in a readable format
static void print_args(const void *function_address, dim3 numBlocks, dim3 dimBlocks, void **args,
                       size_t sharedMemBytes, hipStream_t stream) {
    printf("==================== HIP KERNEL LAUNCH ====================\n");
    printf("Function Address: %p\n", function_address);
    printf("Grid Dimensions (numBlocks): (%u, %u, %u)\n", numBlocks.x, numBlocks.y, numBlocks.z);
    printf("Block Dimensions (dimBlocks): (%u, %u, %u)\n", dimBlocks.x, dimBlocks.y, dimBlocks.z);
    printf("Args Pointer: %p\n", args);
    printf("Shared Memory Bytes: %zu\n", sharedMemBytes);
    printf("Stream: %p\n", stream);


    printf("========================================================\n");
    fflush(stdout);
}

// Our hooked version of hipLaunchKernel
int hipLaunchKernel(const void *function_address, dim3 numBlocks, dim3 dimBlocks, void **args,
                    size_t sharedMemBytes, hipStream_t stream) {
    // Initialize hook if not already done
    init_hook();

    // Only print information if stream is not NULL
    if (stream != NULL) {
        // Print all arguments
        print_args(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream);
    }

    // Call the original function
    int result =
        original_hipLaunchKernel(function_address, numBlocks, dimBlocks, args, sharedMemBytes, stream);

    if (stream != NULL) {
        printf("[HIP HOOK] hipLaunchKernel returned: %d\n", result);
        fflush(stdout);
    }

    return result;
}

// Cleanup function (called when library is unloaded)
__attribute__((destructor)) static void cleanup_hook() {
    if (hip_handle) {
        dlclose(hip_handle);
        printf("[HIP HOOK] Cleanup completed\n");
    }
}