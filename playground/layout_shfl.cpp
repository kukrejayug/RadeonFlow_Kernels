#include <cuda_runtime.h>
#include <iostream>
#include "../include/clangd_workaround.h"

__global__ void warp_relayout_shfl(const float *A, float *B) {
    // Assume blockDim.x == 32, each warp processes a 16×16 block
    int i = threadIdx.x;           // 0..31
    int start_row = (i & 16) >> 1; // 0 or 8
    int col       = i & 15;        // 0..15
    float v[8], out[8];
    // 1) load 8 elements
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        v[k] = A[(start_row + k) * 16 + col];
    }
    unsigned mask = 0xffffffff;
    // 2) shuffle: move the original k dimension along rows to k dimension along columns
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        int tgt = ((i & 16) >> 1) | k;  
        out[k] = __shfl_sync(mask, v[k], tgt);
    }
    // 3) store back to B (in new contiguous layout for the last dimension)
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        // Each thread writes B[i][k]
        B[i * 8 + k] = out[k];
    }
}

HOST_CODE_BELOW

// Host-side example call
void launch() {
    // Define host arrays and fill them
    float h_A[16*16], h_B[32*8];
    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 16; ++c) {
            h_A[r*16 + c] = r*16 + c;
        }
    }
    // Allocate device memory and copy
    float *d_A, *d_B;
    cudaMalloc(&d_A, 16*16*sizeof(float));
    cudaMalloc(&d_B, 32*8*sizeof(float));
    cudaMemcpy(d_A, h_A, 16*16*sizeof(float), cudaMemcpyHostToDevice);
    warp_relayout_shfl<<<1,32>>>(d_A, d_B);
    cudaMemcpy(h_B, d_B, 32*8*sizeof(float), cudaMemcpyDeviceToHost);
    // Print results
    std::cout << "Output B (32 threads x 8 elements):\n";
    for (int i = 0; i < 32; ++i) {
        for (int k = 0; k < 8; ++k) {
            std::cout << h_B[i*8 + k] << " ";
        }
        std::cout << "\n";
    }
    cudaFree(d_A);
    cudaFree(d_B);
}

int main() {
    launch();
    return 0;
}