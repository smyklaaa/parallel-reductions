#include "cuda_vector_add.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <vector>

static void checkCuda(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

__global__ void vectorAddKernel(const float* a, const float* b, float* c, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

void runCudaVectorAddTest() {
    const int size = 1024;
    const int bytes = size * sizeof(float);

    std::vector<float> hostA(size, 1.0f);
    std::vector<float> hostB(size, 2.0f);
    std::vector<float> hostC(size, 0.0f);

    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceC = nullptr;

    checkCuda(cudaMalloc(&deviceA, bytes), "cudaMalloc deviceA failed");
    checkCuda(cudaMalloc(&deviceB, bytes), "cudaMalloc deviceB failed");
    checkCuda(cudaMalloc(&deviceC, bytes), "cudaMalloc deviceC failed");

    checkCuda(cudaMemcpy(deviceA, hostA.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy hostA failed");
    checkCuda(cudaMemcpy(deviceB, hostB.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy hostB failed");

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    vectorAddKernel<<<gridSize, blockSize>>>(deviceA, deviceB, deviceC, size);
    checkCuda(cudaGetLastError(), "vectorAddKernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

    checkCuda(cudaMemcpy(hostC.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy deviceC failed");

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);

    for (int i = 0; i < size; i++) {
        if (hostC[i] != 3.0f) {
            throw std::runtime_error("CUDA vector add test failed.");
        }
    }

    std::cout << "CUDA vector add test passed.\n";
}