#include "cuda_sum.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

static void checkCudaStatus(cudaError_t status, const char* message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
static std::vector<T> generateCudaInputData(std::size_t size) {
    std::vector<T> data(size);
    std::mt19937_64 generator(12345);

    if constexpr (std::is_integral_v<T>) {
        std::uniform_int_distribution<std::int64_t> distribution(1, 100);

        for (std::size_t i = 0; i < size; i++) {
            data[i] = static_cast<T>(distribution(generator));
        }
    } else {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);

        for (std::size_t i = 0; i < size; i++) {
            data[i] = static_cast<T>(distribution(generator));
        }
    }

    return data;
}

template <typename T>
__global__ void cudaSumKernel(const T* input, T* partialSums, std::size_t size) {
    extern __shared__ unsigned char sharedMemoryRaw[];
    T* sharedMemory = reinterpret_cast<T*>(sharedMemoryRaw);

    unsigned int threadId = threadIdx.x;
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    T localSum = static_cast<T>(0);

    for (std::size_t i = globalIndex; i < size; i += stride) {
        localSum += input[i];
    }

    sharedMemory[threadId] = localSum;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadId < offset) {
            sharedMemory[threadId] += sharedMemory[threadId + offset];
        }

        __syncthreads();
    }

    if (threadId == 0) {
        partialSums[blockIdx.x] = sharedMemory[0];
    }
}

template <typename T>
static BenchmarkResult runCudaSumTyped(const AppConfig& config) {
    BenchmarkResult result;
    result.backend = "cuda";
    result.operation = "sum";
    result.dataType = toString(config.dataType);
    result.size = config.size;
    result.threads = 1;
    result.processes = 1;
    result.blockSize = config.blockSize;

    std::vector<T> hostInput = generateCudaInputData<T>(config.size);

    int deviceCount = 0;
    checkCudaStatus(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount failed");

    if (deviceCount == 0) {
        throw std::runtime_error("No CUDA-capable NVIDIA GPU detected.");
    }

    int blockSize = config.blockSize;
    int gridSize = static_cast<int>((config.size + blockSize - 1) / blockSize);

    if (gridSize > 4096) {
        gridSize = 4096;
    }

    std::vector<T> hostPartialSums(gridSize);

    T* deviceInput = nullptr;
    T* devicePartialSums = nullptr;

    std::size_t inputBytes = config.size * sizeof(T);
    std::size_t partialBytes = gridSize * sizeof(T);
    std::size_t sharedMemorySize = blockSize * sizeof(T);

    auto totalStart = std::chrono::high_resolution_clock::now();

    checkCudaStatus(cudaMalloc(&deviceInput, inputBytes), "cudaMalloc deviceInput failed");
    checkCudaStatus(cudaMalloc(&devicePartialSums, partialBytes), "cudaMalloc devicePartialSums failed");

    auto transferStart = std::chrono::high_resolution_clock::now();
    checkCudaStatus(cudaMemcpy(deviceInput, hostInput.data(), inputBytes, cudaMemcpyHostToDevice), "cudaMemcpy input failed");
    auto transferAfterHostToDevice = std::chrono::high_resolution_clock::now();

    auto computeStart = std::chrono::high_resolution_clock::now();
    cudaSumKernel<T><<<gridSize, blockSize, sharedMemorySize>>>(deviceInput, devicePartialSums, config.size);
    checkCudaStatus(cudaGetLastError(), "cudaSumKernel launch failed");
    checkCudaStatus(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    auto computeAfterKernel = std::chrono::high_resolution_clock::now();

    auto transferDeviceToHostStart = std::chrono::high_resolution_clock::now();
    checkCudaStatus(cudaMemcpy(hostPartialSums.data(), devicePartialSums, partialBytes, cudaMemcpyDeviceToHost), "cudaMemcpy partial sums failed");
    auto transferEnd = std::chrono::high_resolution_clock::now();

    auto computeCpuFinishStart = std::chrono::high_resolution_clock::now();
    T gpuSum = std::accumulate(hostPartialSums.begin(), hostPartialSums.end(), static_cast<T>(0));
    auto computeEnd = std::chrono::high_resolution_clock::now();

    auto verificationStart = std::chrono::high_resolution_clock::now();
    T cpuSum = std::accumulate(hostInput.begin(), hostInput.end(), static_cast<T>(0));

    double gpuValue = static_cast<double>(gpuSum);
    double cpuValue = static_cast<double>(cpuSum);
    double absoluteError = std::abs(gpuValue - cpuValue);
    double relativeError = 0.0;

    if (std::abs(cpuValue) > 0.0) {
        relativeError = absoluteError / std::abs(cpuValue);
    }

    auto verificationEnd = std::chrono::high_resolution_clock::now();

    checkCudaStatus(cudaFree(deviceInput), "cudaFree deviceInput failed");
    checkCudaStatus(cudaFree(devicePartialSums), "cudaFree devicePartialSums failed");

    auto totalEnd = std::chrono::high_resolution_clock::now();

    double hostToDeviceMs = std::chrono::duration<double, std::milli>(transferAfterHostToDevice - transferStart).count();
    double deviceToHostMs = std::chrono::duration<double, std::milli>(transferEnd - transferDeviceToHostStart).count();
    double kernelMs = std::chrono::duration<double, std::milli>(computeAfterKernel - computeStart).count();
    double cpuFinishMs = std::chrono::duration<double, std::milli>(computeEnd - computeCpuFinishStart).count();

    result.timeMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
    result.computeTimeMs = kernelMs + cpuFinishMs;
    result.transferTimeMs = hostToDeviceMs + deviceToHostMs;
    result.verificationTimeMs = std::chrono::duration<double, std::milli>(verificationEnd - verificationStart).count();
    result.throughputGBs = (static_cast<double>(config.size * sizeof(T)) / 1e9) / (result.timeMs / 1000.0);
    result.absoluteError = absoluteError;
    result.relativeError = relativeError;
    result.resultSummary =
        "gpu_sum = " + std::to_string(gpuValue) +
        ", cpu_sum = " + std::to_string(cpuValue);

    return result;
}

BenchmarkResult runCudaSum(const AppConfig& config) {
    if (config.dataType == DataType::Int64) {
        return runCudaSumTyped<std::int64_t>(config);
    }

    if (config.dataType == DataType::Float) {
        return runCudaSumTyped<float>(config);
    }

    if (config.dataType == DataType::Double) {
        return runCudaSumTyped<double>(config);
    }

    throw std::runtime_error("Unsupported CUDA data type.");
}