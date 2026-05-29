#include "cuda_minmax.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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

template <typename T, bool FindMin>
__global__ void cudaMinMaxKernel(const T* input, T* partialResults, std::size_t size) {
    extern __shared__ unsigned char sharedMemoryRaw[];
    T* sharedMemory = reinterpret_cast<T*>(sharedMemoryRaw);

    unsigned int threadId = threadIdx.x;
    unsigned int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;

    T localValue = input[0];

    for (std::size_t i = globalIndex; i < size; i += stride) {
        if constexpr (FindMin) {
            if (input[i] < localValue) {
                localValue = input[i];
            }
        } else {
            if (input[i] > localValue) {
                localValue = input[i];
            }
        }
    }

    sharedMemory[threadId] = localValue;
    __syncthreads();

    for (unsigned int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadId < offset) {
            if constexpr (FindMin) {
                if (sharedMemory[threadId + offset] < sharedMemory[threadId]) {
                    sharedMemory[threadId] = sharedMemory[threadId + offset];
                }
            } else {
                if (sharedMemory[threadId + offset] > sharedMemory[threadId]) {
                    sharedMemory[threadId] = sharedMemory[threadId + offset];
                }
            }
        }

        __syncthreads();
    }

    if (threadId == 0) {
        partialResults[blockIdx.x] = sharedMemory[0];
    }
}

template <typename T, bool FindMin>
static BenchmarkResult runCudaMinMaxTyped(const AppConfig& config) {
    BenchmarkResult result;
    result.backend = "cuda";
    result.operation = FindMin ? "min" : "max";
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

    std::vector<T> hostPartialResults(gridSize);

    T* deviceInput = nullptr;
    T* devicePartialResults = nullptr;

    std::size_t inputBytes = config.size * sizeof(T);
    std::size_t partialBytes = gridSize * sizeof(T);
    std::size_t sharedMemorySize = blockSize * sizeof(T);

    auto totalStart = std::chrono::high_resolution_clock::now();

    checkCudaStatus(cudaMalloc(&deviceInput, inputBytes), "cudaMalloc deviceInput failed");
    checkCudaStatus(cudaMalloc(&devicePartialResults, partialBytes), "cudaMalloc devicePartialResults failed");

    auto transferHostToDeviceStart = std::chrono::high_resolution_clock::now();
    checkCudaStatus(cudaMemcpy(deviceInput, hostInput.data(), inputBytes, cudaMemcpyHostToDevice), "cudaMemcpy input failed");
    auto transferHostToDeviceEnd = std::chrono::high_resolution_clock::now();

    auto computeKernelStart = std::chrono::high_resolution_clock::now();
    cudaMinMaxKernel<T, FindMin><<<gridSize, blockSize, sharedMemorySize>>>(deviceInput, devicePartialResults, config.size);
    checkCudaStatus(cudaGetLastError(), "cudaMinMaxKernel launch failed");
    checkCudaStatus(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    auto computeKernelEnd = std::chrono::high_resolution_clock::now();

    auto transferDeviceToHostStart = std::chrono::high_resolution_clock::now();
    checkCudaStatus(cudaMemcpy(hostPartialResults.data(), devicePartialResults, partialBytes, cudaMemcpyDeviceToHost), "cudaMemcpy partial results failed");
    auto transferDeviceToHostEnd = std::chrono::high_resolution_clock::now();

    auto computeCpuStart = std::chrono::high_resolution_clock::now();

    T gpuValue;

    if constexpr (FindMin) {
        gpuValue = *std::min_element(hostPartialResults.begin(), hostPartialResults.end());
    } else {
        gpuValue = *std::max_element(hostPartialResults.begin(), hostPartialResults.end());
    }

    auto computeCpuEnd = std::chrono::high_resolution_clock::now();

    auto verificationStart = std::chrono::high_resolution_clock::now();

    T cpuValue;

    if constexpr (FindMin) {
        cpuValue = *std::min_element(hostInput.begin(), hostInput.end());
    } else {
        cpuValue = *std::max_element(hostInput.begin(), hostInput.end());
    }

    double gpuDouble = static_cast<double>(gpuValue);
    double cpuDouble = static_cast<double>(cpuValue);
    double absoluteError = std::abs(gpuDouble - cpuDouble);
    double relativeError = 0.0;

    if (std::abs(cpuDouble) > 0.0) {
        relativeError = absoluteError / std::abs(cpuDouble);
    }

    auto verificationEnd = std::chrono::high_resolution_clock::now();

    checkCudaStatus(cudaFree(deviceInput), "cudaFree deviceInput failed");
    checkCudaStatus(cudaFree(devicePartialResults), "cudaFree devicePartialResults failed");

    auto totalEnd = std::chrono::high_resolution_clock::now();

    double hostToDeviceMs = std::chrono::duration<double, std::milli>(transferHostToDeviceEnd - transferHostToDeviceStart).count();
    double deviceToHostMs = std::chrono::duration<double, std::milli>(transferDeviceToHostEnd - transferDeviceToHostStart).count();
    double kernelMs = std::chrono::duration<double, std::milli>(computeKernelEnd - computeKernelStart).count();
    double cpuFinishMs = std::chrono::duration<double, std::milli>(computeCpuEnd - computeCpuStart).count();

    result.timeMs = std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
    result.computeTimeMs = kernelMs + cpuFinishMs;
    result.transferTimeMs = hostToDeviceMs + deviceToHostMs;
    result.verificationTimeMs = std::chrono::duration<double, std::milli>(verificationEnd - verificationStart).count();
    result.throughputGBs = (static_cast<double>(config.size * sizeof(T)) / 1e9) / (result.timeMs / 1000.0);
    result.absoluteError = absoluteError;
    result.relativeError = relativeError;

    if constexpr (FindMin) {
        result.resultSummary =
            "gpu_min = " + std::to_string(gpuDouble) +
            ", cpu_min = " + std::to_string(cpuDouble);
    } else {
        result.resultSummary =
            "gpu_max = " + std::to_string(gpuDouble) +
            ", cpu_max = " + std::to_string(cpuDouble);
    }

    return result;
}

BenchmarkResult runCudaMin(const AppConfig& config) {
    if (config.dataType == DataType::Int64) {
        return runCudaMinMaxTyped<std::int64_t, true>(config);
    }

    if (config.dataType == DataType::Float) {
        return runCudaMinMaxTyped<float, true>(config);
    }

    if (config.dataType == DataType::Double) {
        return runCudaMinMaxTyped<double, true>(config);
    }

    throw std::runtime_error("Unsupported CUDA data type.");
}

BenchmarkResult runCudaMax(const AppConfig& config) {
    if (config.dataType == DataType::Int64) {
        return runCudaMinMaxTyped<std::int64_t, false>(config);
    }

    if (config.dataType == DataType::Float) {
        return runCudaMinMaxTyped<float, false>(config);
    }

    if (config.dataType == DataType::Double) {
        return runCudaMinMaxTyped<double, false>(config);
    }

    throw std::runtime_error("Unsupported CUDA data type.");
}