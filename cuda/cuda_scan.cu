#include "cuda_scan.hpp"

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

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
static BenchmarkResult runCudaScanTyped(const AppConfig& config) {
    BenchmarkResult result;
    result.backend = "cuda";
    result.operation = "scan";
    result.dataType = toString(config.dataType);
    result.size = config.size;

    std::vector<T> hostInput = generateCudaInputData<T>(config.size);
    std::vector<T> cpuOutput(config.size);
    std::vector<T> gpuOutput(config.size);

    std::partial_sum(hostInput.begin(), hostInput.end(), cpuOutput.begin());

    int deviceCount = 0;
    checkCudaStatus(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount failed");

    if (deviceCount == 0) {
        throw std::runtime_error("No CUDA-capable NVIDIA GPU detected.");
    }

    auto start = std::chrono::high_resolution_clock::now();

    thrust::device_vector<T> deviceInput(hostInput.begin(), hostInput.end());
    thrust::device_vector<T> deviceOutput(config.size);

    thrust::inclusive_scan(deviceInput.begin(), deviceInput.end(), deviceOutput.begin());

    thrust::copy(deviceOutput.begin(), deviceOutput.end(), gpuOutput.begin());

    auto end = std::chrono::high_resolution_clock::now();

    double maxAbsoluteError = 0.0;
    double maxRelativeError = 0.0;

    for (std::size_t i = 0; i < config.size; i++) {
        double gpuValue = static_cast<double>(gpuOutput[i]);
        double cpuValue = static_cast<double>(cpuOutput[i]);
        double absoluteError = std::abs(gpuValue - cpuValue);
        double relativeError = 0.0;

        if (std::abs(cpuValue) > 0.0) {
            relativeError = absoluteError / std::abs(cpuValue);
        }

        if (absoluteError > maxAbsoluteError) {
            maxAbsoluteError = absoluteError;
        }

        if (relativeError > maxRelativeError) {
            maxRelativeError = relativeError;
        }
    }

    double gpuFirst = static_cast<double>(gpuOutput.front());
    double gpuLast = static_cast<double>(gpuOutput.back());
    double cpuFirst = static_cast<double>(cpuOutput.front());
    double cpuLast = static_cast<double>(cpuOutput.back());

    result.timeMs = std::chrono::duration<double, std::milli>(end - start).count();
    result.throughputGBs = (static_cast<double>(2 * config.size * sizeof(T)) / 1e9) / (result.timeMs / 1000.0);
    result.absoluteError = maxAbsoluteError;
    result.relativeError = maxRelativeError;
    result.resultSummary =
        "gpu_first = " + std::to_string(gpuFirst) +
        ", cpu_first = " + std::to_string(cpuFirst) +
        ", gpu_last = " + std::to_string(gpuLast) +
        ", cpu_last = " + std::to_string(cpuLast);

    return result;
}

BenchmarkResult runCudaScan(const AppConfig& config) {
    if (config.dataType == DataType::Int64) {
        return runCudaScanTyped<std::int64_t>(config);
    }

    if (config.dataType == DataType::Float) {
        return runCudaScanTyped<float>(config);
    }

    if (config.dataType == DataType::Double) {
        return runCudaScanTyped<double>(config);
    }

    throw std::runtime_error("Unsupported CUDA data type.");
}