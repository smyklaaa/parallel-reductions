#include "openmp_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <random>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

template <typename T>
static std::vector<T> generateData(std::size_t size) {
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
static std::string valueToString(T value) {
    std::ostringstream stream;
    stream << std::setprecision(12) << value;
    return stream.str();
}

template <typename T>
static double absoluteDifference(T a, T b) {
    return std::abs(static_cast<double>(a) - static_cast<double>(b));
}

template <typename T>
static double relativeDifference(T a, T b) {
    double absDiff = absoluteDifference(a, b);
    double denominator = std::abs(static_cast<double>(b));

    if (denominator == 0.0) {
        return absDiff;
    }

    return absDiff / denominator;
}

static std::size_t nextPowerOfTwo(std::size_t value) {
    if (value == 0) {
        return 1;
    }

    std::size_t power = 1;

    while (power < value) {
        power *= 2;
    }

    return power;
}

template <typename T>
static long double referenceSumLongDouble(const std::vector<T>& data) {
    long double sum = 0.0L;

    for (T value : data) {
        sum += static_cast<long double>(value);
    }

    return sum;
}

template <typename T>
static double absoluteDifferenceLongDouble(T value, long double reference) {
    return static_cast<double>(
        std::abs(static_cast<long double>(value) - reference)
    );
}

template <typename T>
static double relativeDifferenceLongDouble(T value, long double reference) {
    long double absDiff = std::abs(static_cast<long double>(value) - reference);
    long double denominator = std::abs(reference);

    if (denominator == 0.0L) {
        return static_cast<double>(absDiff);
    }

    return static_cast<double>(absDiff / denominator);
}

template <typename T>
static std::vector<T> blellochInclusiveScanOpenMP(const std::vector<T>& input) {
    const std::size_t originalSize = input.size();
    const std::size_t paddedSize = nextPowerOfTwo(originalSize);

    std::vector<T> scanData(paddedSize, static_cast<T>(0));

    #pragma omp parallel for
    for (std::size_t i = 0; i < originalSize; i++) {
        scanData[i] = input[i];
    }

    for (std::size_t stride = 1; stride < paddedSize; stride *= 2) {
        const std::size_t step = stride * 2;

        #pragma omp parallel for
        for (std::size_t i = step - 1; i < paddedSize; i += step) {
            scanData[i] += scanData[i - stride];
        }
    }

    scanData[paddedSize - 1] = static_cast<T>(0);

    for (std::size_t stride = paddedSize / 2; stride >= 1; stride /= 2) {
        const std::size_t step = stride * 2;

        #pragma omp parallel for
        for (std::size_t i = step - 1; i < paddedSize; i += step) {
            T temporary = scanData[i - stride];
            scanData[i - stride] = scanData[i];
            scanData[i] += temporary;
        }

        if (stride == 1) {
            break;
        }
    }

    std::vector<T> output(originalSize);

    #pragma omp parallel for
    for (std::size_t i = 0; i < originalSize; i++) {
        output[i] = scanData[i] + input[i];
    }

    return output;
}

static void setOpenMPTiming(BenchmarkResult& result, double computeTimeMs, double verificationTimeMs) {
    result.computeTimeMs = computeTimeMs;
    result.transferTimeMs = 0.0;
    result.verificationTimeMs = verificationTimeMs;
    result.timeMs = computeTimeMs + verificationTimeMs;
}

template <typename T>
static BenchmarkResult runOpenMPTyped(const AppConfig& config) {
    BenchmarkResult result;
    result.backend = "openmp";
    result.operation = toString(config.operation);
    result.dataType = toString(config.dataType);
    result.size = config.size;
    result.threads = config.threads;
    result.processes = 1;
    result.blockSize = 0;

    if (config.size == 0) {
        throw std::runtime_error("Size must be greater than zero.");
    }

    int threads = config.threads > 0 ? config.threads : omp_get_max_threads();
    omp_set_num_threads(threads);

    std::vector<T> data = generateData<T>(config.size);

    if (config.operation == OperationType::Sum) {
        T parallelSum = static_cast<T>(0);

        auto computeStart = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for reduction(+ : parallelSum)
        for (std::size_t i = 0; i < data.size(); i++) {
            parallelSum += data[i];
        }

        auto computeEnd = std::chrono::high_resolution_clock::now();

        double verificationTimeMs = 0.0;

        if (config.verify) {
            auto verificationStart = std::chrono::high_resolution_clock::now();

            long double referenceSum = referenceSumLongDouble(data);
            result.absoluteError = absoluteDifferenceLongDouble(parallelSum, referenceSum);
            result.relativeError = relativeDifferenceLongDouble(parallelSum, referenceSum);

            auto verificationEnd = std::chrono::high_resolution_clock::now();
            verificationTimeMs = std::chrono::duration<double, std::milli>(verificationEnd - verificationStart).count();
        }

        double computeTimeMs = std::chrono::duration<double, std::milli>(computeEnd - computeStart).count();

        setOpenMPTiming(result, computeTimeMs, verificationTimeMs);
        result.throughputGBs =
            (static_cast<double>(config.size * sizeof(T)) / 1e9) /
            (result.timeMs / 1000.0);

        result.resultSummary = "sum = " + valueToString(parallelSum);

        return result;
    }

    if (config.operation == OperationType::Min) {
        T parallelMin = std::numeric_limits<T>::max();

        auto computeStart = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for reduction(min : parallelMin)
        for (std::size_t i = 0; i < data.size(); i++) {
            parallelMin = std::min(parallelMin, data[i]);
        }

        auto computeEnd = std::chrono::high_resolution_clock::now();

        double verificationTimeMs = 0.0;

        if (config.verify) {
            auto verificationStart = std::chrono::high_resolution_clock::now();

            T sequentialMin = *std::min_element(data.begin(), data.end());
            result.absoluteError = absoluteDifference(parallelMin, sequentialMin);
            result.relativeError = relativeDifference(parallelMin, sequentialMin);

            auto verificationEnd = std::chrono::high_resolution_clock::now();
            verificationTimeMs = std::chrono::duration<double, std::milli>(verificationEnd - verificationStart).count();
        }

        double computeTimeMs = std::chrono::duration<double, std::milli>(computeEnd - computeStart).count();

        setOpenMPTiming(result, computeTimeMs, verificationTimeMs);
        result.throughputGBs =
            (static_cast<double>(config.size * sizeof(T)) / 1e9) /
            (result.timeMs / 1000.0);

        result.resultSummary = "min = " + valueToString(parallelMin);

        return result;
    }

    if (config.operation == OperationType::Max) {
        T parallelMax = std::numeric_limits<T>::lowest();

        auto computeStart = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for reduction(max : parallelMax)
        for (std::size_t i = 0; i < data.size(); i++) {
            parallelMax = std::max(parallelMax, data[i]);
        }

        auto computeEnd = std::chrono::high_resolution_clock::now();

        double verificationTimeMs = 0.0;

        if (config.verify) {
            auto verificationStart = std::chrono::high_resolution_clock::now();

            T sequentialMax = *std::max_element(data.begin(), data.end());
            result.absoluteError = absoluteDifference(parallelMax, sequentialMax);
            result.relativeError = relativeDifference(parallelMax, sequentialMax);

            auto verificationEnd = std::chrono::high_resolution_clock::now();
            verificationTimeMs = std::chrono::duration<double, std::milli>(verificationEnd - verificationStart).count();
        }

        double computeTimeMs = std::chrono::duration<double, std::milli>(computeEnd - computeStart).count();

        setOpenMPTiming(result, computeTimeMs, verificationTimeMs);
        result.throughputGBs =
            (static_cast<double>(config.size * sizeof(T)) / 1e9) /
            (result.timeMs / 1000.0);

        result.resultSummary = "max = " + valueToString(parallelMax);

        return result;
    }

    if (config.operation == OperationType::Scan) {
        auto computeStart = std::chrono::high_resolution_clock::now();

        std::vector<T> output = blellochInclusiveScanOpenMP(data);

        auto computeEnd = std::chrono::high_resolution_clock::now();

        double verificationTimeMs = 0.0;

        if (config.verify) {
            auto verificationStart = std::chrono::high_resolution_clock::now();

            long double referenceLast = referenceSumLongDouble(data);
            result.absoluteError = absoluteDifferenceLongDouble(output.back(), referenceLast);
            result.relativeError = relativeDifferenceLongDouble(output.back(), referenceLast);

            auto verificationEnd = std::chrono::high_resolution_clock::now();
            verificationTimeMs = std::chrono::duration<double, std::milli>(verificationEnd - verificationStart).count();
        }

        double computeTimeMs = std::chrono::duration<double, std::milli>(computeEnd - computeStart).count();

        setOpenMPTiming(result, computeTimeMs, verificationTimeMs);
        result.throughputGBs =
            (static_cast<double>(2 * config.size * sizeof(T)) / 1e9) /
            (result.timeMs / 1000.0);

        result.resultSummary =
            "first = " + valueToString(output.front()) +
            ", last = " + valueToString(output.back());

        return result;
    }

    throw std::runtime_error("Unsupported OpenMP operation.");
}

BenchmarkResult runOpenMP(const AppConfig& config) {
    if (config.dataType == DataType::Int64) {
        return runOpenMPTyped<std::int64_t>(config);
    }

    if (config.dataType == DataType::Float) {
        return runOpenMPTyped<float>(config);
    }

    if (config.dataType == DataType::Double) {
        return runOpenMPTyped<double>(config);
    }

    throw std::runtime_error("Unsupported OpenMP data type.");
}