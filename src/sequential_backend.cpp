#include "sequential_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
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
static BenchmarkResult runSequentialTyped(const AppConfig& config) {
    BenchmarkResult result;
    result.backend = "sequential";
    result.operation = toString(config.operation);
    result.dataType = toString(config.dataType);
    result.size = config.size;

    std::vector<T> data = generateData<T>(config.size);

    auto start = std::chrono::high_resolution_clock::now();

    if (config.operation == OperationType::Sum) {
        T sum = std::accumulate(data.begin(), data.end(), static_cast<T>(0));

        auto end = std::chrono::high_resolution_clock::now();
        result.timeMs = std::chrono::duration<double, std::milli>(end - start).count();

        result.resultSummary = "sum = " + valueToString(sum);
        result.throughputGBs = (static_cast<double>(config.size * sizeof(T)) / 1e9) / (result.timeMs / 1000.0);

        return result;
    }

    if (config.operation == OperationType::Min) {
        T minValue = *std::min_element(data.begin(), data.end());

        auto end = std::chrono::high_resolution_clock::now();
        result.timeMs = std::chrono::duration<double, std::milli>(end - start).count();

        result.resultSummary = "min = " + valueToString(minValue);
        result.throughputGBs = (static_cast<double>(config.size * sizeof(T)) / 1e9) / (result.timeMs / 1000.0);

        return result;
    }

    if (config.operation == OperationType::Max) {
        T maxValue = *std::max_element(data.begin(), data.end());

        auto end = std::chrono::high_resolution_clock::now();
        result.timeMs = std::chrono::duration<double, std::milli>(end - start).count();

        result.resultSummary = "max = " + valueToString(maxValue);
        result.throughputGBs = (static_cast<double>(config.size * sizeof(T)) / 1e9) / (result.timeMs / 1000.0);

        return result;
    }

    if (config.operation == OperationType::Scan) {
        std::vector<T> output(config.size);
        std::partial_sum(data.begin(), data.end(), output.begin());

        auto end = std::chrono::high_resolution_clock::now();
        result.timeMs = std::chrono::duration<double, std::milli>(end - start).count();

        result.resultSummary =
            "first = " + valueToString(output.front()) +
            ", last = " + valueToString(output.back());

        result.throughputGBs = (static_cast<double>(2 * config.size * sizeof(T)) / 1e9) / (result.timeMs / 1000.0);

        return result;
    }

    throw std::runtime_error("Unsupported sequential operation.");
}

BenchmarkResult runSequential(const AppConfig& config) {
    if (config.dataType == DataType::Int64) {
        return runSequentialTyped<std::int64_t>(config);
    }

    if (config.dataType == DataType::Float) {
        return runSequentialTyped<float>(config);
    }

    if (config.dataType == DataType::Double) {
        return runSequentialTyped<double>(config);
    }

    throw std::runtime_error("Unsupported sequential data type.");
}

void printBenchmarkResult(const BenchmarkResult& result) {
    std::cout
        << "Result:\n"
        << "  Backend: " << result.backend << "\n"
        << "  Operation: " << result.operation << "\n"
        << "  Data type: " << result.dataType << "\n"
        << "  Size: " << result.size << "\n"
        << "  Time: " << std::fixed << std::setprecision(4) << result.timeMs << " ms\n"
        << "  Throughput: " << std::fixed << std::setprecision(4) << result.throughputGBs << " GB/s\n"
        << "  " << result.resultSummary << "\n";
}