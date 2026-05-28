#include "mpi_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <mpi.h>
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
static long double referenceSumLongDouble(const std::vector<T>& data) {
    long double sum = 0.0L;

    for (T value : data) {
        sum += static_cast<long double>(value);
    }

    return sum;
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
static MPI_Datatype getMPIType();

template <>
MPI_Datatype getMPIType<std::int64_t>() {
    return MPI_INT64_T;
}

template <>
MPI_Datatype getMPIType<float>() {
    return MPI_FLOAT;
}

template <>
MPI_Datatype getMPIType<double>() {
    return MPI_DOUBLE;
}

static std::pair<std::size_t, std::size_t> getLocalRange(
    std::size_t globalSize,
    int rank,
    int worldSize
) {
    std::size_t base = globalSize / static_cast<std::size_t>(worldSize);
    std::size_t remainder = globalSize % static_cast<std::size_t>(worldSize);

    std::size_t begin =
        static_cast<std::size_t>(rank) * base +
        std::min<std::size_t>(static_cast<std::size_t>(rank), remainder);

    std::size_t count = base;

    if (static_cast<std::size_t>(rank) < remainder) {
        count++;
    }

    return {begin, begin + count};
}

template <typename T>
static BenchmarkResult runMPITyped(const AppConfig& config) {
    MPI_Init(nullptr, nullptr);

    int rank = 0;
    int worldSize = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    BenchmarkResult result;

    result.backend = "mpi";
    result.operation = toString(config.operation);
    result.dataType = toString(config.dataType);
    result.size = config.size;
    result.threads = 1;
    result.processes = worldSize;
    result.blockSize = 0;

    if (config.size == 0) {
        MPI_Finalize();
        throw std::runtime_error("Size must be greater than zero.");
    }

    MPI_Datatype mpiType = getMPIType<T>();

    std::vector<T> fullData = generateData<T>(config.size);

    auto [begin, end] = getLocalRange(config.size, rank, worldSize);
    std::vector<T> localData(fullData.begin() + begin, fullData.begin() + end);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (config.operation == OperationType::Sum) {
        T localValue = std::accumulate(localData.begin(), localData.end(), static_cast<T>(0));
        T globalValue = static_cast<T>(0);

        MPI_Allreduce(&localValue, &globalValue, 1, mpiType, MPI_SUM, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        auto finish = std::chrono::high_resolution_clock::now();

        double localTimeMs = std::chrono::duration<double, std::milli>(finish - start).count();
        double globalTimeMs = 0.0;

        MPI_Reduce(&localTimeMs, &globalTimeMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            result.timeMs = globalTimeMs;
            result.throughputGBs =
                (static_cast<double>(config.size * sizeof(T)) / 1e9) /
                (result.timeMs / 1000.0);

            if (config.verify) {
                long double reference = referenceSumLongDouble(fullData);
                result.absoluteError = absoluteDifferenceLongDouble(globalValue, reference);
                result.relativeError = relativeDifferenceLongDouble(globalValue, reference);
            }

            result.resultSummary = "sum = " + valueToString(globalValue);
        }

        MPI_Finalize();

        if (rank != 0) {
            return BenchmarkResult{};
        }

        return result;
    }

    if (config.operation == OperationType::Min) {
        T localValue = localData.empty()
            ? std::numeric_limits<T>::max()
            : *std::min_element(localData.begin(), localData.end());

        T globalValue = std::numeric_limits<T>::max();

        MPI_Allreduce(&localValue, &globalValue, 1, mpiType, MPI_MIN, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        auto finish = std::chrono::high_resolution_clock::now();

        double localTimeMs = std::chrono::duration<double, std::milli>(finish - start).count();
        double globalTimeMs = 0.0;

        MPI_Reduce(&localTimeMs, &globalTimeMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            result.timeMs = globalTimeMs;
            result.throughputGBs =
                (static_cast<double>(config.size * sizeof(T)) / 1e9) /
                (result.timeMs / 1000.0);

            if (config.verify) {
                T reference = *std::min_element(fullData.begin(), fullData.end());
                result.absoluteError = absoluteDifference(globalValue, reference);
                result.relativeError = relativeDifference(globalValue, reference);
            }

            result.resultSummary = "min = " + valueToString(globalValue);
        }

        MPI_Finalize();

        if (rank != 0) {
            return BenchmarkResult{};
        }

        return result;
    }

    if (config.operation == OperationType::Max) {
        T localValue = localData.empty()
            ? std::numeric_limits<T>::lowest()
            : *std::max_element(localData.begin(), localData.end());

        T globalValue = std::numeric_limits<T>::lowest();

        MPI_Allreduce(&localValue, &globalValue, 1, mpiType, MPI_MAX, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        auto finish = std::chrono::high_resolution_clock::now();

        double localTimeMs = std::chrono::duration<double, std::milli>(finish - start).count();
        double globalTimeMs = 0.0;

        MPI_Reduce(&localTimeMs, &globalTimeMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            result.timeMs = globalTimeMs;
            result.throughputGBs =
                (static_cast<double>(config.size * sizeof(T)) / 1e9) /
                (result.timeMs / 1000.0);

            if (config.verify) {
                T reference = *std::max_element(fullData.begin(), fullData.end());
                result.absoluteError = absoluteDifference(globalValue, reference);
                result.relativeError = relativeDifference(globalValue, reference);
            }

            result.resultSummary = "max = " + valueToString(globalValue);
        }

        MPI_Finalize();

        if (rank != 0) {
            return BenchmarkResult{};
        }

        return result;
    }

    if (config.operation == OperationType::Scan) {
        std::vector<T> localOutput(localData.size());

        if (!localData.empty()) {
            std::partial_sum(localData.begin(), localData.end(), localOutput.begin());
        }

        T localBlockSum = localOutput.empty() ? static_cast<T>(0) : localOutput.back();

        std::vector<T> blockSums(worldSize, static_cast<T>(0));

        MPI_Allgather(
            &localBlockSum,
            1,
            mpiType,
            blockSums.data(),
            1,
            mpiType,
            MPI_COMM_WORLD
        );

        T offset = static_cast<T>(0);

        for (int i = 0; i < rank; i++) {
            offset += blockSums[i];
        }

        for (T& value : localOutput) {
            value += offset;
        }

        T globalLast = static_cast<T>(0);

        MPI_Allreduce(
            &localBlockSum,
            &globalLast,
            1,
            mpiType,
            MPI_SUM,
            MPI_COMM_WORLD
        );

        MPI_Barrier(MPI_COMM_WORLD);
        auto finish = std::chrono::high_resolution_clock::now();

        double localTimeMs = std::chrono::duration<double, std::milli>(finish - start).count();
        double globalTimeMs = 0.0;

        MPI_Reduce(&localTimeMs, &globalTimeMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            result.timeMs = globalTimeMs;
            result.throughputGBs =
                (static_cast<double>(2 * config.size * sizeof(T)) / 1e9) /
                (result.timeMs / 1000.0);

            if (config.verify) {
                long double reference = referenceSumLongDouble(fullData);
                result.absoluteError = absoluteDifferenceLongDouble(globalLast, reference);
                result.relativeError = relativeDifferenceLongDouble(globalLast, reference);
            }

            result.resultSummary =
                "first = " + valueToString(fullData.front()) +
                ", last = " + valueToString(globalLast);
        }

        MPI_Finalize();

        if (rank != 0) {
            return BenchmarkResult{};
        }

        return result;
    }

    MPI_Finalize();
    throw std::runtime_error("Unsupported MPI operation.");
}

BenchmarkResult runMPI(const AppConfig& config) {
    if (config.dataType == DataType::Int64) {
        return runMPITyped<std::int64_t>(config);
    }

    if (config.dataType == DataType::Float) {
        return runMPITyped<float>(config);
    }

    if (config.dataType == DataType::Double) {
        return runMPITyped<double>(config);
    }

    throw std::runtime_error("Unsupported MPI data type.");
}