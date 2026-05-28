#include "cuda_backend.hpp"
#include "csv_writer.hpp"
#include "cuda_minmax.hpp"
#include "cuda_scan.hpp"
#include "cuda_sum.hpp"
#include "cuda_vector_add.hpp"
#include "sequential_backend.hpp"

#include <array>
#include <cstdio>
#include <iostream>
#include <string>

#ifdef _WIN32
#define POPEN _popen
#define PCLOSE _pclose
#else
#define POPEN popen
#define PCLOSE pclose
#endif

static std::string executeCommand(const std::string& command) {
    std::array<char, 256> buffer;
    std::string result;

    FILE* pipe = POPEN(command.c_str(), "r");

    if (!pipe) {
        return "";
    }

    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        result += buffer.data();
    }

    PCLOSE(pipe);

    return result;
}

static bool isNvidiaGpuAvailable(std::string& gpuInfo) {
    gpuInfo = executeCommand("nvidia-smi -L");

    if (gpuInfo.empty()) {
        return false;
    }

    if (gpuInfo.find("GPU") == std::string::npos &&
        gpuInfo.find("NVIDIA") == std::string::npos) {
        return false;
    }

    return true;
}

void runCUDA(const AppConfig& config) {
    std::cout << "CUDA backend selected.\n";

    std::string gpuInfo;

    if (!isNvidiaGpuAvailable(gpuInfo)) {
        std::cout << "No NVIDIA GPU detected or NVIDIA drivers are not available.\n";
        std::cout << "CUDA backend cannot be executed on this machine.\n";
        return;
    }

    std::cout << "NVIDIA GPU detected:\n";
    std::cout << gpuInfo;

    if (config.operation == OperationType::Sum) {
        BenchmarkResult result = runCudaSum(config);
        printBenchmarkResult(result);
        appendResultToCsv(result, config.outputFile);
        std::cout << "Result saved to CSV: " << config.outputFile << "\n";
        return;
    }

    if (config.operation == OperationType::Min) {
        BenchmarkResult result = runCudaMin(config);
        printBenchmarkResult(result);
        appendResultToCsv(result, config.outputFile);
        std::cout << "Result saved to CSV: " << config.outputFile << "\n";
        return;
    }

    if (config.operation == OperationType::Max) {
        BenchmarkResult result = runCudaMax(config);
        printBenchmarkResult(result);
        appendResultToCsv(result, config.outputFile);
        std::cout << "Result saved to CSV: " << config.outputFile << "\n";
        return;
    }

    if (config.operation == OperationType::Scan) {
        BenchmarkResult result = runCudaScan(config);
        printBenchmarkResult(result);
        appendResultToCsv(result, config.outputFile);
        std::cout << "Result saved to CSV: " << config.outputFile << "\n";
        return;
    }

    std::cout << "CUDA operation is not implemented yet.\n";
}