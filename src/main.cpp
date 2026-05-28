#include "argument_parser.hpp"
#include "csv_writer.hpp"
#include "cuda_backend.hpp"
#include "sequential_backend.hpp"
#include "openmp_backend.hpp"
#include "mpi_backend.hpp"

#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        AppConfig config = parseArguments(argc, argv);

        printConfig(config);

        if (config.backend == BackendType::Sequential) {
            BenchmarkResult result = runSequential(config);
            printBenchmarkResult(result);
            appendResultToCsv(result, config.outputFile);
            std::cout << "Result saved to CSV: " << config.outputFile << "\n";
            return 0;
}

        if (config.backend == BackendType::CUDA) {
            runCUDA(config);
            return 0;
        }

        if (config.backend == BackendType::OpenMP) {
            BenchmarkResult result = runOpenMP(config);
            printBenchmarkResult(result);
            appendResultToCsv(result, config.outputFile);
            std::cout << "Result saved to CSV: " << config.outputFile << "\n";
            return 0;
        }

        if (config.backend == BackendType::MPI) {
            BenchmarkResult result = runMPI(config);

            if (!result.backend.empty()) {
                printBenchmarkResult(result);
                appendResultToCsv(result, config.outputFile);
                std::cout << "Result saved to CSV: " << config.outputFile << "\n";
            }

            return 0;
        }

        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << "\n";
        std::cerr << "Use --help to show available options.\n";
        return 1;
    }
}