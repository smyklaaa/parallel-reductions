#include "argument_parser.hpp"
#include "sequential_backend.hpp"

#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        AppConfig config = parseArguments(argc, argv);

        printConfig(config);

        if (config.backend == BackendType::Sequential) {
            BenchmarkResult result = runSequential(config);
            printBenchmarkResult(result);
            return 0;
        }

        if (config.backend == BackendType::CUDA) {
            std::cout << "CUDA backend selected.\n";
            std::cout << "CUDA availability check will be added in the next step.\n";
            return 0;
        }

        if (config.backend == BackendType::OpenMP) {
            std::cout << "OpenMP backend is reserved for later implementation.\n";
            return 0;
        }

        if (config.backend == BackendType::MPI) {
            std::cout << "MPI backend is reserved for later implementation.\n";
            return 0;
        }

        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Error: " << exception.what() << "\n";
        std::cerr << "Use --help to show available options.\n";
        return 1;
    }
}