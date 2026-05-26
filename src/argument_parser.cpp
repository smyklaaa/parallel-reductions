#include "argument_parser.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

BackendType parseBackendType(const std::string& value) {
    if (value == "sequential") return BackendType::Sequential;
    if (value == "openmp") return BackendType::OpenMP;
    if (value == "mpi") return BackendType::MPI;
    if (value == "cuda") return BackendType::CUDA;

    throw std::runtime_error("Unsupported backend: " + value);
}

std::string toString(BackendType backend) {
    if (backend == BackendType::Sequential) return "sequential";
    if (backend == BackendType::OpenMP) return "openmp";
    if (backend == BackendType::MPI) return "mpi";
    if (backend == BackendType::CUDA) return "cuda";

    return "unknown";
}

OperationType parseOperationType(const std::string& value) {
    if (value == "sum") return OperationType::Sum;
    if (value == "min") return OperationType::Min;
    if (value == "max") return OperationType::Max;
    if (value == "scan") return OperationType::Scan;

    throw std::runtime_error("Unsupported operation: " + value);
}

std::string toString(OperationType operation) {
    if (operation == OperationType::Sum) return "sum";
    if (operation == OperationType::Min) return "min";
    if (operation == OperationType::Max) return "max";
    if (operation == OperationType::Scan) return "scan";

    return "unknown";
}

DataType parseDataType(const std::string& value) {
    if (value == "int64") return DataType::Int64;
    if (value == "float") return DataType::Float;
    if (value == "double") return DataType::Double;

    throw std::runtime_error("Unsupported data type: " + value);
}

std::string toString(DataType dataType) {
    if (dataType == DataType::Int64) return "int64";
    if (dataType == DataType::Float) return "float";
    if (dataType == DataType::Double) return "double";

    return "unknown";
}

static bool parseBool(const std::string& value) {
    if (value == "true") return true;
    if (value == "false") return false;

    throw std::runtime_error("Boolean value must be true or false");
}

AppConfig parseArguments(int argc, char** argv) {
    AppConfig config;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        auto requireValue = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }

            return argv[++i];
        };

        if (arg == "--backend") {
            config.backend = parseBackendType(requireValue(arg));
        } else if (arg == "--operation") {
            config.operation = parseOperationType(requireValue(arg));
        } else if (arg == "--type") {
            config.dataType = parseDataType(requireValue(arg));
        } else if (arg == "--size") {
            config.size = std::stoull(requireValue(arg));
        } else if (arg == "--threads") {
            config.threads = std::stoi(requireValue(arg));
        } else if (arg == "--processes") {
            config.processes = std::stoi(requireValue(arg));
        } else if (arg == "--block-size") {
            config.blockSize = std::stoi(requireValue(arg));
        } else if (arg == "--verify") {
            config.verify = parseBool(requireValue(arg));
        } else if (arg == "--output") {
            config.outputFile = requireValue(arg);
        } else if (arg == "--help") {
            printHelp();
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (config.size == 0) {
        throw std::runtime_error("Size must be greater than zero");
    }

    if (config.threads <= 0) {
        throw std::runtime_error("Threads value must be greater than zero");
    }

    if (config.processes <= 0) {
        throw std::runtime_error("Processes value must be greater than zero");
    }

    if (config.blockSize <= 0) {
        throw std::runtime_error("Block size must be greater than zero");
    }

    return config;
}

void printHelp() {
    std::cout
        << "Parallel reductions and scans\n\n"
        << "Usage:\n"
        << "  ./parallel_reductions [options]\n\n"
        << "Options:\n"
        << "  --backend <sequential|openmp|mpi|cuda>\n"
        << "  --operation <sum|min|max|scan>\n"
        << "  --type <int64|float|double>\n"
        << "  --size <number>\n"
        << "  --threads <number>\n"
        << "  --processes <number>\n"
        << "  --block-size <number>\n"
        << "  --verify <true|false>\n"
        << "  --output <file>\n"
        << "  --help\n\n"
        << "Examples:\n"
        << "  ./parallel_reductions --backend sequential --operation sum --type double --size 1000000\n"
        << "  ./parallel_reductions --backend cuda --operation scan --type float --size 1000000 --block-size 256\n";
}

void printConfig(const AppConfig& config) {
    std::cout
        << "Configuration:\n"
        << "  Backend: " << toString(config.backend) << "\n"
        << "  Operation: " << toString(config.operation) << "\n"
        << "  Data type: " << toString(config.dataType) << "\n"
        << "  Size: " << config.size << "\n"
        << "  Threads: " << config.threads << "\n"
        << "  Processes: " << config.processes << "\n"
        << "  Block size: " << config.blockSize << "\n"
        << "  Verify: " << (config.verify ? "true" : "false") << "\n"
        << "  Output file: " << config.outputFile << "\n";
}