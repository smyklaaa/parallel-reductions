#pragma once

#include "backend_type.hpp"
#include "operation_type.hpp"
#include "data_type.hpp"

#include <cstddef>
#include <string>

struct AppConfig {
    BackendType backend = BackendType::Sequential;
    OperationType operation = OperationType::Sum;
    DataType dataType = DataType::Double;
    std::size_t size = 1000000;
    int threads = 1;
    int processes = 1;
    int blockSize = 256;
    bool verify = true;
    std::string outputFile = "results/results.csv";
};