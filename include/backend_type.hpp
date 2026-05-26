#pragma once

#include <string>

enum class BackendType {
    Sequential,
    OpenMP,
    MPI,
    CUDA
};

BackendType parseBackendType(const std::string& value);
std::string toString(BackendType backend);