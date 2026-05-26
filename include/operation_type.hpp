#pragma once

#include <string>

enum class OperationType {
    Sum,
    Min,
    Max,
    Scan
};

OperationType parseOperationType(const std::string& value);
std::string toString(OperationType operation);