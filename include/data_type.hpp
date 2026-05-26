#pragma once

#include <string>

enum class DataType {
    Int64,
    Float,
    Double
};

DataType parseDataType(const std::string& value);
std::string toString(DataType dataType);