#pragma once

#include <cstddef>
#include <string>

struct BenchmarkResult {
    std::string backend;
    std::string operation;
    std::string dataType;
    std::size_t size = 0;
    double timeMs = 0.0;
    double throughputGBs = 0.0;
    double absoluteError = 0.0;
    double relativeError = 0.0;
    std::string resultSummary;
};