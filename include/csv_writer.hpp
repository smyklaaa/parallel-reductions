#pragma once

#include "benchmark_result.hpp"

#include <string>

void appendResultToCsv(const BenchmarkResult& result, const std::string& outputFile);