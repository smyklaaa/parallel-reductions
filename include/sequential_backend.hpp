#pragma once

#include "app_config.hpp"
#include "benchmark_result.hpp"

BenchmarkResult runSequential(const AppConfig& config);
void printBenchmarkResult(const BenchmarkResult& result);