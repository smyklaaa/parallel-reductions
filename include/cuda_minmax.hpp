#pragma once

#include "app_config.hpp"
#include "benchmark_result.hpp"

BenchmarkResult runCudaMin(const AppConfig& config);
BenchmarkResult runCudaMax(const AppConfig& config);