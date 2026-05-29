#include "csv_writer.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

static bool fileExistsAndNotEmpty(const std::string& path) {
    return std::filesystem::exists(path) && std::filesystem::file_size(path) > 0;
}

void appendResultToCsv(const BenchmarkResult& result, const std::string& outputFile) {
    std::filesystem::path path(outputFile);

    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }

    bool writeHeader = !fileExistsAndNotEmpty(outputFile);

    std::ofstream file(outputFile, std::ios::app);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CSV output file: " + outputFile);
    }

    if (writeHeader) {
        file << "backend,operation,data_type,size,threads,processes,block_size,time_ms,throughput_gbs,absolute_error,relative_error,result_summary\n";
    }

    file << result.backend << ","
            << result.operation << ","
            << result.dataType << ","
            << result.size << ","
            << result.threads << ","
            << result.processes << ","
            << result.blockSize << ","
            << result.timeMs << ","
            << result.throughputGBs << ","
            << result.absoluteError << ","
            << result.relativeError << ","
            << "\"" << result.resultSummary << "\""
            << "\n";
}