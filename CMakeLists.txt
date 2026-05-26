cmake_minimum_required(VERSION 3.20)

project(parallel_reductions LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(parallel_reductions
    src/main.cpp
    src/argument_parser.cpp
    src/sequential_backend.cpp
)

target_include_directories(parallel_reductions PRIVATE include)