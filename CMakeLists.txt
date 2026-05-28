cmake_minimum_required(VERSION 3.20)

project(parallel_reductions LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(parallel_reductions
    src/main.cpp
    src/argument_parser.cpp
    src/sequential_backend.cpp
    src/cuda_backend.cpp
    src/csv_writer.cpp
    src/openmp_backend.cpp
    src/mpi_backend.cpp
    cuda/cuda_vector_add.cu
    cuda/cuda_sum.cu
    cuda/cuda_minmax.cu
)

target_include_directories(parallel_reductions PRIVATE include)

find_package(OpenMP REQUIRED)
target_link_libraries(parallel_reductions PRIVATE OpenMP::OpenMP_CXX)
target_compile_definitions(parallel_reductions PRIVATE USE_OPENMP)

find_package(MPI REQUIRED)
target_link_libraries(parallel_reductions PRIVATE MPI::MPI_CXX)
target_compile_definitions(parallel_reductions PRIVATE USE_MPI)