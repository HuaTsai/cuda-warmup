cmake_minimum_required(VERSION 3.24)

# Default to building SASS for the local GPU (native) instead of relying on PTX JIT:
# JIT fails with cudaErrorUnsupportedPtxVersion when the toolkit is newer than the
# driver, and SASS also avoids JIT overhead. Must be set before project();
# override with -DCMAKE_CUDA_ARCHITECTURES=... for cross-machine builds.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(cuda_warmup LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # for clangd

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA::cudart carries both the toolkit include dir and the runtime library.
# Link it PUBLIC so plain-C++ (.cc) consumers of the host-safe headers in
# include/ (which pull in <cuda_runtime.h>) resolve headers and cudaMalloc/etc.
# without relying on the compiler's default search paths.
find_package(CUDAToolkit REQUIRED)

add_library(vector_add src/vector_add.cu)
target_include_directories(vector_add PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(vector_add PUBLIC CUDA::cudart)

add_executable(vector_add_demo src/main.cc)
target_link_libraries(vector_add_demo PRIVATE vector_add)

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(test_vector_add tests/test_vector_add.cc)
target_link_libraries(test_vector_add PRIVATE vector_add GTest::gtest_main)
gtest_discover_tests(test_vector_add)
