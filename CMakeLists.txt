cmake_minimum_required(VERSION 3.24)
project(cuda_warmup LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 未指定時 CMake 會採用 nvcc 的預設架構（含 PTX，可 JIT 到較新的 GPU）。
# 想針對本機 GPU 最佳化可用：cmake -DCMAKE_CUDA_ARCHITECTURES=native ...

add_library(vector_add src/vector_add.cu)
target_include_directories(vector_add PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(test_vector_add tests/test_vector_add.cu)
target_link_libraries(test_vector_add PRIVATE vector_add GTest::gtest_main)
gtest_discover_tests(test_vector_add)
