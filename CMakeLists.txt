cmake_minimum_required(VERSION 3.24)

# Default to building SASS for the local GPU (native) instead of relying on PTX JIT:
# JIT fails with cudaErrorUnsupportedPtxVersion when the toolkit is newer than the
# driver, and SASS also avoids JIT overhead. Must be set before project();
# override with -DCMAKE_CUDA_ARCHITECTURES=... for cross-machine builds.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(cuda_warmup LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(${PROJECT_NAME} src/vector_add.cu src/reduce.cu)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart)

add_executable(main src/main.cc)
target_link_libraries(main PRIVATE ${PROJECT_NAME})

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(test_vector_add tests/test_vector_add.cc)
target_link_libraries(test_vector_add PRIVATE ${PROJECT_NAME} GTest::gtest_main)
gtest_discover_tests(test_vector_add)

add_executable(test_reduce tests/test_reduce.cc)
target_link_libraries(test_reduce PRIVATE ${PROJECT_NAME} GTest::gtest_main)
gtest_discover_tests(test_reduce)

