cmake_minimum_required(VERSION 3.24)

# Default to building SASS for the local GPU (native) instead of relying on PTX JIT:
# JIT fails with cudaErrorUnsupportedPtxVersion when the toolkit is newer than the
# driver, and SASS also avoids JIT overhead. Must be set before project();
# override with -DCMAKE_CUDA_ARCHITECTURES=... for cross-machine builds.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(cuda_warmup LANGUAGES CXX CUDA)

# Default to RelWithDebInfo so device code is optimized yet still carries -lineinfo
# (added below): compute-sanitizer and Nsight Compute can then map errors/stalls back
# to source lines. Pick another mode with -DCMAKE_BUILD_TYPE=Debug|Release or the
# pixi configure-* tasks. Debug adds -G (cuda-gdb, disables device optimization);
# Release drops line info for clean benchmark numbers.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(${PROJECT_NAME} src/vector_add.cu src/reduce.cu)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(${PROJECT_NAME} PUBLIC CUDA::cudart)

# Per-config device build flags for debugging/profiling the kernels:
#   RelWithDebInfo -> -lineinfo : source-line mapping in compute-sanitizer & Nsight,
#                                 while keeping optimizations on (real perf numbers).
#   Debug          -> -G        : full device debug for cuda-gdb (disables device opt).
# The two are mutually exclusive (nvcc rejects -G with -lineinfo), which the CONFIG
# generator expressions guarantee by never emitting both in one build type.
target_compile_options(${PROJECT_NAME} PRIVATE
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:RelWithDebInfo>>:--generate-line-info>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:--device-debug>)

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

