cmake_minimum_required(VERSION 3.24)

# 預設編譯本機 GPU 的 SASS（native），避免只嵌 PTX：
# driver（CUDA 13.0）比 toolkit（13.3）舊時，PTX JIT 會失敗（cudaErrorUnsupportedPtxVersion）。
# 必須在 project() 之前設定；跨機器建置可用 -DCMAKE_CUDA_ARCHITECTURES=... 覆蓋。
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(cuda_warmup LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(vector_add src/vector_add.cu)
target_include_directories(vector_add PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(test_vector_add tests/test_vector_add.cu)
target_link_libraries(test_vector_add PRIVATE vector_add GTest::gtest_main)
gtest_discover_tests(test_vector_add)
