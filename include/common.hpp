#pragma once

#include <cuda_runtime.h>

#include <format>
#include <source_location>
#include <stdexcept>

// Throws std::runtime_error on any non-success CUDA status. The default
// source_location argument captures the call site, so the message carries
// file:line without needing a macro. Use at call sites that consume the
// error-code API, e.g. cuda_check(cudaMalloc(&p, bytes)).
inline void cuda_check(
    cudaError_t err,
    std::source_location loc = std::source_location::current()) {
  if (err != cudaSuccess) [[unlikely]] {
    throw std::runtime_error(std::format(
        "CUDA error at {}:{} in {}: {} ({})", loc.file_name(), loc.line(),
        loc.function_name(), cudaGetErrorName(err), cudaGetErrorString(err)));
  }
}
