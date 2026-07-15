#pragma once

#include <cuda_runtime.h>

#include <cstddef>

#include "check.hpp"

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t n) : n_(n) { CudaCheck(cudaMalloc(&d_data_, n * sizeof(T))); }
  ~DeviceBuffer() { cudaFree(d_data_); }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;
  DeviceBuffer(DeviceBuffer &&other) noexcept : d_data_(other.d_data_), n_(other.n_) {
    other.d_data_ = nullptr;
    other.n_ = 0;
  }
  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
      cudaFree(d_data_);
      d_data_ = other.d_data_;
      n_ = other.n_;
      other.d_data_ = nullptr;
      other.n_ = 0;
    }
    return *this;
  }

  T *get() const { return d_data_; }
  void from_host(const T *h_data) {
    CudaCheck(cudaMemcpy(d_data_, h_data, n_ * sizeof(T), cudaMemcpyHostToDevice));
  }
  void to_host(T *h_data) const {
    CudaCheck(cudaMemcpy(h_data, d_data_, n_ * sizeof(T), cudaMemcpyDeviceToHost));
  }

 private:
  T *d_data_;
  std::size_t n_;
};
