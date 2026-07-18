#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "check.hpp"
#include "device_buffer.hpp"
#include "reduce.hpp"

TEST(ReduceSumV0, SimpleSum) {
  constexpr int n = 1024;
  std::vector<float> h_in(n);
  std::iota(h_in.begin(), h_in.end(), 1.0f);
  DeviceBuffer<float> d_in(n), d_out(1);
  d_in.from_host(h_in.data());
  CudaCheck(ReduceSumV0(d_in.get(), d_out.get(), n));
  float res = 0;
  d_out.to_host(&res);
  ASSERT_EQ(res, (1 + n) * n / 2);
}

TEST(ReduceSumV0, NotPowersOfTwo) {
  constexpr int n = 1000;
  std::vector<float> h_in(n);
  std::iota(h_in.begin(), h_in.end(), 1.0f);
  DeviceBuffer<float> d_in(n), d_out(1);
  d_in.from_host(h_in.data());
  CudaCheck(ReduceSumV0(d_in.get(), d_out.get(), n));
  float res = 0;
  d_out.to_host(&res);
  ASSERT_EQ(res, (1 + n) * n / 2);
}

TEST(ReduceSumV0, ThrowWithInvalidN) {
  DeviceBuffer<float> d_in(1), d_out(1);
  EXPECT_EQ(ReduceSumV0(d_in.get(), d_out.get(), 0), cudaErrorInvalidValue);
}
