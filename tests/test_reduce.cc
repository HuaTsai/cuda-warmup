#include <gtest/gtest.h>

#include <vector>

#include "check.hpp"
#include "device_buffer.hpp"
#include "reduce.hpp"

void ReduceTestHelper(cudaError_t (*func)(const float *, float *, int)) {
  constexpr int n = 10'000'000;
  std::vector<float> h_in(n, 1.f);
  DeviceBuffer<float> d_in(n), d_out(1);
  d_in.from_host(h_in.data());
  CudaCheck(func(d_in.get(), d_out.get(), n));
  float res = 0;
  d_out.to_host(&res);
  ASSERT_EQ(res, n);
}

TEST(ReduceSumV0, SimpleSum) { ReduceTestHelper(ReduceSumV0); }

TEST(ReduceSumV1, SimpleSum) { ReduceTestHelper(ReduceSumV1); }

TEST(ReduceSumV2, SimpleSum) { ReduceTestHelper(ReduceSumV2); }

TEST(ReduceSumV3, SimpleSum) { ReduceTestHelper(ReduceSumV3); }

TEST(ReduceSumV4, SimpleSum) { ReduceTestHelper(ReduceSumV4); }

TEST(ReduceSumV5, SimpleSum) { ReduceTestHelper(ReduceSumV5); }

TEST(ReduceSumV0, ThrowWithInvalidN) {
  DeviceBuffer<float> d_in(1), d_out(1);
  EXPECT_EQ(ReduceSumV0(d_in.get(), d_out.get(), 0), cudaErrorInvalidValue);
}
