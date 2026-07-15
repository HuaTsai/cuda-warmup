#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#include "check.hpp"
#include "device_buffer.hpp"
#include "vector_add.hpp"

class VectorAddTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "no usable CUDA device";
    }
  }

  // Full round trip: prepare data on host, copy to device, run the kernel,
  // copy back and verify.
  void RunAndVerify(int n) {
    std::vector<float> ha(n), hb(n), hc(n, -1.0f);
    for (int i = 0; i < n; ++i) {
      ha[i] = static_cast<float>(i);
      hb[i] = static_cast<float>(2 * i);
    }

    DeviceBuffer<float> da(n), db(n), dc(n);
    da.from_host(ha.data());
    db.from_host(hb.data());
    CudaCheck(VectorAdd(da.get(), db.get(), dc.get(), n));
    CudaCheck(cudaDeviceSynchronize());
    dc.to_host(hc.data());

    for (int i = 0; i < n; ++i) {
      ASSERT_EQ(hc[i], ha[i] + hb[i]) << "mismatch at index " << i;
    }
  }
};

TEST_F(VectorAddTest, SingleElement) { RunAndVerify(1); }

TEST_F(VectorAddTest, ExactlyOneBlock) { RunAndVerify(256); }

TEST_F(VectorAddTest, NotMultipleOfBlockSize) { RunAndVerify(1000); }

TEST_F(VectorAddTest, MultipleBlocks) { RunAndVerify(1 << 20); }

TEST_F(VectorAddTest, ZeroLengthIsNoop) {
  EXPECT_EQ(VectorAdd(nullptr, nullptr, nullptr, 0), cudaSuccess);
}
