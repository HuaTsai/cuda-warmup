#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#include "vector_add.cuh"

// ASSERT_* 只能用在回傳 void 的函式裡，所以包成 macro。
#define CUDA_CHECK(expr)                                      \
  do {                                                        \
    const cudaError_t err_ = (expr);                          \
    ASSERT_EQ(err_, cudaSuccess) << cudaGetErrorString(err_); \
  } while (0)

class VectorAddTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "no usable CUDA device";
    }
  }

  // host 準備資料 → 拷上 device → 跑 kernel → 拷回驗證的完整流程。
  void RunAndVerify(int n) {
    std::vector<float> ha(n), hb(n), hc(n, -1.0f);
    for (int i = 0; i < n; ++i) {
      ha[i] = static_cast<float>(i);
      hb[i] = static_cast<float>(2 * i);
    }

    float *da = nullptr, *db = nullptr, *dc = nullptr;
    const size_t bytes = n * sizeof(float);
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&dc, bytes));
    CUDA_CHECK(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(vector_add(da, db, dc, n));  // launch 設定錯誤在這裡回報
    CUDA_CHECK(cudaDeviceSynchronize());    // kernel 執行期錯誤在這裡回報

    CUDA_CHECK(cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
    CUDA_CHECK(cudaFree(dc));

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
  EXPECT_EQ(vector_add(nullptr, nullptr, nullptr, 0), cudaSuccess);
}
