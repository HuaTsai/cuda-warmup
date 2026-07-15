#include <cstdio>
#include <vector>

#include "common.hpp"
#include "vector_add.cuh"

int main() {
  constexpr int n = 1 << 20;
  const size_t bytes = n * sizeof(float);

  std::vector<float> ha(n), hb(n), hc(n);
  for (int i = 0; i < n; ++i) {
    ha[i] = static_cast<float>(i);
    hb[i] = static_cast<float>(2 * i);
  }

  try {
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    cuda_check(cudaMalloc(&da, bytes));
    cuda_check(cudaMalloc(&db, bytes));
    cuda_check(cudaMalloc(&dc, bytes));

    cuda_check(cudaMemcpy(da, ha.data(), bytes, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(db, hb.data(), bytes, cudaMemcpyHostToDevice));

    cuda_check(vector_add(da, db, dc, n));  // launch configuration errors
    cuda_check(cudaDeviceSynchronize());    // kernel runtime errors

    cuda_check(cudaMemcpy(hc.data(), dc, bytes, cudaMemcpyDeviceToHost));

    cuda_check(cudaFree(da));
    cuda_check(cudaFree(db));
    cuda_check(cudaFree(dc));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }

  std::printf("vector_add ok: c[0]=%.1f c[%d]=%.1f\n", hc[0], n - 1,
              hc[n - 1]);
  return 0;
}
