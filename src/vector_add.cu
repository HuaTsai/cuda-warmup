#include "vector_add.cuh"

namespace {

__global__ void vector_add_kernel(const float* a, const float* b, float* c,
                                  int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

}  // namespace

cudaError_t vector_add(const float* a, const float* b, float* c, int n) {
  if (n <= 0) {
    return cudaSuccess;
  }
  constexpr int kBlockSize = 256;
  const int grid = (n + kBlockSize - 1) / kBlockSize;
  vector_add_kernel<<<grid, kBlockSize>>>(a, b, c, n);
  return cudaGetLastError();
}
