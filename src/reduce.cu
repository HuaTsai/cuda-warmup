#include "reduce.hpp"

namespace {
__global__ void ReduceSumV0Kernel(float *d_in, float *d_out, int n) {
  int tid = threadIdx.x;
  for (int s = 1; s < n; s <<= 1) {
    if (tid % (2 * s) == 0 && tid + s < n) {
      d_in[tid] += d_in[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    *d_out = d_in[0];
  }
}
}  // namespace

// Maximum: 1024 elements = 1024 threads
cudaError_t ReduceSumV0(float *d_in, float *d_out, int n) {
  constexpr int kBlocks = 1;
  constexpr int kMaxThreadsPerBlock = 1024;
  if (n <= 0 || n > kMaxThreadsPerBlock) {
    return cudaErrorInvalidValue;
  }
  ReduceSumV0Kernel<<<kBlocks, n>>>(d_in, d_out, n);
  return cudaGetLastError();
}

// cudaError_t ReduceSumV1(float *d_in, float *d_out, int n);

// cudaError_t ReduceSumV2(float *d_in, float *d_out, int n);

// cudaError_t ReduceSumV3(float *d_in, float *d_out, int n);

// cudaError_t ReduceSumV4(float *d_in, float *d_out, int n);

// cudaError_t ReduceSumV5(float *d_in, float *d_out, int n);

// cudaError_t ReduceSumV6(float *d_in, float *d_out, int n);

// cudaError_t ReduceSumV7(float *d_in, float *d_out, int n);
