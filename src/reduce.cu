#include "device_buffer.hpp"
#include "reduce.hpp"

namespace {
__global__ void ReduceSumV0Kernel(const float *d_in, float *d_out, int n) {
  __shared__ float sdata[256];
  int bdim = blockDim.x;
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int idx = bid * bdim + tid;

  sdata[tid] = (idx < n) ? d_in[idx] : 0.0f;
  __syncthreads();

  for (int s = 1; s < bdim; s <<= 1) {
    if (tid % (2 * s) == 0) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    d_out[bid] = sdata[0];
  }
}
}  // namespace

cudaError_t ReduceSumV0(const float *d_in, float *d_out, int n) {
  if (n <= 0) {
    return cudaErrorInvalidValue;
  }

  constexpr int kBlockSize = 256;
  int blocks = (n + kBlockSize - 1) / kBlockSize;
  DeviceBuffer<float> d_mid(blocks);
  ReduceSumV0Kernel<<<blocks, kBlockSize>>>(d_in, d_mid.get(), n);
  n = blocks;

  while (n > 1) {
    blocks = (n + kBlockSize - 1) / kBlockSize;
    DeviceBuffer<float> d_mid2(blocks);
    ReduceSumV0Kernel<<<blocks, kBlockSize>>>(d_mid.get(), d_mid2.get(), n);

    d_mid = std::move(d_mid2);
    n = blocks;
  }
  cudaMemcpy(d_out, d_mid.get(), sizeof(float), cudaMemcpyDeviceToDevice);

  return cudaGetLastError();
}

// cudaError_t ReduceSumV1(const float *d_in, float *d_out, int n);

// cudaError_t ReduceSumV2(const float *d_in, float *d_out, int n);

// cudaError_t ReduceSumV3(const float *d_in, float *d_out, int n);

// cudaError_t ReduceSumV4(const float *d_in, float *d_out, int n);

// cudaError_t ReduceSumV5(const float *d_in, float *d_out, int n);

// cudaError_t ReduceSumV6(const float *d_in, float *d_out, int n);

// cudaError_t ReduceSumV7(const float *d_in, float *d_out, int n);
