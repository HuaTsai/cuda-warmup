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

__global__ void ReduceSumV1Kernel(const float *d_in, float *d_out, int n) {
  __shared__ float sdata[256];
  int bdim = blockDim.x;
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int idx = bid * bdim + tid;

  sdata[tid] = (idx < n) ? d_in[idx] : 0.0f;
  __syncthreads();

  for (int s = 1; s < bdim; s <<= 1) {
    int index = 2 * s * tid;
    if (index < bdim) {
      sdata[index] += sdata[index + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    d_out[bid] = sdata[0];
  }
}

__global__ void ReduceSumV2Kernel(const float *d_in, float *d_out, int n) {
  __shared__ float sdata[256];
  int bdim = blockDim.x;
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int idx = bid * bdim + tid;

  sdata[tid] = (idx < n) ? d_in[idx] : 0.f;
  __syncthreads();

  for (int s = bdim / 2; s >= 1; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    d_out[bid] = sdata[0];
  }
}

__global__ void ReduceSumV3Kernel(const float *d_in, float *d_out, int n) {
  __shared__ float sdata[256];
  int bdim = blockDim.x;
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int idx1 = (bid * 2) * bdim + tid;
  int idx2 = idx1 + bdim;

  sdata[tid] = (idx1 < n ? d_in[idx1] : 0.f) + (idx2 < n ? d_in[idx2] : 0.f);
  __syncthreads();

  for (int s = bdim / 2; s >= 1; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    d_out[bid] = sdata[0];
  }
}

__global__ void ReduceSumV4Kernel(const float *d_in, float *d_out, int n) {
  __shared__ float sdata[256];
  int bdim = blockDim.x;
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int idx1 = (bid * 2) * bdim + tid;
  int idx2 = idx1 + bdim;

  sdata[tid] = (idx1 < n ? d_in[idx1] : 0.f) + (idx2 < n ? d_in[idx2] : 0.f);
  __syncthreads();

  for (int s = bdim / 2; s >= 64; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    float val = sdata[tid] + sdata[tid + 32];
    for (int s = 16; s >= 1; s >>= 1) {
      val += __shfl_down_sync(0xffffffff, val, s);
    }
    if (tid == 0) {
      d_out[bid] = val;
    }
  }
}

template <int kBlockSize>
__global__ void ReduceSumV5Kernel(const float *d_in, float *d_out, int n) {
  __shared__ float sdata[kBlockSize];
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int idx1 = (bid * 2) * kBlockSize + tid;
  int idx2 = idx1 + kBlockSize;

  sdata[tid] = (idx1 < n ? d_in[idx1] : 0.f) + (idx2 < n ? d_in[idx2] : 0.f);
  __syncthreads();

  for (int s = kBlockSize / 2; s >= 64; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid < 32) {
    float val = sdata[tid] + sdata[tid + 32];
    for (int s = 16; s >= 1; s >>= 1) {
      val += __shfl_down_sync(0xffffffff, val, s);
    }
    if (tid == 0) {
      d_out[bid] = val;
    }
  }
}
}  // namespace

template <void (*Kernel)(const float *, float *, int)>
cudaError_t ReduceSumV0toV2(const float *d_in, float *d_out, int n) {
  if (n <= 0) {
    return cudaErrorInvalidValue;
  }

  constexpr int kBlockSize = 256;
  int blocks = (n + kBlockSize - 1) / kBlockSize;
  DeviceBuffer<float> d_mid(blocks);
  Kernel<<<blocks, kBlockSize>>>(d_in, d_mid.get(), n);
  n = blocks;

  while (n > 1) {
    blocks = (n + kBlockSize - 1) / kBlockSize;
    DeviceBuffer<float> d_mid2(blocks);
    Kernel<<<blocks, kBlockSize>>>(d_mid.get(), d_mid2.get(), n);

    d_mid = std::move(d_mid2);
    n = blocks;
  }
  cudaMemcpy(d_out, d_mid.get(), sizeof(float), cudaMemcpyDeviceToDevice);

  return cudaGetLastError();
}

cudaError_t ReduceSumV0(const float *d_in, float *d_out, int n) {
  return ReduceSumV0toV2<ReduceSumV0Kernel>(d_in, d_out, n);
}

cudaError_t ReduceSumV1(const float *d_in, float *d_out, int n) {
  return ReduceSumV0toV2<ReduceSumV1Kernel>(d_in, d_out, n);
}

cudaError_t ReduceSumV2(const float *d_in, float *d_out, int n) {
  return ReduceSumV0toV2<ReduceSumV2Kernel>(d_in, d_out, n);
}

template <void (*Kernel)(const float *, float *, int)>
cudaError_t ReduceSumV3toV4(const float *d_in, float *d_out, int n) {
  if (n <= 0) {
    return cudaErrorInvalidValue;
  }

  constexpr int kBlockSize = 256;
  int blocks = (n + (2 * kBlockSize) - 1) / (2 * kBlockSize);
  DeviceBuffer<float> d_mid(blocks);
  Kernel<<<blocks, kBlockSize>>>(d_in, d_mid.get(), n);
  n = blocks;

  while (n > 1) {
    blocks = (n + (2 * kBlockSize) - 1) / (2 * kBlockSize);
    DeviceBuffer<float> d_mid2(blocks);
    Kernel<<<blocks, kBlockSize>>>(d_mid.get(), d_mid2.get(), n);

    d_mid = std::move(d_mid2);
    n = blocks;
  }
  cudaMemcpy(d_out, d_mid.get(), sizeof(float), cudaMemcpyDeviceToDevice);

  return cudaGetLastError();
}

cudaError_t ReduceSumV3(const float *d_in, float *d_out, int n) {
  return ReduceSumV3toV4<ReduceSumV3Kernel>(d_in, d_out, n);
}

cudaError_t ReduceSumV4(const float *d_in, float *d_out, int n) {
  return ReduceSumV3toV4<ReduceSumV4Kernel>(d_in, d_out, n);
}

template <int kBlockSize>
cudaError_t ReduceSumV5(const float *d_in, float *d_out, int n) {
  if (n <= 0) {
    return cudaErrorInvalidValue;
  }

  int blocks = (n + (2 * kBlockSize) - 1) / (2 * kBlockSize);
  DeviceBuffer<float> d_mid(blocks);
  ReduceSumV5Kernel<kBlockSize><<<blocks, kBlockSize>>>(d_in, d_mid.get(), n);
  n = blocks;

  while (n > 1) {
    blocks = (n + (2 * kBlockSize) - 1) / (2 * kBlockSize);
    DeviceBuffer<float> d_mid2(blocks);
    ReduceSumV5Kernel<kBlockSize><<<blocks, kBlockSize>>>(d_mid.get(), d_mid2.get(), n);

    d_mid = std::move(d_mid2);
    n = blocks;
  }
  cudaMemcpy(d_out, d_mid.get(), sizeof(float), cudaMemcpyDeviceToDevice);

  return cudaGetLastError();
}

template cudaError_t ReduceSumV5<64>(const float *d_in, float *d_out, int n);

template cudaError_t ReduceSumV5<128>(const float *d_in, float *d_out, int n);

template cudaError_t ReduceSumV5<256>(const float *d_in, float *d_out, int n);

template cudaError_t ReduceSumV5<512>(const float *d_in, float *d_out, int n);

template cudaError_t ReduceSumV5<1024>(const float *d_in, float *d_out, int n);

// cudaError_t ReduceSumV6(const float *d_in, float *d_out, int n);

// cudaError_t ReduceSumV7(const float *d_in, float *d_out, int n);
