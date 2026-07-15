#pragma once

#include <cuda_runtime.h>

// c = a + b, where a/b/c are device pointers of length n.
// Launches the kernel asynchronously on the default stream; the return value
// only covers whether the launch succeeded — kernel runtime errors surface
// at the caller's cudaDeviceSynchronize.
cudaError_t VectorAdd(const float *a, const float *b, float *c, int n);
