#pragma once

#include <cuda_runtime.h>

// c = a + b，a/b/c 皆為 device pointer，長度 n。
// 在 default stream 上非同步啟動 kernel；回傳值只涵蓋 launch 是否成功，
// kernel 執行期錯誤要等呼叫端 cudaDeviceSynchronize 才會浮現。
cudaError_t vector_add(const float* a, const float* b, float* c, int n);
