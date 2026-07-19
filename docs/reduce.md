# Nsight Compute Profiler Report

GPU: NVIDIA GeForce RTX 4060 Laptop

N = 10,000,000 elements

| Version | Duration (us) | Compute % | Memory % | L1/TEX % | DRAM % | Comment                                                |
| ------- | ------------- | --------- | -------- | -------- | ------ | ------------------------------------------------------ |
| v0      | 683.55        | 73.34     | 54.08    | 54.18    | 28.89  | Baseline                                               |
| v1      | 430.62        | 85.84     | 85.84    | 86.08    | 38.22  | Remove warp divergence, but introduce bank conflicts   |
| v2      | 415.20        | 89.02     | 89.02    | 89.28    | 39.65  | Remove bank conflicts                                  |
| v3      | 240.03        | 79.84     | 89.87    | 80.31    | 89.87  | Use 1 thread block to compute 2 array blocks           |
| v4      | 217.25        | 34.35     | 95.98    | 34.72    | 95.98  | Warp-shuffle the last warp, dropping its __syncthreads |
| v5      | 219.68        | 33.96     | 96.09    | 34.37    | 96.09  | Use `template <int kBlockSize>` for kernel             |
