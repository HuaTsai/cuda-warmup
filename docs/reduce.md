# Nsight Compute Profiler Report

GPU: NVIDIA GeForce RTX 4060 Laptop

N = 10,000,000 elements

| Version | Duration (us) | Compute % | Memory % | L1/TEX % | DRAM % | Comment                      |
| ------- | ------------- | --------- | -------- | -------- | ------ | ---------------------------- |
| v0      | 683.55        | 73.34     | 54.08    | 54.18    | 28.89  | Baseline                     |
| v1      | 430.62        | 85.84     | 85.84    | 86.08    | 38.22  | Remove divergent warps in v0 |
| v2      | 415.20        | 89.02     | 89.02    | 89.28    | 39.65  | Remove bank conflicts in v1  |
