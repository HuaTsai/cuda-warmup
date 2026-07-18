# Nsight Compute Profiler Report

GPU: NVIDIA GeForce RTX 4060 Laptop

N = 10,000,000 elements

| Version | Duration (us) | Compute % | Memory % | L1/TEX % | DRAM % | Comment                      |
| ------- | ------------- | --------- | -------- | -------- | ------ | ---------------------------- |
| v0      | 685.66        | 73.12     | 53.91    | 54.01    | 31.67  | Baseline                     |
| v1      | 433.28        | 85.31     | 85.31    | 85.55    | 50.08  | Remove divergent warps in v0 |
