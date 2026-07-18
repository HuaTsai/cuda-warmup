# Nsight Compute Profiler Report

GPU: NVIDIA GeForce RTX 4060 Laptop

| Version | N   | Duration (us) | Compute % | Memory % | L1/TEX % | DRAM % | Comment                      |
| ------- | --- | ------------- | --------- | -------- | -------- | ------ | ---------------------------- |
| v0      | 1M  | 70.78         | 70.97     | 52.28    | 53.23    | 22.74  | Baseline                     |
| v1      | 1M  | 45.15         | 81.96     | 81.96    | 84.21    | 35.74  | Remove divergent warps in v0 |
