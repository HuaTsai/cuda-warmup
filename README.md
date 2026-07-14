# cuda-warmup

CUDA 練習專案，環境由 [pixi](https://pixi.sh) 管理（CUDA 13.3、CMake、Ninja、GoogleTest）。

## 使用

```sh
pixi run test    # configure + build + 跑測試
pixi run build   # 只建置
```

## 結構

- `src/` — CUDA kernels
- `tests/` — GoogleTest 測試
