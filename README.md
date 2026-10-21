# 煉丹爐

## 安裝

預設安裝 CPU 版本的 PyTorch：

```bash
uv sync
```

安裝 GPU 版本的 PyTorch (二選一)：

```bash
uv sync --extra cu130  # CUDA 13.0，需 NVIDIA 驅動 580 以上
uv sync --extra cu128  # CUDA 12.8 (PyTorch 官方最高只提供到 torch 2.11)
```
