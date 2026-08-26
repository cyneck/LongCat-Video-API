# ONNX Runtime GPU acceleration for vocal separation

Avatar preprocessing uses `audio-separator` with `Kim_Vocal_2.onnx`. If logs show:

```text
CUDAExecutionProvider not available
```

then the separator is running on CPU even though PyTorch can use the GPU.

## Recommended environment

This repository targets PyTorch 2.6.0 + CUDA 12.4. `requirements_avatar.txt` therefore uses:

```text
onnxruntime-gpu==1.26.0
```

ORT 1.21–1.26 GPU wheels use CUDA 12.x + cuDNN 9 and are compatible with the PyTorch 2.4+ CUDA 12 stack. ORT 1.27+ defaults to CUDA 13, so this deployment intentionally stays on 1.26.x.

Do **not** keep both `onnxruntime` and `onnxruntime-gpu` installed. They expose the same Python package name and can leave the CPU build active.

## Upgrade an existing deployment

After `git pull`, clean the old CPU package and reinstall the Avatar requirements:

```bash
python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install -r requirements_avatar.txt
```

Then verify the real separator model:

```bash
python scripts/verify_onnx_cuda.py
```

Expected result:

```text
available_providers=['CUDAExecutionProvider', ...]
session_providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
[onnx-cuda] OK: vocal separator ONNX session is CUDA accelerated
```

The worker imports PyTorch before ONNX Runtime / `audio-separator`, allowing ORT to reuse the CUDA and cuDNN runtime libraries shipped with the PyTorch CUDA package. ORT 1.26 also exposes `preload_dlls()`; the verification utility calls it when available.

## Failure behavior

The GPU package still contains CPU execution support. If CUDA provider initialization fails, `audio-separator` can fall back to CPU rather than breaking Avatar generation. Check the worker log for the provider list before comparing performance.

GPU acceleration here only affects vocal separation. The DiT / VAE / Whisper Avatar inference path remains unchanged.
