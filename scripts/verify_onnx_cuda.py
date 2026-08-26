#!/usr/bin/env python3
"""Verify that ONNX Runtime can actually use CUDA for the vocal separator.

Import torch first so its CUDA/cuDNN shared libraries are loaded before ORT,
which is the compatibility path recommended by ONNX Runtime for PyTorch-based
applications. ORT >= 1.21 also exposes preload_dlls(); call it when available.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import onnxruntime as ort


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="weights/LongCat-Video-Avatar-1.5/vocal_separator/Kim_Vocal_2.onnx",
        help="ONNX model used to validate a real CUDA session",
    )
    args = parser.parse_args()

    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception as exc:  # diagnostic only; provider check below is authoritative
            print(f"[onnx-cuda] preload_dlls warning: {exc}")

    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"onnxruntime={ort.__version__}")
    providers = ort.get_available_providers()
    print(f"available_providers={providers}")

    if "CUDAExecutionProvider" not in providers:
        print(
            "[onnx-cuda] FAIL: CUDAExecutionProvider is unavailable. "
            "Remove the CPU onnxruntime package and install the CUDA 12 GPU package."
        )
        return 2

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"[onnx-cuda] CUDA provider is available; model not found, skipping session test: {model_path}")
        return 0

    session = ort.InferenceSession(
        str(model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    active = session.get_providers()
    print(f"session_providers={active}")
    if not active or active[0] != "CUDAExecutionProvider":
        print("[onnx-cuda] FAIL: real model session did not select CUDAExecutionProvider first")
        return 3

    print("[onnx-cuda] OK: vocal separator ONNX session is CUDA accelerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
