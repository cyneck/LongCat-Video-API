"""Compatibility entrypoint for avatar-single with standardized task progress.

On the low-VRAM A100 profile, large CPU-loaded components are moved to the
current CUDA device immediately after loading. The original worker kept UMT5,
VAE and then DiT resident in host RAM until ``pipe.to(cuda)``, which can trigger
the Linux/container OOM killer on a 40GB-RAM host before inference starts.
"""
import gc
import os

import torch
import torch.distributed as dist

from api.progress import install_worker_progress_hooks
import run_avatar_single_impl as impl


def _rss_gb():
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024 / 1024
    except Exception:
        pass
    return None


def _move_loaded_component_to_cuda(name, model):
    """Move one freshly loaded module off host RAM as early as possible."""
    if not torch.cuda.is_available():
        return model
    try:
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    except ValueError:
        local_rank = 0
    local_rank %= max(1, torch.cuda.device_count())

    before = _rss_gb()
    model = model.to(local_rank)
    gc.collect()
    torch.cuda.empty_cache()
    after = _rss_gb()
    if int(os.environ.get("RANK", "0")) == 0:
        before_s = f"{before:.2f}GB" if before is not None else "n/a"
        after_s = f"{after:.2f}GB" if after is not None else "n/a"
        allocated = torch.cuda.memory_allocated(local_rank) / 1024**3
        print(
            f"[longcat][memory] early_cuda_offload component={name} "
            f"rss_before={before_s} rss_after={after_s} gpu_alloc={allocated:.2f}GB",
            flush=True,
        )
    return model


def _install_low_host_ram_load_hooks():
    """Avoid cumulative CPU residency while loading sharded Avatar components."""
    low_ram_profile = bool(getattr(impl.config, "LOW_VRAM_PROFILE_ENABLED", False))
    if not low_ram_profile:
        return

    original_t5_from_pretrained = impl.UMT5EncoderModel.from_pretrained
    original_vae_from_pretrained = impl.AutoencoderKLWan.from_pretrained
    original_load_quantized_dit = impl.load_quantized_dit

    class _T5Loader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            model = original_t5_from_pretrained(*args, **kwargs)
            return _move_loaded_component_to_cuda("umt5", model)

    class _VAELoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            model = original_vae_from_pretrained(*args, **kwargs)
            return _move_loaded_component_to_cuda("vae", model)

    def _load_quantized_dit_low_ram(*args, **kwargs):
        model = original_load_quantized_dit(*args, **kwargs)
        return _move_loaded_component_to_cuda("dit_int8", model)

    impl.UMT5EncoderModel = _T5Loader
    impl.AutoencoderKLWan = _VAELoader
    impl.load_quantized_dit = _load_quantized_dit_low_ram


_install_low_host_ram_load_hooks()
install_worker_progress_hooks(impl)


if __name__ == "__main__":
    args = impl._parse_args()
    try:
        impl.generate(args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
