"""Compatibility entrypoint for avatar-single with standardized task progress.

The low-memory profile keeps large components on CPU while they are being
assembled, but asks Hugging Face / Diffusers to use their low-CPU-memory loading
path.  Do NOT move UMT5/VAE/DiT to CUDA eagerly here: doing so makes all large
components overlap on a 40GB GPU before the pipeline has a chance to offload
short-lived encoders, which can push startup to ~39.5GiB and OOM.
"""
import os

import torch.distributed as dist

from api.progress import install_worker_progress_hooks
import run_avatar_single_impl as impl


def _install_balanced_memory_load_hooks():
    """Reduce host-RAM load peaks without eagerly consuming GPU memory."""
    if not bool(getattr(impl.config, "LOW_VRAM_PROFILE_ENABLED", False)):
        return

    original_t5_from_pretrained = impl.UMT5EncoderModel.from_pretrained
    original_vae_from_pretrained = impl.AutoencoderKLWan.from_pretrained

    class _T5Loader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            kwargs.setdefault("low_cpu_mem_usage", True)
            model = original_t5_from_pretrained(*args, **kwargs)
            if int(os.environ.get("RANK", "0")) == 0:
                print("[longcat][memory] loaded component=umt5 mode=low_cpu_mem_usage device=cpu", flush=True)
            return model

    class _VAELoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            kwargs.setdefault("low_cpu_mem_usage", True)
            model = original_vae_from_pretrained(*args, **kwargs)
            if int(os.environ.get("RANK", "0")) == 0:
                print("[longcat][memory] loaded component=vae mode=low_cpu_mem_usage device=cpu", flush=True)
            return model

    impl.UMT5EncoderModel = _T5Loader
    impl.AutoencoderKLWan = _VAELoader
    # INT8 DiT already uses a meta skeleton + shard-by-shard materialization in
    # longcat_video.modules.quantization.load_quantized_dit.  Leave it on CPU
    # until the normal pipeline placement step; eager CUDA placement caused the
    # observed 38.88GiB allocated / 39.49GiB total startup OOM.


_install_balanced_memory_load_hooks()
install_worker_progress_hooks(impl)


if __name__ == "__main__":
    args = impl._parse_args()
    try:
        impl.generate(args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
