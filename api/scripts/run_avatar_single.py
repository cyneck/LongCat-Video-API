"""Compatibility entrypoint for avatar-single with standardized task progress."""
import torch.distributed as dist

from api.progress import install_worker_progress_hooks
import run_avatar_single_impl as impl


install_worker_progress_hooks(impl)


if __name__ == "__main__":
    args = impl._parse_args()
    try:
        impl.generate(args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
