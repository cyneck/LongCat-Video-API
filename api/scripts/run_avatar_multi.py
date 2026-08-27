"""Compatibility entrypoint for avatar-multi with task progress hooks."""
import torch.distributed as dist

from api.progress import install_worker_progress_hooks

import run_avatar_multi_impl as impl  # noqa: E402

install_worker_progress_hooks(impl)


if __name__ == "__main__":
    args = impl._parse_args()
    try:
        impl.generate(args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
