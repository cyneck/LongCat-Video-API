"""Compatibility entrypoint for avatar-multi.

Preserves lightweight text-encoder metadata after UMT5 offload and installs the
shared machine-readable task progress reporter before entering the worker.
"""
import torch
import torch.distributed as dist

from api.progress import install_worker_progress_hooks
from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline


_original_setattr = LongCatVideoAvatarPipeline.__setattr__
_original_cache_clean_latents = LongCatVideoAvatarPipeline._cache_clean_latents


def _capture_text_encoder_metadata(self, name, value):
    if name == "text_encoder" and value is not None:
        cfg = getattr(value, "config", None)
        d_model = getattr(cfg, "d_model", None)
        if d_model is not None:
            _original_setattr(self, "_longcat_text_d_model", int(d_model))
    return _original_setattr(self, name, value)


def _cache_clean_latents_without_text_encoder(
    self,
    cond_latents,
    model_max_length,
    offload_kv_cache,
    device,
    dtype,
    audio_embs,
    num_cond_latents,
    num_ref_latents,
    ref_img_index,
):
    text_encoder = getattr(self, "text_encoder", None)
    if text_encoder is not None:
        return _original_cache_clean_latents(
            self,
            cond_latents,
            model_max_length,
            offload_kv_cache,
            device,
            dtype,
            audio_embs,
            num_cond_latents,
            num_ref_latents,
            ref_img_index,
        )

    d_model = getattr(self, "_longcat_text_d_model", None)
    if d_model is None:
        raise RuntimeError(
            "Avatar continuation needs cached text encoder d_model metadata after UMT5 offload"
        )

    timestep = torch.zeros(cond_latents.shape[0], cond_latents.shape[2]).to(
        device=device, dtype=dtype
    )
    empty_embeds = torch.zeros(
        [cond_latents.shape[0], 1, model_max_length, d_model],
        device=device,
        dtype=dtype,
    )
    _, kv_cache_dict = self.dit(
        hidden_states=cond_latents,
        timestep=timestep,
        encoder_hidden_states=empty_embeds,
        num_cond_latents=num_cond_latents,
        return_kv=True,
        skip_crs_attn=True,
        offload_kv_cache=offload_kv_cache,
        audio_embs=audio_embs,
        num_ref_latents=num_ref_latents,
        ref_img_index=ref_img_index,
    )
    self._update_kv_cache_dict(kv_cache_dict)


LongCatVideoAvatarPipeline.__setattr__ = _capture_text_encoder_metadata
LongCatVideoAvatarPipeline._cache_clean_latents = _cache_clean_latents_without_text_encoder

import run_avatar_multi_impl as impl  # noqa: E402

install_worker_progress_hooks(impl)


if __name__ == "__main__":
    args = impl._parse_args()
    try:
        impl.generate(args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
