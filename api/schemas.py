"""Pydantic request schemas for the generation endpoints.

All material inputs (image / video / audio) are passed as paths returned by
the /files/upload endpoints — we keep heavy binary data out of the JSON body
and avoid re-uploading on retries.

The default avatar-v1.5 runtime profile targets one A100 40GB GPU. While
LONGCAT_A100_40G_PROFILE / config.toml a100_40g_enabled (unset ⇒ 按真实显存自动判定：≤50GB 才启用), v1.5 requests are clamped to SAFE
CEILINGS (max resolution / max segments, and forced INT8 + 8-step distill for
memory safety) so a 40GB card does not OOM. Caller-supplied values BELOW the
ceiling, and the guidance scales, are always honoured — this profile only ever
*reduces* memory-bound settings, never silently rewrites creative intent. Any
change it makes is logged as a warning. All profile values live in
config.toml [profile] (or the LONGCAT_A100_40G_* env vars), not in code.
Set LONGCAT_A100_40G_PROFILE=0 on larger systems to disable the clamp entirely.
"""
import os
import logging
from typing import Optional, Dict, List, Union
from pydantic import BaseModel, Field, model_validator

from . import config

logger = logging.getLogger("longcat.schemas")

# resolution string -> (height, width) for ceiling comparison
_RES_PX = {
    "480p": (480, 832),
    "720p": (720, 1280),
    "1080p": (1080, 1920),
}


def _a100_40g_profile_enabled() -> bool:
    # value is resolved in config.py (config.toml / env > 真实显存自动探测；不再默认开启)
    return config.LOW_VRAM_PROFILE_ENABLED


class TextToVideoRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    height: int = 480
    width: int = 832
    num_frames: int = 93
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    spatial_refine_only: bool = False
    seed: int = 42


class ImageToVideoRequest(BaseModel):
    cond_image: str = Field(..., description="upload path from /files/image")
    prompt: str
    negative_prompt: Optional[str] = None
    resolution: str = "480p"
    num_frames: int = 93
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    spatial_refine_only: bool = False
    seed: int = 42


class VideoContinuationRequest(BaseModel):
    cond_video: str = Field(..., description="upload path from /files/video")
    prompt: str
    negative_prompt: Optional[str] = None
    resolution: str = "480p"
    num_frames: int = 93
    num_cond_frames: int = 13
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    spatial_refine_only: bool = False
    seed: int = 42


class _Avatar40GDefaults(BaseModel):
    """Safe-ceiling normalization for the single-A100 production profile.

    Values come from config.LOW_VRAM (config.toml [profile], overridable by the
    LONGCAT_A100_40G_* env vars). Nothing here is hard-coded in this file.
    """

    @model_validator(mode="after")
    def apply_a100_40g_profile(self):
        if not _a100_40g_profile_enabled() or self.model_type != "avatar-v1.5":
            return self
        p = config.LOW_VRAM
        changed = []

        # --- memory-safety: required on a 40G card, cannot be opted out ---
        if p["force_use_int8"] and not self.use_int8:
            self.use_int8 = True
            changed.append("use_int8=True")
        if p["force_use_distill"] and not self.use_distill:
            self.use_distill = True
            changed.append("use_distill=True")
        if self.num_inference_steps != p["forced_num_inference_steps"]:
            self.num_inference_steps = p["forced_num_inference_steps"]
            changed.append(f"num_inference_steps={p['forced_num_inference_steps']}")

        # --- ceiling clamp: only reduce, never upgrade ---
        cur = _RES_PX.get(self.resolution)
        cap = _RES_PX.get(p["max_resolution"])
        if cur and cap and (cur[0] > cap[0] or cur[1] > cap[1]):
            self.resolution = p["max_resolution"]
            changed.append(f"resolution 封顶 {p['max_resolution']}")
        # 段数：仅当调用方传整数、且配置了上限(max>0)时做封顶；'auto' 字符串保留，
        # 交由 run 脚本按音频时长计算（滑动窗口串行，不额外占显存，无需强制压成 1）。
        if isinstance(self.num_segments, int) and p["max_num_segments"] > 0 and self.num_segments > p["max_num_segments"]:
            self.num_segments = p["max_num_segments"]
            changed.append(f"num_segments 封顶 {p['max_num_segments']}")

        # guidance scales are intentionally NOT touched — caller keeps creative control.
        if changed:
            logger.warning(
                "avatar-v1.5 请求已被 A100-40G profile 调整: %s", "; ".join(changed)
            )
        return self


class AvatarSingleRequest(_Avatar40GDefaults):
    prompt: str
    cond_audio: Dict[str, str] = Field(..., description='{"person1": "<audio upload path>"}')
    cond_image: Optional[str] = Field(None, description="required when stage_1='ai2v'")
    stage_1: str = Field("ai2v", pattern="^(at2v|ai2v)$")
    resolution: str = "480p"
    num_segments: Union[int, str] = Field("auto", description="视频段数：整数或 'auto'（按音频时长自动计算）")
    num_inference_steps: int = Field(8, ge=1, le=100)
    text_guidance_scale: float = config.LOW_VRAM["text_guidance_scale"]
    audio_guidance_scale: float = config.LOW_VRAM["audio_guidance_scale"]
    audio_drive_gain: float = Field(
        0.85,
        ge=0.5,
        le=1.2,
        description="音频驱动 embedding 增益；<1 可抑制夸张嘴型，不改变最终输出音量",
    )
    ref_img_index: int = 10
    mask_frame_range: int = 3
    model_type: str = Field("avatar-v1.5", pattern="^(avatar-v1.0|avatar-v1.5)$")
    use_distill: bool = True
    use_int8: bool = True
    seed: int = 42


class AvatarMultiRequest(_Avatar40GDefaults):
    prompt: str
    cond_image: str = Field(..., description="upload path from /files/image")
    cond_audio: Dict[str, str] = Field(
        ..., description='{"person1": "<path>", "person2": "<path>"} — at least one required'
    )
    audio_type: str = Field("para", pattern="^(para|add)$")
    bbox: Optional[Dict[str, List[int]]] = None
    resolution: str = "480p"
    num_segments: Union[int, str] = Field("auto", description="视频段数：整数或 'auto'（按音频时长自动计算）")
    num_inference_steps: int = Field(8, ge=1, le=100)
    text_guidance_scale: float = config.LOW_VRAM["text_guidance_scale"]
    audio_guidance_scale: float = config.LOW_VRAM["audio_guidance_scale"]
    audio_drive_gain: float = Field(
        0.85,
        ge=0.5,
        le=1.2,
        description="音频驱动 embedding 增益；<1 可抑制夸张嘴型，不改变最终输出音量",
    )
    ref_img_index: int = 10
    mask_frame_range: int = 3
    model_type: str = Field("avatar-v1.5", pattern="^(avatar-v1.0|avatar-v1.5)$")
    use_distill: bool = True
    use_int8: bool = True
    seed: int = 42
