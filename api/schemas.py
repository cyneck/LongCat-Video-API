"""Pydantic request schemas for the generation endpoints.

All material inputs (image / video / audio) are passed as paths returned by
the /files/upload endpoints — we keep heavy binary data out of the JSON body
and avoid re-uploading on retries.

The default avatar-v1.5 runtime profile targets one A100 40GB GPU. While
LONGCAT_A100_40G_PROFILE=1 (the default), v1.5 requests are normalized to the
INT8 + 8-step distilled path even if an older client/H5 explicitly sends the
legacy high-memory values. Set LONGCAT_A100_40G_PROFILE=0 on larger systems to
restore caller-controlled tuning.
"""
import os
from typing import Optional, Dict, List
from pydantic import BaseModel, Field, model_validator


def _a100_40g_profile_enabled() -> bool:
    return os.environ.get("LONGCAT_A100_40G_PROFILE", "1") == "1"


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
    """Shared normalization for the single-A100 production profile."""

    @model_validator(mode="after")
    def apply_a100_40g_profile(self):
        if _a100_40g_profile_enabled() and self.model_type == "avatar-v1.5":
            self.resolution = "480p"
            self.num_segments = 1
            self.num_inference_steps = 8
            self.text_guidance_scale = 1.0
            self.audio_guidance_scale = 1.0
            self.use_distill = True
            self.use_int8 = True
        return self


class AvatarSingleRequest(_Avatar40GDefaults):
    prompt: str
    cond_audio: Dict[str, str] = Field(..., description='{"person1": "<audio upload path>"}')
    cond_image: Optional[str] = Field(None, description="required when stage_1='ai2v'")
    stage_1: str = Field("ai2v", pattern="^(at2v|ai2v)$")
    resolution: str = "480p"
    num_segments: int = Field(1, ge=1, le=10)
    num_inference_steps: int = Field(8, ge=1, le=100)
    text_guidance_scale: float = 1.0
    audio_guidance_scale: float = 1.0
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
    num_segments: int = Field(1, ge=1, le=10)
    num_inference_steps: int = Field(8, ge=1, le=100)
    text_guidance_scale: float = 1.0
    audio_guidance_scale: float = 1.0
    ref_img_index: int = 10
    mask_frame_range: int = 3
    model_type: str = Field("avatar-v1.5", pattern="^(avatar-v1.0|avatar-v1.5)$")
    use_distill: bool = True
    use_int8: bool = True
    seed: int = 42
