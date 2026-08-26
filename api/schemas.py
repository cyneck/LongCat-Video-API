"""Pydantic request schemas for the generation endpoints.

All material inputs (image / video / audio) are passed as paths returned by
the /files/upload endpoints — we keep heavy binary data out of the JSON body
and avoid re-uploading on retries.

Avatar v1.5 defaults intentionally target a single 40 GB Ampere GPU: INT8 DiT,
8-step distilled inference, one segment and 480p. Callers can explicitly
opt out when running on larger/multi-GPU systems.
"""
from typing import Optional, Dict, List
from pydantic import BaseModel, Field


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


class AvatarSingleRequest(BaseModel):
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


class AvatarMultiRequest(BaseModel):
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
