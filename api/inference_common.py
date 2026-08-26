"""Shared avatar model loading for the in-process (resident worker) path.

Both ``api/scripts/run_avatar_single.py`` and ``api/scripts/run_avatar_multi.py``
used to duplicate the ~60-line block that loads the tokenizer / text_encoder /
vae / scheduler / DiT / audio models and builds the pipeline. This module
extracts that logic into a single ``load_avatar_models(...)`` so the resident
``inference_worker`` can load the model **once** at startup and reuse it for
every request (no per-request cold start).
"""
import os
from pathlib import Path

import torch

from transformers import AutoTokenizer, UMT5EncoderModel

from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from longcat_video.modules.quantization import load_quantized_dit
from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
from audio_separator.separator import Separator


def resolve_base_model_dir(checkpoint_dir: str) -> str:
    """The base video model supplies the shared tokenizer / text_encoder / vae /
    scheduler. Default: sibling dir ``LongCat-Video`` next to the avatar
    checkpoint. Override with ``LONGCAT_CHECKPOINT_DIR_VIDEO`` to support
    arbitrary weights layouts.
    """
    return os.environ.get("LONGCAT_CHECKPOINT_DIR_VIDEO") or os.path.join(
        checkpoint_dir, "..", "LongCat-Video"
    )


def load_avatar_models(
    checkpoint_dir: str,
    model_type: str = "avatar-v1.5",
    use_int8: bool = False,
    use_distill: bool = False,
    local_rank: int = 0,
    cp_split_hw=None,
    avatar_mode: str = "single",
):
    """Load all avatar models once and return them as a dict.

    ``avatar_mode`` selects the DiT subfolder for the v1.0 checkpoint
    (``avatar_single`` vs ``avatar_multi``); v1.5 uses ``base_model`` for both.
    The caller is responsible for having already initialised the distributed
    process group + context parallel (so ``local_rank`` / ``cp_split_hw`` are
    meaningful).
    """
    base_model_dir = resolve_base_model_dir(checkpoint_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        base_model_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    vae = AutoencoderKLWan.from_pretrained(
        base_model_dir, subfolder="vae", torch_dtype=torch.bfloat16
    )

    if model_type == "avatar-v1.0":
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base_model_dir, subfolder="scheduler", torch_dtype=torch.bfloat16
        )
    elif model_type == "avatar-v1.5":
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16
        )
    else:
        raise ValueError(
            f"Unsupported model_type: {model_type}. Expected 'avatar-v1.0' or 'avatar-v1.5'."
        )

    if model_type == "avatar-v1.0":
        dit_subfolder = "avatar_single" if avatar_mode == "single" else "avatar_multi"
        dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(
            checkpoint_dir,
            subfolder=dit_subfolder,
            cp_split_hw=cp_split_hw,
            torch_dtype=torch.bfloat16,
        )
    elif model_type == "avatar-v1.5":
        if use_int8:
            print("[INFO] Loading INT8 quantized DiT model...")
            dit = load_quantized_dit(
                checkpoint_dir, subfolder="base_model_int8", cp_split_hw=cp_split_hw
            )
        else:
            dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(
                checkpoint_dir,
                subfolder="base_model",
                cp_split_hw=cp_split_hw,
                torch_dtype=torch.bfloat16,
            )
        if use_distill:
            distill_checkpoint_path = os.path.join(checkpoint_dir, "lora", "dmd_lora.safetensors")
            if os.path.exists(distill_checkpoint_path):
                dit.load_lora(
                    distill_checkpoint_path, "dmd", multiplier=1.0,
                    lora_network_dim=128, lora_network_alpha=64,
                )
                dit.enable_loras(["dmd"])
    else:
        raise ValueError(
            f"Unsupported model_type: {model_type}. Expected 'avatar-v1.0' or 'avatar-v1.5'."
        )

    # --- audio models ---
    if model_type == "avatar-v1.0":
        audio_model_checkpoint_path = os.path.join(checkpoint_dir, "chinese-wav2vec2-base")
    elif model_type == "avatar-v1.5":
        audio_model_checkpoint_path = os.path.join(checkpoint_dir, "whisper-large-v3")
    audio_encoder = get_audio_encoder(audio_model_checkpoint_path, model_type).to(local_rank)
    audio_feature_extractor = get_audio_feature_extractor(audio_model_checkpoint_path, model_type)

    vocal_separator_path = os.path.join(checkpoint_dir, "vocal_separator/Kim_Vocal_2.onnx")
    audio_output_dir_temp = Path("./audio_temp_file")
    os.makedirs(audio_output_dir_temp, exist_ok=True)
    audio_separator_model_path = os.path.dirname(vocal_separator_path)
    audio_separator_model_name = os.path.basename(vocal_separator_path)
    vocal_separator = Separator(
        output_dir=audio_output_dir_temp / "vocals",
        output_single_stem="vocals",
        model_file_dir=audio_separator_model_path,
    )
    vocal_separator.load_model(audio_separator_model_name)

    pipe = LongCatVideoAvatarPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
        audio_encoder=audio_encoder,
        audio_feature_extractor=audio_feature_extractor,
        model_type=model_type,
    )
    pipe.to(local_rank)

    return {
        "pipe": pipe,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "scheduler": scheduler,
        "dit": dit,
        "audio_encoder": audio_encoder,
        "audio_feature_extractor": audio_feature_extractor,
        "vocal_separator": vocal_separator,
        "audio_output_dir_temp": audio_output_dir_temp,
        "model_type": model_type,
        "local_rank": local_rank,
    }
