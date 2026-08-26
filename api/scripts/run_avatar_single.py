import os
import json
import time
import math
import random
import argparse
import datetime
import PIL.Image
import numpy as np
from pathlib import Path

import torch
import torch.distributed as dist

from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers.utils import load_image

from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from longcat_video.modules.quantization import load_quantized_dit
from longcat_video.context_parallel import context_parallel_util

import librosa
from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
from longcat_video.audio_process.torch_utils import save_video_ffmpeg
from audio_separator.separator import Separator

from api.inference_common import load_avatar_models


DEFAULT_NEGATIVE_PROMPT = (
    "Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards"
)


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def generate_random_uid():
    timestamp_part = str(int(time.time()))[-6:]
    random_part = str(random.randint(100000, 999999))
    return timestamp_part + random_part


def extract_vocal_from_speech(source_path, target_path, vocal_separator, audio_output_dir_temp):
    outputs = vocal_separator.separate(source_path)
    if len(outputs) <= 0:
        print("Audio separate failed. Using raw audio.")
        return None
    default_vocal_path = audio_output_dir_temp / "vocals" / outputs[0]
    default_vocal_path = default_vocal_path.resolve().as_posix()
    cmd = f"mv '{default_vocal_path}' '{target_path}'"
    os.system(cmd)
    return target_path


def init_dist(context_parallel_size: int):
    """Initialise distributed process group + context parallel (torchrun sets
    RANK / WORLD_SIZE / MASTER_* before importing this module)."""
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24))
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    context_parallel_util.init_context_parallel(
        context_parallel_size=context_parallel_size,
        global_rank=global_rank,
        world_size=num_processes,
    )
    cp_rank = context_parallel_util.get_cp_rank()
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)
    return local_rank, global_rank, num_processes, cp_rank, cp_size, cp_split_hw


def run_inference(models: dict, cfg: dict, args):
    """Reusable single-avatar inference core.

    ``models`` is the dict returned by :func:`load_avatar_models`; ``cfg`` is the
    parsed task_input json; ``args`` is any object exposing ``.output_dir``
    (a real argparse namespace for the subprocess path, or a lightweight
    SimpleNamespace built by the resident worker).
    """
    pipe = models["pipe"]
    vocal_separator = models["vocal_separator"]
    audio_output_dir_temp = models["audio_output_dir_temp"]
    model_type = models["model_type"]
    local_rank = models["local_rank"]

    prompt = cfg["prompt"]
    negative_prompt = cfg.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    raw_speech_path = cfg["cond_audio"]["person1"]
    image_path = cfg.get("cond_image")  # None for at2v
    stage_1 = cfg.get("stage_1", "ai2v")  # at2v / ai2v
    resolution = cfg.get("resolution", "480p")
    num_segments = max(1, int(cfg.get("num_segments", 1)))
    num_inference_steps = int(cfg.get("num_inference_steps", 50))
    text_guidance_scale = float(cfg.get("text_guidance_scale", 4.0))
    audio_guidance_scale = float(cfg.get("audio_guidance_scale", 4.0))
    ref_img_index = int(cfg.get("ref_img_index", 10))
    mask_frame_range = int(cfg.get("mask_frame_range", 3))
    use_distill = bool(cfg.get("use_distill", False))
    seed = int(cfg.get("seed", 42))

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if use_distill and model_type == "avatar-v1.5":
        num_inference_steps = 8
        text_guidance_scale = 1.0
        audio_guidance_scale = 1.0

    save_fps = 16
    audio_stride = 2
    if model_type == "avatar-v1.5":
        save_fps = 25
        audio_stride = 1
    num_frames = 93
    num_cond_frames = 13

    if resolution == "480p":
        height, width = 480, 832
    elif resolution == "720p":
        height, width = 768, 1280
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")

    global_rank = dist.get_rank()
    cp_rank = context_parallel_util.get_cp_rank()
    cp_size = context_parallel_util.get_cp_size()

    generator = torch.Generator(device=local_rank)
    generator.manual_seed(seed + global_rank)

    if cp_rank == 0:
        temp_vocal_path = extract_vocal_from_speech(
            raw_speech_path,
            f"/tmp/temp_speech_{generate_random_uid()}_{global_rank}_vocal.wav",
            vocal_separator, audio_output_dir_temp,
        )
        assert temp_vocal_path is not None and os.path.exists(temp_vocal_path), "No vocal detected"

        generate_duration = num_frames / save_fps + (num_segments - 1) * (num_frames - num_cond_frames) / save_fps
        speech_array, sr = librosa.load(temp_vocal_path, sr=16000)
        source_duration = len(speech_array) / sr
        added_sample_nums = math.ceil((generate_duration - source_duration) * sr)
        if added_sample_nums > 0:
            speech_array = np.append(speech_array, [0.0] * added_sample_nums)

        full_audio_emb = pipe.get_audio_embedding(
            speech_array, fps=save_fps * audio_stride, device=local_rank,
            sample_rate=sr, model_type=model_type,
        )
        if torch.isnan(full_audio_emb).any():
            raise ValueError("broken audio embedding with nan values")

        if cp_size > 1:
            full_audio_emb_shape_list = list(full_audio_emb.size())
            full_audio_emb_tensor_shape_list = torch.tensor(
                full_audio_emb_shape_list, dtype=torch.int64, device=full_audio_emb.device
            )
            context_parallel_util.cp_broadcast(full_audio_emb_tensor_shape_list)
            context_parallel_util.cp_broadcast(full_audio_emb)

        if os.path.exists(temp_vocal_path):
            os.remove(temp_vocal_path)

    elif cp_size > 1:
        full_audio_emb_tensor_shape_list = torch.zeros(3, dtype=torch.int64, device=local_rank)
        context_parallel_util.cp_broadcast(full_audio_emb_tensor_shape_list)
        full_audio_emb_shape_list = full_audio_emb_tensor_shape_list.tolist()
        full_audio_emb = torch.zeros(*full_audio_emb_shape_list, dtype=torch.float32, device=local_rank)
        context_parallel_util.cp_broadcast(full_audio_emb)

    indices = torch.arange(2 * 2 + 1) - 2
    audio_start_idx = 0
    audio_end_idx = audio_start_idx + audio_stride * num_frames

    center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
    audio_emb = full_audio_emb[center_indices][None, ...].to(local_rank)

    if local_rank == 0:
        print(f"Generating segment 1/{num_segments}...")

    if stage_1 == "at2v":
        output_tuple = pipe.generate_at2v(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            generator=generator,
            output_type="both",
            audio_emb=audio_emb,
            use_distill=use_distill,
        )
        output, latent = output_tuple
        output = output[0]
        video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        video = [PIL.Image.fromarray(img) for img in video]

        if cp_rank == 0:
            output_tensor = torch.from_numpy(np.array(video))
            save_video_ffmpeg(output_tensor, os.path.join(output_dir, "segment_1"), raw_speech_path, fps=save_fps, quality=5)
        del output
        torch_gc()

    elif stage_1 == "ai2v":
        image = load_image(image_path)
        output_tuple = pipe.generate_ai2v(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=resolution,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            output_type="both",
            generator=generator,
            audio_emb=audio_emb,
            use_distill=use_distill,
        )
        output, latent = output_tuple
        output = output[0]
        video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        video = [PIL.Image.fromarray(img) for img in video]

        if cp_rank == 0:
            output_tensor = torch.from_numpy(np.array(video))
            save_video_ffmpeg(output_tensor, os.path.join(output_dir, "segment_1"), raw_speech_path, fps=save_fps, quality=5)
        del output
        torch_gc()
    else:
        raise NotImplementedError(f"Not supported type of stage_1: {stage_1}")

    if cp_size > 1:
        torch.distributed.barrier(group=context_parallel_util.get_cp_group())

    width, height = video[0].size
    current_video = video
    ref_latent = latent[:, :, :1].clone()
    all_generated_frames = video

    for segment_idx in range(1, num_segments):
        if local_rank == 0:
            print(f"Generating segment {segment_idx+1}/{num_segments}...")

        audio_start_idx = audio_start_idx + audio_stride * (num_frames - num_cond_frames)
        audio_end_idx = audio_start_idx + audio_stride * num_frames
        center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
        audio_emb = full_audio_emb[center_indices][None, ...].to(local_rank)

        output_tuple = pipe.generate_avc(
            video=current_video,
            video_latent=latent,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_cond_frames=num_cond_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            generator=generator,
            output_type="both",
            use_kv_cache=True,
            offload_kv_cache=False,
            enhance_hf=True if not use_distill else False,
            audio_emb=audio_emb,
            ref_latent=ref_latent,
            ref_img_index=ref_img_index,
            mask_frame_range=mask_frame_range,
            use_distill=use_distill,
        )
        output, latent = output_tuple
        output = output[0]
        new_video = [(output[i] * 255).astype(np.uint8) for i in range(output.shape[0])]
        new_video = [PIL.Image.fromarray(img) for img in new_video]
        del output

        all_generated_frames.extend(new_video[num_cond_frames:])
        current_video = new_video

        if cp_rank == 0:
            output_tensor = torch.from_numpy(np.array(all_generated_frames))
            save_video_ffmpeg(output_tensor, os.path.join(output_dir, f"video_continue_{segment_idx+1}"), raw_speech_path, fps=save_fps, quality=5)
            del output_tensor

    # final merged output (segment_1)
    if cp_rank == 0 and num_segments == 1:
        final_src = os.path.join(output_dir, "segment_1.mp4")
        if os.path.exists(final_src):
            os.symlink(os.path.basename(final_src), os.path.join(output_dir, "output.mp4"))


def generate(args):
    """Subprocess entry point (used when LONGCAT_INPROCESS=0). Loads the model,
    runs one task, then exits."""
    with open(args.task_input, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    local_rank, global_rank, num_processes, cp_rank, cp_size, cp_split_hw = init_dist(args.context_parallel_size)

    models = load_avatar_models(
        checkpoint_dir=args.checkpoint_dir,
        model_type=cfg.get("model_type", "avatar-v1.0"),
        use_int8=bool(cfg.get("use_int8", False)),
        use_distill=bool(cfg.get("use_distill", False)),
        local_rank=local_rank,
        cp_split_hw=cp_split_hw,
        avatar_mode="single",
    )
    run_inference(models, cfg, args)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--context_parallel_size", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
