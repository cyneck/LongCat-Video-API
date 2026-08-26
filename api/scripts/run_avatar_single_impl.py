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

# Avoid tokenizers' fork warning when ffmpeg is spawned after prompt tokenization.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

import subprocess
try:
    from api import config
except Exception:
    import config


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


def _round_up_vae_frames(frame_count: int, max_frames: int = 93) -> int:
    """Round up to the VAE temporal constraint: (frames - 1) % 4 == 0."""
    frame_count = max(1, min(int(frame_count), max_frames))
    rounded = 1 + math.ceil((frame_count - 1) / 4) * 4
    return min(max_frames, rounded)


def _auto_frame_plan(duration: float, fps: int, max_frames: int, cond_frames: int) -> list[int]:
    """Plan only the frames needed to cover the source audio.

    Full 93-frame clips are retained for all intermediate segments. Only a
    short single segment or the final continuation is reduced, so the long-video
    conditioning topology remains unchanged while avoiding useless denoising.
    """
    target_frames = max(1, math.ceil(duration * fps))
    if target_frames <= max_frames:
        return [_round_up_vae_frames(target_frames, max_frames)]

    plan = [max_frames]
    covered = max_frames
    max_new_frames = max_frames - cond_frames
    while covered < target_frames:
        remaining = target_frames - covered
        new_frames = min(max_new_frames, remaining)
        segment_frames = _round_up_vae_frames(cond_frames + new_frames, max_frames)
        segment_frames = max(segment_frames, cond_frames + 4)
        segment_frames = _round_up_vae_frames(segment_frames, max_frames)
        plan.append(segment_frames)
        covered += segment_frames - cond_frames
    return plan


def _generated_frame_count(frame_plan: list[int], cond_frames: int) -> int:
    if not frame_plan:
        return 0
    return frame_plan[0] + sum(frames - cond_frames for frames in frame_plan[1:])


def _log_timing(name: str, started_at: float, rank: int = 0) -> float:
    elapsed = time.perf_counter() - started_at
    if rank == 0:
        print(f"[longcat][timing] {name}={elapsed:.2f}s", flush=True)
    return elapsed


def _log_onnx_provider_status(rank: int = 0):
    if rank != 0:
        return
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"[longcat][onnx] providers={providers}", flush=True)
        if "CUDAExecutionProvider" not in providers:
            print(
                "[longcat][onnx] CUDAExecutionProvider 不可用，vocal separator 将使用 CPU；"
                "若部署环境兼容，可安装匹配 CUDA 的 onnxruntime-gpu 以缩短音频预处理时间",
                flush=True,
            )
    except Exception as exc:
        print(f"[longcat][onnx] provider detection failed: {exc}", flush=True)


def _concat_and_mux(segment_paths, output_base, audio_path):
    list_path = output_base + ".concat.txt"
    merged = output_base + ".merged.mp4"
    final = output_base + ".mp4"
    with open(list_path, "w", encoding="utf-8") as f:
        for p in segment_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", merged],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", merged, "-i", audio_path,
             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
             "-map", "0:v:0", "-map", "1:a:0", "-shortest", final],
            check=True,
        )
    finally:
        for p in (list_path, merged):
            if os.path.exists(p):
                os.remove(p)
    return final


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


def generate(args):
    total_started = time.perf_counter()
    timings = {}

    with open(args.task_input, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    prompt = cfg["prompt"]
    negative_prompt = cfg.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    raw_speech_path = cfg["cond_audio"]["person1"]
    image_path = cfg.get("cond_image")
    stage_1 = cfg.get("stage_1", "ai2v")
    resolution = cfg.get("resolution", "480p")
    num_segments_raw = cfg.get("num_segments", "auto")
    print(f"[longcat][seg] 收到 num_segments={num_segments_raw!r}（'auto'=按原始上传音频时长自适应）", flush=True)
    num_inference_steps = int(cfg.get("num_inference_steps", 50))
    text_guidance_scale = float(cfg.get("text_guidance_scale", 4.0))
    audio_guidance_scale = float(cfg.get("audio_guidance_scale", 4.0))
    audio_drive_gain = float(cfg.get("audio_drive_gain", 0.85))
    if not 0.5 <= audio_drive_gain <= 1.2:
        raise ValueError(f"audio_drive_gain must be in [0.5, 1.2], got {audio_drive_gain}")
    ref_img_index = int(cfg.get("ref_img_index", 10))
    mask_frame_range = int(cfg.get("mask_frame_range", 3))
    model_type = cfg.get("model_type", "avatar-v1.0")
    use_distill = bool(cfg.get("use_distill", False))
    use_int8 = bool(cfg.get("use_int8", False))
    seed = int(cfg.get("seed", 42))

    checkpoint_dir = args.checkpoint_dir
    context_parallel_size = args.context_parallel_size
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    segment_paths = []

    if use_distill and model_type == "avatar-v1.5":
        num_inference_steps = 8
        text_guidance_scale = 1.0
        audio_guidance_scale = 1.0

    save_fps = 16
    audio_stride = 2
    if model_type == "avatar-v1.5":
        save_fps = 25
        audio_stride = 1
    max_num_frames = 93
    num_cond_frames = 13

    if resolution == "480p":
        height, width = 480, 832
    elif resolution == "720p":
        height, width = 768, 1280
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")

    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24))
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()

    context_parallel_util.init_context_parallel(context_parallel_size=context_parallel_size, global_rank=global_rank, world_size=num_processes)
    cp_rank = context_parallel_util.get_cp_rank()
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    model_started = time.perf_counter()
    _fallback = os.path.normpath(os.path.join(checkpoint_dir, "..", "LongCat-Video"))
    base_model_dir = os.environ.get("LONGCAT_CHECKPOINT_DIR_VIDEO") or _fallback
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16)
    text_encoder = UMT5EncoderModel.from_pretrained(base_model_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(base_model_dir, subfolder="vae", torch_dtype=torch.bfloat16)
    if model_type == "avatar-v1.0":
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base_model_dir, subfolder="scheduler", torch_dtype=torch.bfloat16)
    elif model_type == "avatar-v1.5":
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Expected 'avatar-v1.0' or 'avatar-v1.5'.")

    if model_type == "avatar-v1.0":
        dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="avatar_single", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16)
    elif model_type == "avatar-v1.5":
        if use_int8:
            print("[INFO] Loading INT8 quantized DiT model...")
            dit = load_quantized_dit(checkpoint_dir, subfolder="base_model_int8", cp_split_hw=cp_split_hw)
        else:
            dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="base_model", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16)
        if use_distill:
            distill_checkpoint_path = os.path.join(checkpoint_dir, "lora", "dmd_lora.safetensors")
            if os.path.exists(distill_checkpoint_path):
                dit.load_lora(distill_checkpoint_path, "dmd", multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)
                dit.enable_loras(["dmd"])
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Expected 'avatar-v1.0' or 'avatar-v1.5'.")

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
    _log_onnx_provider_status(global_rank)
    separator_init_started = time.perf_counter()
    vocal_separator = Separator(
        output_dir=audio_output_dir_temp / "vocals",
        output_single_stem="vocals",
        model_file_dir=audio_separator_model_path,
    )
    vocal_separator.load_model(audio_separator_model_name)
    timings["separator_init"] = _log_timing("separator_init", separator_init_started, global_rank)

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
    timings["model_load"] = _log_timing("model_load", model_started, global_rank)

    original_encode_prompt = pipe.encode_prompt
    prompt_cache = {}

    def cached_encode_prompt(*encode_args, **encode_kwargs):
        cache_key = (
            repr(encode_kwargs.get("prompt")),
            repr(encode_kwargs.get("negative_prompt")),
            bool(encode_kwargs.get("do_classifier_free_guidance", True)),
            int(encode_kwargs.get("num_videos_per_prompt", 1)),
            int(encode_kwargs.get("max_sequence_length", 512)),
            str(encode_kwargs.get("dtype")),
            str(encode_kwargs.get("device")),
        )
        cached = prompt_cache.get(cache_key)
        if cached is not None:
            print("[longcat][prompt] reuse cached prompt embeddings", flush=True)
            return cached
        encoded = original_encode_prompt(*encode_args, **encode_kwargs)
        prompt_cache[cache_key] = tuple(
            tensor.detach() if isinstance(tensor, torch.Tensor) else tensor for tensor in encoded
        )
        return prompt_cache[cache_key]

    pipe.encode_prompt = cached_encode_prompt

    generator = torch.Generator(device=local_rank)
    generator.manual_seed(seed + global_rank)

    frame_plan = [max_num_frames]
    if cp_rank == 0:
        raw_duration = float(librosa.get_duration(path=raw_speech_path))
        if not math.isfinite(raw_duration) or raw_duration <= 0:
            raise ValueError(f"Invalid input audio duration: {raw_duration}")

        separation_started = time.perf_counter()
        temp_vocal_path = extract_vocal_from_speech(
            raw_speech_path,
            f"/tmp/temp_speech_{generate_random_uid()}_{global_rank}_vocal.wav",
            vocal_separator,
            audio_output_dir_temp,
        )
        timings["vocal_separation"] = _log_timing("vocal_separation", separation_started, global_rank)
        assert temp_vocal_path is not None and os.path.exists(temp_vocal_path), "No vocal detected"

        speech_array, sr = librosa.load(temp_vocal_path, sr=16000)
        vocal_duration = len(speech_array) / sr

        if isinstance(num_segments_raw, str) and num_segments_raw.strip().lower() == "auto":
            frame_plan = _auto_frame_plan(raw_duration, save_fps, max_num_frames, num_cond_frames)
            _max_seg = config.LOW_VRAM.get("max_num_segments", 0)
            if _max_seg and _max_seg > 0 and len(frame_plan) > _max_seg:
                print(
                    f"[longcat][auto] 计算得到 {len(frame_plan)} 段，但 profile max_num_segments={_max_seg}，"
                    f"将封顶为 {_max_seg}；若希望完整覆盖音频，请将 max_num_segments 设为 0",
                    flush=True,
                )
                frame_plan = frame_plan[:_max_seg]
            num_segments = len(frame_plan)
            generated_frames = _generated_frame_count(frame_plan, num_cond_frames)
            print(
                f"[longcat][auto] 原始音频 {raw_duration:.2f}s（分离后人声 {vocal_duration:.2f}s）"
                f" → frame_plan={frame_plan}，有效输出 {generated_frames} 帧/{generated_frames/save_fps:.2f}s",
                flush=True,
            )
        else:
            num_segments = max(1, int(num_segments_raw))
            frame_plan = [max_num_frames] * num_segments
            print(f"[longcat][seg] 使用固定段数 num_segments={num_segments}，frame_plan={frame_plan}", flush=True)

        generate_duration = _generated_frame_count(frame_plan, num_cond_frames) / save_fps
        added_sample_nums = math.ceil((generate_duration - vocal_duration) * sr)
        if added_sample_nums > 0:
            speech_array = np.append(speech_array, np.zeros(added_sample_nums, dtype=speech_array.dtype))

        audio_embedding_started = time.perf_counter()
        full_audio_emb = pipe.get_audio_embedding(
            speech_array,
            fps=save_fps * audio_stride,
            device=local_rank,
            sample_rate=sr,
            model_type=model_type,
        )
        full_audio_emb = full_audio_emb * audio_drive_gain
        print(
            f"[longcat][audio] audio_drive_gain={audio_drive_gain:.2f} applied to audio embedding "
            "(output audio volume unchanged)",
            flush=True,
        )
        timings["audio_embedding"] = _log_timing("audio_embedding", audio_embedding_started, global_rank)
        if torch.isnan(full_audio_emb).any():
            raise ValueError("broken audio embedding with nan values")

        if context_parallel_util.get_cp_size() > 1:
            full_audio_emb_shape_list = list(full_audio_emb.size())
            full_audio_emb_tensor_shape_list = torch.tensor(full_audio_emb_shape_list, dtype=torch.int64, device=full_audio_emb.device)
            context_parallel_util.cp_broadcast(full_audio_emb_tensor_shape_list)
            context_parallel_util.cp_broadcast(full_audio_emb)

        if os.path.exists(temp_vocal_path):
            os.remove(temp_vocal_path)

    elif context_parallel_util.get_cp_size() > 1:
        full_audio_emb_tensor_shape_list = torch.zeros(3, dtype=torch.int64, device=local_rank)
        context_parallel_util.cp_broadcast(full_audio_emb_tensor_shape_list)
        full_audio_emb_shape_list = full_audio_emb_tensor_shape_list.tolist()
        full_audio_emb = torch.zeros(*full_audio_emb_shape_list, dtype=torch.float32, device=local_rank)
        context_parallel_util.cp_broadcast(full_audio_emb)

    if cp_rank != 0:
        num_segments = max(1, int(num_segments_raw)) if not isinstance(num_segments_raw, str) else 1
        frame_plan = [max_num_frames] * num_segments

    pipe.audio_encoder = pipe.audio_encoder.to("cpu")
    torch.cuda.empty_cache()

    indices = torch.arange(2 * 2 + 1) - 2
    audio_start_idx = 0
    first_num_frames = frame_plan[0]
    audio_end_idx = audio_start_idx + audio_stride * first_num_frames

    center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
    center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
    audio_emb = full_audio_emb[center_indices][None, ...].to(local_rank)

    if local_rank == 0:
        print(f"Generating segment 1/{num_segments} ({first_num_frames} frames)...")
    segment_started = time.perf_counter()

    if stage_1 == "at2v":
        output_tuple = pipe.generate_at2v(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=first_num_frames,
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
            seg_path = os.path.join(output_dir, "segment_1")
            save_video_ffmpeg(torch.from_numpy(np.array(video)), seg_path, audio_path=None, fps=save_fps, quality=5)
            segment_paths.append(seg_path + ".mp4")
        del output
        torch_gc()

    elif stage_1 == "ai2v":
        image = load_image(image_path)
        output_tuple = pipe.generate_ai2v(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            resolution=resolution,
            num_frames=first_num_frames,
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
            seg_path = os.path.join(output_dir, "segment_1")
            save_video_ffmpeg(torch.from_numpy(np.array(video)), seg_path, audio_path=None, fps=save_fps, quality=5)
            segment_paths.append(seg_path + ".mp4")
        del output
        torch_gc()
    else:
        raise NotImplementedError(f"Not supported type of stage_1: {stage_1}")

    timings["segment_1"] = _log_timing("segment_1", segment_started, global_rank)

    if context_parallel_util.get_cp_size() > 1:
        torch.distributed.barrier(group=context_parallel_util.get_cp_group())

    width, height = video[0].size
    current_video = video
    ref_latent = latent[:, :, :1].clone()

    for segment_idx in range(1, num_segments):
        current_num_frames = frame_plan[segment_idx]
        previous_num_frames = frame_plan[segment_idx - 1]
        if local_rank == 0:
            print(f"Generating segment {segment_idx+1}/{num_segments} ({current_num_frames} frames)...")

        audio_start_idx = audio_start_idx + audio_stride * (previous_num_frames - num_cond_frames)
        audio_end_idx = audio_start_idx + audio_stride * current_num_frames
        center_indices = torch.arange(audio_start_idx, audio_end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1)
        audio_emb = full_audio_emb[center_indices][None, ...].to(local_rank)

        segment_started = time.perf_counter()
        output_tuple = pipe.generate_avc(
            video=current_video,
            video_latent=latent,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=current_num_frames,
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
        current_video = new_video

        if cp_rank == 0:
            new_frames = new_video[num_cond_frames:]
            if not new_frames:
                raise RuntimeError(
                    f"Continuation segment {segment_idx+1} produced no new frames after trimming {num_cond_frames} conditioning frames"
                )
            seg_path = os.path.join(output_dir, f"video_continue_{segment_idx+1}")
            save_video_ffmpeg(
                torch.from_numpy(np.array(new_frames)), seg_path, audio_path=None, fps=save_fps, quality=5
            )
            segment_paths.append(seg_path + ".mp4")
        torch_gc()
        timings[f"segment_{segment_idx+1}"] = _log_timing(f"segment_{segment_idx+1}", segment_started, global_rank)

    if cp_rank == 0 and segment_paths:
        mux_started = time.perf_counter()
        _concat_and_mux(segment_paths, os.path.join(output_dir, "output"), raw_speech_path)
        timings["mux"] = _log_timing("mux", mux_started, global_rank)

    timings["total"] = _log_timing("total", total_started, global_rank)
    if global_rank == 0:
        summary = ", ".join(f"{name}={seconds:.2f}s" for name, seconds in timings.items())
        print(f"[longcat][timing] summary: {summary}", flush=True)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--context_parallel_size", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    try:
        generate(args)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
