import os
import gc
import json
import time
import math
import random
import argparse
import datetime
import subprocess
from pathlib import Path

import PIL.Image
import numpy as np
import torch
import torch.distributed as dist
import librosa
import soundfile as sf

from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers.utils import load_image

from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from longcat_video.modules.quantization import load_quantized_dit
from longcat_video.context_parallel import context_parallel_util
from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor
from longcat_video.audio_process.torch_utils import save_video_ffmpeg
from audio_separator.separator import Separator

try:
    from api import config
except Exception:
    import config


DEFAULT_NEGATIVE_PROMPT = (
    "Close-up, bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards"
)


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _round_up_vae_frames(frame_count: int, max_frames: int = 93) -> int:
    frame_count = max(1, min(int(frame_count), max_frames))
    rounded = 1 + math.ceil((frame_count - 1) / 4) * 4
    return min(max_frames, rounded)


def _auto_frame_plan(duration: float, fps: int, max_frames: int, cond_frames: int) -> list[int]:
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


def _log_memory(name: str, rank: int = 0):
    if rank != 0:
        return
    rss_gb = None
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_gb = int(line.split()[1]) / 1024 / 1024
                    break
    except Exception:
        pass

    gpu = ""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        gpu = f" gpu_alloc={allocated:.2f}GB gpu_reserved={reserved:.2f}GB gpu_peak={peak:.2f}GB"
    rss = f" rss={rss_gb:.2f}GB" if rss_gb is not None else ""
    print(f"[longcat][memory] {name}:{rss}{gpu}", flush=True)


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
                "若环境兼容，可安装匹配 CUDA 的 onnxruntime-gpu",
                flush=True,
            )
    except Exception as exc:
        print(f"[longcat][onnx] provider 检测失败: {exc}", flush=True)


def _concat_and_mux(segment_paths, output_base, audio_path):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    list_path = output_base + ".concat.txt"
    merged = output_base + ".merged.mp4"
    final = output_base + ".mp4"
    with open(list_path, "w", encoding="utf-8") as f:
        for path in segment_paths:
            escaped = os.path.abspath(path).replace("'", r"'\''")
            f.write(f"file '{escaped}'\n")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", merged],
            check=True,
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", merged, "-i", audio_path,
                "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v:0", "-map", "1:a:0", "-shortest", final,
            ],
            check=True,
        )
    finally:
        for path in (list_path, merged):
            if os.path.exists(path):
                os.remove(path)
    return final


def generate_random_uid():
    timestamp_part = str(int(time.time()))[-6:]
    random_part = str(random.randint(100000, 999999))
    return timestamp_part + random_part


def extract_vocal_from_speech(source_path, target_path, vocal_separator, audio_output_dir_temp):
    if source_path is None:
        return None
    outputs = vocal_separator.separate(source_path)
    if not outputs:
        print("Audio separate failed. Using raw audio.", flush=True)
        return None
    default_vocal_path = (audio_output_dir_temp / "vocals" / outputs[0]).resolve().as_posix()
    os.replace(default_vocal_path, target_path)
    return target_path


def _load_audio_or_none(path, sample_rate):
    if path is None:
        return None
    data, _ = librosa.load(path, sr=sample_rate)
    return np.asarray(data, dtype=np.float32)


def _pad_to_length(data, length):
    if data is None:
        return np.zeros(length, dtype=np.float32)
    if len(data) >= length:
        return data[:length]
    return np.pad(data, (0, length - len(data)))


def audio_prepare_multi(
    left_temp_vocal_path,
    right_temp_vocal_path,
    generate_duration,
    left_raw_speech_path,
    right_raw_speech_path,
    sample_rate=16000,
    audio_type="para",
):
    left_vocal = _load_audio_or_none(left_temp_vocal_path, sample_rate)
    right_vocal = _load_audio_or_none(right_temp_vocal_path, sample_rate)
    left_raw = _load_audio_or_none(left_raw_speech_path, sample_rate)
    right_raw = _load_audio_or_none(right_raw_speech_path, sample_rate)

    if left_vocal is None and right_vocal is None:
        raise ValueError("No valid vocal audio found")

    if audio_type == "para":
        target_len = max(
            len(left_vocal) if left_vocal is not None else 0,
            len(right_vocal) if right_vocal is not None else 0,
            len(left_raw) if left_raw is not None else 0,
            len(right_raw) if right_raw is not None else 0,
        )
        left_drive = _pad_to_length(left_vocal, target_len)
        right_drive = _pad_to_length(right_vocal, target_len)
        merge_raw = _pad_to_length(left_raw, target_len) + _pad_to_length(right_raw, target_len)
    elif audio_type == "add":
        left_len = max(
            len(left_vocal) if left_vocal is not None else 0,
            len(left_raw) if left_raw is not None else 0,
        )
        right_len = max(
            len(right_vocal) if right_vocal is not None else 0,
            len(right_raw) if right_raw is not None else 0,
        )
        left_vocal_p = _pad_to_length(left_vocal, left_len)
        right_vocal_p = _pad_to_length(right_vocal, right_len)
        left_raw_p = _pad_to_length(left_raw, left_len)
        right_raw_p = _pad_to_length(right_raw, right_len)
        left_drive = np.concatenate([left_vocal_p, np.zeros(right_len, dtype=np.float32)])
        right_drive = np.concatenate([np.zeros(left_len, dtype=np.float32), right_vocal_p])
        merge_raw = np.concatenate([left_raw_p, right_raw_p])
    else:
        raise ValueError(f"Unsupported audio_type: {audio_type}")

    needed_len = math.ceil(generate_duration * sample_rate)
    if len(left_drive) < needed_len:
        left_drive = np.pad(left_drive, (0, needed_len - len(left_drive)))
        right_drive = np.pad(right_drive, (0, needed_len - len(right_drive)))

    return (
        np.asarray(left_drive, dtype=np.float32),
        np.asarray(right_drive, dtype=np.float32),
        np.asarray(merge_raw, dtype=np.float32),
    )


def _multi_source_duration(left_path, right_path, audio_type):
    left = float(librosa.get_duration(path=left_path)) if left_path else 0.0
    right = float(librosa.get_duration(path=right_path)) if right_path else 0.0
    duration = left + right if audio_type == "add" else max(left, right)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"Invalid multi-speaker audio duration: left={left}, right={right}")
    return duration, left, right


def generate(args):
    total_started = time.perf_counter()
    timings = {}

    with open(args.task_input, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    prompt = input_data["prompt"]
    negative_prompt = input_data.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    left_raw_speech_path = input_data["cond_audio"].get("person1")
    right_raw_speech_path = input_data["cond_audio"].get("person2")
    if left_raw_speech_path is None and right_raw_speech_path is None:
        raise ValueError("At least one speech is required")
    image_path = input_data["cond_image"]
    audio_type = input_data.get("audio_type", "para")
    resolution = input_data.get("resolution", "480p")
    num_segments_raw = input_data.get("num_segments", "auto")
    num_inference_steps = int(input_data.get("num_inference_steps", 50))
    text_guidance_scale = float(input_data.get("text_guidance_scale", 4.0))
    audio_guidance_scale = float(input_data.get("audio_guidance_scale", 4.0))
    audio_drive_gain = float(input_data.get("audio_drive_gain", 0.85))
    if not 0.5 <= audio_drive_gain <= 1.2:
        raise ValueError(f"audio_drive_gain must be in [0.5, 1.2], got {audio_drive_gain}")
    ref_img_index = int(input_data.get("ref_img_index", 10))
    mask_frame_range = int(input_data.get("mask_frame_range", 3))
    model_type = input_data.get("model_type", "avatar-v1.0")
    use_distill = bool(input_data.get("use_distill", False))
    use_int8 = bool(input_data.get("use_int8", False))
    seed = int(input_data.get("seed", 42))

    left_person_bbox = right_person_bbox = other_person_bbox = None
    use_background_silent_audio = False
    if input_data.get("bbox"):
        left_person_bbox = input_data["bbox"].get("person1")
        right_person_bbox = input_data["bbox"].get("person2")
        other_person_bbox = input_data["bbox"].get("others")
        use_background_silent_audio = bool(other_person_bbox)

    checkpoint_dir = args.checkpoint_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    segment_paths = []

    if use_distill and model_type == "avatar-v1.5":
        num_inference_steps = 8
        text_guidance_scale = 1.0
        audio_guidance_scale = 1.0

    save_fps, audio_stride = (25, 1) if model_type == "avatar-v1.5" else (16, 2)
    max_num_frames = 93
    num_cond_frames = 13

    if resolution == "480p":
        height, width = 480, 832
    elif resolution == "720p":
        height, width = 768, 1280
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")

    rank = int(os.environ["RANK"])
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24))
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()

    context_parallel_util.init_context_parallel(
        context_parallel_size=args.context_parallel_size,
        global_rank=global_rank,
        world_size=num_processes,
    )
    cp_rank = context_parallel_util.get_cp_rank()
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    _log_memory("start", global_rank)
    _log_onnx_provider_status(global_rank)

    model_started = time.perf_counter()
    base_model_dir = os.environ.get("LONGCAT_CHECKPOINT_DIR_VIDEO") or os.path.normpath(
        os.path.join(checkpoint_dir, "..", "LongCat-Video")
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, subfolder="tokenizer")
    text_encoder = UMT5EncoderModel.from_pretrained(
        base_model_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16
    )
    vae = AutoencoderKLWan.from_pretrained(
        base_model_dir, subfolder="vae", torch_dtype=torch.bfloat16
    )
    scheduler_dir = base_model_dir if model_type == "avatar-v1.0" else checkpoint_dir
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        scheduler_dir, subfolder="scheduler", torch_dtype=torch.bfloat16
    )

    if model_type == "avatar-v1.0":
        dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(
            checkpoint_dir,
            subfolder="avatar_multi",
            cp_split_hw=cp_split_hw,
            torch_dtype=torch.bfloat16,
        )
    elif model_type == "avatar-v1.5":
        if use_int8:
            print("[INFO] Loading INT8 quantized DiT model...", flush=True)
            dit = load_quantized_dit(checkpoint_dir, subfolder="base_model_int8", cp_split_hw=cp_split_hw)
        else:
            dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(
                checkpoint_dir,
                subfolder="base_model",
                cp_split_hw=cp_split_hw,
                torch_dtype=torch.bfloat16,
            )
        if use_distill:
            lora_path = os.path.join(checkpoint_dir, "lora", "dmd_lora.safetensors")
            if os.path.exists(lora_path):
                dit.load_lora(
                    lora_path, "dmd", multiplier=1.0,
                    lora_network_dim=128, lora_network_alpha=64
                )
                dit.enable_loras(["dmd"])
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    audio_model_checkpoint_path = os.path.join(
        checkpoint_dir,
        "whisper-large-v3" if model_type == "avatar-v1.5" else "chinese-wav2vec2-base",
    )
    audio_encoder = get_audio_encoder(audio_model_checkpoint_path, model_type).to(local_rank)
    audio_feature_extractor = get_audio_feature_extractor(audio_model_checkpoint_path, model_type)

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
    _log_memory("after_model_load", global_rank)

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

    source_duration, left_duration, right_duration = _multi_source_duration(
        left_raw_speech_path, right_raw_speech_path, audio_type
    )
    if isinstance(num_segments_raw, str) and num_segments_raw.strip().lower() == "auto":
        frame_plan = _auto_frame_plan(source_duration, save_fps, max_num_frames, num_cond_frames)
        max_segments = config.LOW_VRAM.get("max_num_segments", 0)
        if max_segments and max_segments > 0:
            frame_plan = frame_plan[:max_segments]
    else:
        num_segments = max(1, int(num_segments_raw))
        frame_plan = [max_num_frames] * num_segments
    num_segments = len(frame_plan)
    generated_frames = _generated_frame_count(frame_plan, num_cond_frames)
    generate_duration = generated_frames / save_fps
    if global_rank == 0:
        print(
            f"[longcat][auto] multi audio_type={audio_type} left={left_duration:.2f}s right={right_duration:.2f}s "
            f"source={source_duration:.2f}s frame_plan={frame_plan} output={generated_frames}f/{generate_duration:.2f}s",
            flush=True,
        )

    separator_started = time.perf_counter()
    vocal_separator_path = os.path.join(checkpoint_dir, "vocal_separator", "Kim_Vocal_2.onnx")
    audio_output_dir_temp = Path("./audio_temp_file")
    os.makedirs(audio_output_dir_temp / "vocals", exist_ok=True)
    vocal_separator = Separator(
        output_dir=audio_output_dir_temp / "vocals",
        output_single_stem="vocals",
        model_file_dir=os.path.dirname(vocal_separator_path),
    )
    vocal_separator.load_model(os.path.basename(vocal_separator_path))
    timings["separator_init"] = _log_timing("separator_init", separator_started, global_rank)

    merge_speech_path = None
    if cp_rank == 0:
        separation_started = time.perf_counter()
        left_temp = os.path.join(audio_output_dir_temp, f"{generate_random_uid()}_left.wav")
        right_temp = os.path.join(audio_output_dir_temp, f"{generate_random_uid()}_right.wav")
        left_temp = extract_vocal_from_speech(left_raw_speech_path, left_temp, vocal_separator, audio_output_dir_temp)
        right_temp = extract_vocal_from_speech(right_raw_speech_path, right_temp, vocal_separator, audio_output_dir_temp)
        timings["vocal_separation"] = _log_timing("vocal_separation", separation_started, global_rank)

        audio_started = time.perf_counter()
        left_drive, right_drive, merge_speech = audio_prepare_multi(
            left_temp,
            right_temp,
            generate_duration,
            left_raw_speech_path,
            right_raw_speech_path,
            sample_rate=16000,
            audio_type=audio_type,
        )
        merge_speech_path = f"/tmp/longcat_multi_{generate_random_uid()}_{global_rank}.wav"
        sf.write(merge_speech_path, merge_speech, 16000)

        left_full_audio_emb = pipe.get_audio_embedding(
            left_drive, fps=save_fps * audio_stride, device=local_rank,
            sample_rate=16000, model_type=model_type
        ).detach().cpu() * audio_drive_gain
        right_full_audio_emb = pipe.get_audio_embedding(
            right_drive, fps=save_fps * audio_stride, device=local_rank,
            sample_rate=16000, model_type=model_type
        ).detach().cpu() * audio_drive_gain
        if use_background_silent_audio:
            back_full_audio_emb = pipe.get_audio_embedding(
                np.zeros_like(left_drive), fps=save_fps * audio_stride, device=local_rank,
                sample_rate=16000, model_type=model_type
            ).detach().cpu()
        print(
            f"[longcat][audio] audio_drive_gain={audio_drive_gain:.2f} applied to speaker embeddings "
            "(background silence/output audio unchanged)",
            flush=True,
        )
        if torch.isnan(left_full_audio_emb).any() or torch.isnan(right_full_audio_emb).any():
            raise ValueError("broken audio embedding with nan values")
        timings["audio_embedding"] = _log_timing("audio_embedding", audio_started, global_rank)

        for temp_path in (left_temp, right_temp):
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

        del left_drive, right_drive, merge_speech
        gc.collect()

    elif cp_size > 1:
        raise NotImplementedError("Optimized avatar-multi path currently targets single-GPU cp_size=1")

    del vocal_separator
    pipe.audio_encoder = None
    pipe.audio_feature_extractor = None
    audio_encoder = None
    audio_feature_extractor = None
    gc.collect()
    torch_gc()
    _log_memory("after_audio_release", global_rank)

    indices = torch.arange(5) - 2

    def build_audio_window(start_idx, frame_count):
        end_idx = start_idx + audio_stride * frame_count
        center = torch.arange(start_idx, end_idx, audio_stride).unsqueeze(1) + indices.unsqueeze(0)
        center = torch.clamp(center, min=0, max=left_full_audio_emb.shape[0] - 1)
        chunks = [
            left_full_audio_emb[center][None, ...].to(local_rank),
            right_full_audio_emb[center][None, ...].to(local_rank),
        ]
        if use_background_silent_audio:
            chunks.append(back_full_audio_emb[center][None, ...].to(local_rank))
        return torch.cat(chunks), end_idx

    image = load_image(image_path)
    src_width, src_height = image.size

    background_mask = torch.zeros((src_height, src_width), dtype=torch.float32)
    human_mask1 = torch.zeros((src_height, src_width), dtype=torch.float32)
    human_mask2 = torch.zeros((src_height, src_width), dtype=torch.float32)

    if left_person_bbox is None and right_person_bbox is None:
        face_scale = 0.1
        y_min, y_max = int(src_height * face_scale), int(src_height * (1 - face_scale))
        half_width = src_width // 2
        left_y_min, left_y_max = y_min, y_max
        right_y_min, right_y_max = y_min, y_max
        left_x_min, left_x_max = int(half_width * face_scale), int(half_width * (1 - face_scale))
        right_x_min = int(half_width * face_scale + half_width)
        right_x_max = int(half_width * (1 - face_scale) + half_width)
    elif left_person_bbox is not None and right_person_bbox is not None:
        left_y_min, left_x_min, left_y_max, left_x_max = left_person_bbox
        right_y_min, right_x_min, right_y_max, right_x_max = right_person_bbox
    else:
        raise ValueError("person1/person2 bbox must either both be supplied or both omitted")

    human_mask1[left_y_min:left_y_max, left_x_min:left_x_max] = 1
    human_mask2[right_y_min:right_y_max, right_x_min:right_x_max] = 1
    background_mask = 1 - torch.clamp(human_mask1 + human_mask2, 0, 1)
    total_mask = [human_mask1, human_mask2, background_mask]
    if use_background_silent_audio and other_person_bbox:
        for i in range(len(other_person_bbox) // 4):
            y1, x1, y2, x2 = other_person_bbox[i * 4:(i + 1) * 4]
            other_mask = torch.zeros((src_height, src_width), dtype=torch.float32)
            other_mask[y1:y2, x1:x2] = 1
            total_mask.append(other_mask)
    ref_target_masks = torch.stack(total_mask, dim=0).to(local_rank)
    del background_mask, human_mask1, human_mask2, total_mask
    gc.collect()

    first_frames = frame_plan[0]
    audio_start_idx = 0
    audio_embs, _ = build_audio_window(audio_start_idx, first_frames)

    if global_rank == 0:
        print(f"Generating segment 1/{num_segments} ({first_frames} frames)...", flush=True)
    segment_started = time.perf_counter()
    output, latent = pipe.generate_ai2v(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        resolution=resolution,
        num_frames=first_frames,
        num_inference_steps=num_inference_steps,
        text_guidance_scale=text_guidance_scale,
        audio_guidance_scale=audio_guidance_scale,
        output_type="both",
        generator=generator,
        audio_emb=audio_embs,
        ref_target_masks=ref_target_masks,
        use_distill=use_distill,
    )
    output = output[0]
    video = [PIL.Image.fromarray((output[i] * 255).astype(np.uint8)) for i in range(output.shape[0])]
    del output, audio_embs
    torch_gc()

    if cp_rank == 0:
        seg_path = os.path.join(output_dir, "segment_1")
        save_video_ffmpeg(
            torch.from_numpy(np.array(video)), seg_path, audio_path=None, fps=save_fps, quality=5
        )
        segment_paths.append(seg_path + ".mp4")
    timings["segment_1"] = _log_timing("segment_1", segment_started, global_rank)
    _log_memory("after_segment_1", global_rank)

    actual_width, actual_height = video[0].size
    latent_height = int(latent.shape[-2] * pipe.vae_scale_factor_spatial)
    latent_width = int(latent.shape[-1] * pipe.vae_scale_factor_spatial)
    if (actual_height, actual_width) != (latent_height, latent_width):
        raise RuntimeError(
            f"AI2V output/latent spatial mismatch: video={actual_width}x{actual_height}, latent-derived={latent_width}x{latent_height}"
        )
    height, width = actual_height, actual_width
    if global_rank == 0:
        print(
            f"[longcat][shape] continuation size={width}x{height}, latent={latent.shape[-1]}x{latent.shape[-2]}",
            flush=True,
        )

    if cp_size == 1 and prompt_cache:
        pipe.text_encoder = None
        text_encoder = None
        gc.collect()

    current_video = video
    ref_latent = latent[:, :, :1].clone()

    for segment_idx in range(1, num_segments):
        current_frames = frame_plan[segment_idx]
        previous_frames = frame_plan[segment_idx - 1]
        audio_start_idx += audio_stride * (previous_frames - num_cond_frames)
        audio_embs, _ = build_audio_window(audio_start_idx, current_frames)

        if global_rank == 0:
            print(f"Generating segment {segment_idx + 1}/{num_segments} ({current_frames} frames)...", flush=True)
        segment_started = time.perf_counter()
        output, latent = pipe.generate_avc(
            video=current_video,
            video_latent=latent,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=current_frames,
            num_cond_frames=num_cond_frames,
            num_inference_steps=num_inference_steps,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            generator=generator,
            output_type="both",
            use_kv_cache=True,
            offload_kv_cache=False,
            enhance_hf=not use_distill,
            audio_emb=audio_embs,
            ref_latent=ref_latent,
            ref_img_index=ref_img_index,
            mask_frame_range=mask_frame_range,
            ref_target_masks=ref_target_masks,
            use_distill=use_distill,
        )
        output = output[0]
        new_video = [PIL.Image.fromarray((output[i] * 255).astype(np.uint8)) for i in range(output.shape[0])]
        del output, audio_embs

        new_frames = new_video[num_cond_frames:]
        if not new_frames:
            raise RuntimeError(
                f"Continuation segment {segment_idx + 1} produced no new frames after trimming {num_cond_frames} conditioning frames"
            )
        if cp_rank == 0:
            seg_path = os.path.join(output_dir, f"segment_{segment_idx + 1}")
            save_video_ffmpeg(
                torch.from_numpy(np.array(new_frames)), seg_path, audio_path=None, fps=save_fps, quality=5
            )
            segment_paths.append(seg_path + ".mp4")

        current_video = new_video
        timings[f"segment_{segment_idx + 1}"] = _log_timing(f"segment_{segment_idx + 1}", segment_started, global_rank)
        torch_gc()
        gc.collect()
        _log_memory(f"after_segment_{segment_idx + 1}", global_rank)

    if cp_rank == 0 and segment_paths:
        mux_started = time.perf_counter()
        _concat_and_mux(segment_paths, os.path.join(output_dir, "output"), merge_speech_path)
        timings["mux"] = _log_timing("mux", mux_started, global_rank)

    if cp_rank == 0 and merge_speech_path and os.path.exists(merge_speech_path):
        os.remove(merge_speech_path)

    timings["total"] = _log_timing("total", total_started, global_rank)
    if global_rank == 0:
        summary = ", ".join(f"{k}={v:.2f}s" for k, v in timings.items())
        print(f"[longcat][timing] summary: {summary}", flush=True)
        _log_memory("final", global_rank)


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
