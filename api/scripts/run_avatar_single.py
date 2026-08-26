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


def _concat_and_mux(segment_paths, output_base, audio_path):
    """ffmpeg concat 各段纯视频 + 一次性 mux 完整音频 → output_base+'.mp4'。

    全程磁盘操作，内存仅「单段」级别（每段已落盘即释放），契合滑动窗口串行：
    段数只影响总耗时，不额外占用显存/内存，故可在不溢出的前提下处理任意时长音频。
    """
    list_path = output_base + ".concat.txt"
    with open(list_path, "w") as f:
        for p in segment_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    merged = output_base + ".merged.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", merged],
        check=True,
    )
    final = output_base + ".mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-i", merged, "-i", audio_path,
         "-c", "copy", "-map", "0:v:0", "-map", "1:a:0", "-shortest", final],
        check=True,
    )
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
    with open(args.task_input, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    prompt = cfg["prompt"]
    negative_prompt = cfg.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    raw_speech_path = cfg["cond_audio"]["person1"]
    image_path = cfg.get("cond_image")  # None for at2v
    stage_1 = cfg.get("stage_1", "ai2v")  # at2v / ai2v
    resolution = cfg.get("resolution", "480p")
    num_segments_raw = cfg.get("num_segments", "auto")  # int 或 "auto"（按音频时长自适应）
    print(f"[longcat][seg] 收到 num_segments={num_segments_raw!r}（'auto'=按音频时长自适应）", flush=True)
    num_inference_steps = int(cfg.get("num_inference_steps", 50))
    text_guidance_scale = float(cfg.get("text_guidance_scale", 4.0))
    audio_guidance_scale = float(cfg.get("audio_guidance_scale", 4.0))
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
    segment_paths = []  # 各段纯视频路径，最后统一 concat + mux 完整音频（内存仅单段级别）

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

    # prepare distributed environment
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

    # initialize models
    # Base video model supplies the shared tokenizer/text_encoder/vae/scheduler.
    # Default: sibling dir `LongCat-Video` next to the avatar checkpoint. Override with
    # LONGCAT_CHECKPOINT_DIR_VIDEO to support arbitrary weights layouts.
    # normpath collapses the `..` so the path resolves even when the intermediate
    # avatar directory is absent (otherwise os.path.isdir trips on the missing
    # component and HuggingFace reports "Incorrect path_or_model_id").
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

    # initialize audio models
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

    generator = torch.Generator(device=local_rank)
    generator.manual_seed(seed + global_rank)

    if cp_rank == 0:
        temp_vocal_path = extract_vocal_from_speech(raw_speech_path, f"/tmp/temp_speech_{generate_random_uid()}_{global_rank}_vocal.wav", vocal_separator, audio_output_dir_temp)
        assert temp_vocal_path is not None and os.path.exists(temp_vocal_path), "No vocal detected"

        speech_array, sr = librosa.load(temp_vocal_path, sr=16000)
        source_duration = len(speech_array) / sr

        # --- 自适应段数（audio 模式）：按音频时长算，不硬编码魔法数 ---
        if isinstance(num_segments_raw, str) and num_segments_raw.strip().lower() == "auto":
            first_seg_dur = num_frames / save_fps
            seg_advance = (num_frames - num_cond_frames) / save_fps
            if source_duration <= first_seg_dur:
                num_segments = 1
            else:
                num_segments = 1 + math.ceil((source_duration - first_seg_dur) / seg_advance)
            _max_seg = config.A100_40G.get("max_num_segments", 0)
            if _max_seg and _max_seg > 0 and num_segments > _max_seg:
                print(f"[longcat][auto] num_segments={num_segments} 超过 profile 上限，封顶为 {_max_seg}", flush=True)
                num_segments = _max_seg
            print(f"[longcat][auto] 音频 {source_duration:.1f}s → 自动 {num_segments} 段 (每段推进 {seg_advance:.2f}s)", flush=True)
        else:
            num_segments = max(1, int(num_segments_raw))
            print(f"[longcat][seg] 使用固定段数 num_segments={num_segments}", flush=True)

        generate_duration = num_frames / save_fps + (num_segments - 1) * (num_frames - num_cond_frames) / save_fps
        added_sample_nums = math.ceil((generate_duration - source_duration) * sr)
        if added_sample_nums > 0:
            speech_array = np.append(speech_array, [0.0] * added_sample_nums)

        full_audio_emb = pipe.get_audio_embedding(speech_array, fps=save_fps * audio_stride, device=local_rank, sample_rate=sr, model_type=model_type)
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

    # Audio encoder (whisper-large-v3, ~3GB) is only needed for the one-time
    # embedding extraction above. Offload it to CPU now to free GPU memory for
    # the DiT forward — on a single A100-40GB the INT8 DiT + distill LoRA leave
    # only ~720MB free and the LoRA branch OOMs without this headroom.
    pipe.audio_encoder = pipe.audio_encoder.to("cpu")
    torch.cuda.empty_cache()

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
            seg_path = os.path.join(output_dir, "segment_1")
            save_video_ffmpeg(torch.from_numpy(np.array(video)), seg_path, audio_path=None, fps=save_fps, quality=5)
            segment_paths.append(seg_path + ".mp4")
        del output
        torch_gc()
    else:
        raise NotImplementedError(f"Not supported type of stage_1: {stage_1}")

    if context_parallel_util.get_cp_size() > 1:
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

        current_video = new_video

        if cp_rank == 0:
            seg_path = os.path.join(output_dir, f"video_continue_{segment_idx+1}")
            save_video_ffmpeg(torch.from_numpy(np.array(new_video)), seg_path, audio_path=None, fps=save_fps, quality=5)
            segment_paths.append(seg_path + ".mp4")
        torch_gc()

    # 合并所有段纯视频 + 一次性 mux 完整音频（滑动窗口串行：内存仅单段级别，不溢出）
    if cp_rank == 0 and segment_paths:
        _concat_and_mux(segment_paths, os.path.join(output_dir, "output"), raw_speech_path)


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
