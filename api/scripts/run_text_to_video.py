import os
import json
import argparse
import datetime
import PIL.Image
import numpy as np

import torch
import torch.distributed as dist

from transformers import AutoTokenizer, UMT5EncoderModel
from torchvision.io import write_video

from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.context_parallel import context_parallel_util
from longcat_video.context_parallel.context_parallel_util import init_context_parallel


DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards"
)


def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def generate(args):
    with open(args.task_input, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    prompt = cfg["prompt"]
    negative_prompt = cfg.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    height = int(cfg.get("height", 480))
    width = int(cfg.get("width", 832))
    num_frames = int(cfg.get("num_frames", 93))
    num_inference_steps = int(cfg.get("num_inference_steps", 50))
    guidance_scale = float(cfg.get("guidance_scale", 4.0))
    spatial_refine_only = bool(cfg.get("spatial_refine_only", False))
    seed = int(cfg.get("seed", 42))

    checkpoint_dir = args.checkpoint_dir
    context_parallel_size = args.context_parallel_size
    enable_compile = args.enable_compile
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare distributed environment
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24))
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()

    init_context_parallel(context_parallel_size=context_parallel_size, global_rank=global_rank, world_size=num_processes)
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, subfolder="tokenizer", torch_dtype=torch.bfloat16)
    text_encoder = UMT5EncoderModel.from_pretrained(checkpoint_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(checkpoint_dir, subfolder="vae", torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(checkpoint_dir, subfolder="scheduler", torch_dtype=torch.bfloat16)
    dit = LongCatVideoTransformer3DModel.from_pretrained(checkpoint_dir, subfolder="dit", cp_split_hw=cp_split_hw, torch_dtype=torch.bfloat16)

    if enable_compile:
        dit = torch.compile(dit)

    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )
    pipe.to(local_rank)

    generator = torch.Generator(device=local_rank)
    generator.manual_seed(seed + global_rank)

    # t2v (480p)
    output = pipe.generate_t2v(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )[0]

    if local_rank == 0:
        output_tensor = torch.from_numpy(np.array(output))
        output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)
        write_video(os.path.join(args.output_dir, "stage1_480p.mp4"), output_tensor, fps=15, video_codec="libx264", options={"crf": "18"})
    del output
    torch_gc()

    # t2v distill (480p)
    cfg_step_lora_path = os.path.join(checkpoint_dir, "lora/cfg_step_lora.safetensors")
    pipe.dit.load_lora(cfg_step_lora_path, "cfg_step_lora")
    pipe.dit.enable_loras(["cfg_step_lora"])

    if enable_compile:
        dit = torch.compile(dit)

    output_distill = pipe.generate_t2v(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=16,
        use_distill=True,
        guidance_scale=1.0,
        generator=generator,
    )[0]
    pipe.dit.disable_all_loras()

    if local_rank == 0:
        output_processed_tensor = torch.from_numpy(np.array(output_distill))
        output_processed_tensor = (output_processed_tensor * 255).clamp(0, 255).to(torch.uint8)
        write_video(os.path.join(args.output_dir, "stage2_distill_480p.mp4"), output_processed_tensor, fps=15, video_codec="libx264", options={"crf": "18"})

    # t2v refinement (720p)
    refinement_lora_path = os.path.join(checkpoint_dir, "lora/refinement_lora.safetensors")
    pipe.dit.load_lora(refinement_lora_path, "refinement_lora")
    pipe.dit.enable_loras(["refinement_lora"])
    pipe.dit.enable_bsa()

    if enable_compile:
        dit = torch.compile(dit)

    stage1_video = [(output_distill[i] * 255).astype(np.uint8) for i in range(output_distill.shape[0])]
    stage1_video = [PIL.Image.fromarray(img) for img in stage1_video]
    del output_distill
    torch_gc()

    output_refine = pipe.generate_refine(
        prompt=prompt,
        stage1_video=stage1_video,
        num_inference_steps=50,
        generator=generator,
        spatial_refine_only=spatial_refine_only,
    )[0]

    pipe.dit.disable_all_loras()
    pipe.dit.disable_bsa()

    if local_rank == 0:
        output_tensor = torch.from_numpy(output_refine)
        output_tensor = (output_tensor * 255).clamp(0, 255).to(torch.uint8)
        fps = 15 if spatial_refine_only else 30
        write_video(os.path.join(args.output_dir, "output.mp4"), output_tensor, fps=fps, video_codec="libx264", options={"crf": "10"})
        # compatibility alias
        write_video(os.path.join(args.output_dir, "output_t2v_refine.mp4"), output_tensor, fps=fps, video_codec="libx264", options={"crf": "10"})


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_input", type=str, required=True, help="path to task input json")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--context_parallel_size", type=int, default=1)
    parser.add_argument("--enable_compile", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
