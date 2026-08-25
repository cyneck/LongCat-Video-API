"""Configuration for the LongCat-Video FastAPI service.

Override any value via environment variables (e.g. LONGCAT_CHECKPOINT_DIR,
LONGCAT_NUM_GPUS, LONGCAT_WORK_DIR, LONGCAT_HOST, LONGCAT_PORT).
"""
import os
from pathlib import Path


# --- repository layout ---
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"

# --- model weights ---
# Each task maps to a checkpoint directory. Adjust if you keep weights elsewhere.
CHECKPOINT_DIR_VIDEO = os.environ.get(
    "LONGCAT_CHECKPOINT_DIR_VIDEO",
    str(REPO_ROOT / "weights" / "LongCat-Video"),
)

# Avatar (digital-human) weights ship as two independent revisions with
# *different* directory layouts:
#   - v1.0  -> LongCat-Video-Avatar      (avatar_single/avatar_multi + chinese-wav2vec2-base)
#   - v1.5  -> LongCat-Video-Avatar-1.5  (base_model + whisper-large-v3)
# They must NOT be mixed: requesting avatar-v1.0 against the v1.5 directory (or
# vice-versa) fails at model load. We therefore resolve the checkpoint dir from
# the requested model_type instead of a single shared path. Defaults only —
# actual resolution happens in avatar_checkpoint_dir(), which re-reads the
# environment on every call so overrides always take effect.
DEFAULT_CHECKPOINT_DIR_AVATAR_V1 = str(REPO_ROOT / "weights" / "LongCat-Video-Avatar")
DEFAULT_CHECKPOINT_DIR_AVATAR_V15 = str(REPO_ROOT / "weights" / "LongCat-Video-Avatar-1.5")

# Kept for backward compatibility: legacy single-override deployments that set
# only LONGCAT_CHECKPOINT_DIR_AVATAR keep working (see avatar_checkpoint_dir).
CHECKPOINT_DIR_AVATAR = os.environ.get(
    "LONGCAT_CHECKPOINT_DIR_AVATAR",
    DEFAULT_CHECKPOINT_DIR_AVATAR_V15,
)


def avatar_checkpoint_dir(model_type: str) -> str:
    """Resolve the avatar checkpoint directory for a given model_type.

    Reads LONGCAT_CHECKPOINT_DIR_AVATAR (legacy, overrides both revisions)
    then LONGCAT_CHECKPOINT_DIR_AVATAR_V1 / _V15 (per-revision) at call time,
    falling back to the default weights/ layout.
    """
    override = os.environ.get("LONGCAT_CHECKPOINT_DIR_AVATAR")
    if override:
        return override
    if model_type == "avatar-v1.0":
        return os.environ.get("LONGCAT_CHECKPOINT_DIR_AVATAR_V1") or DEFAULT_CHECKPOINT_DIR_AVATAR_V1
    # default / avatar-v1.5
    return os.environ.get("LONGCAT_CHECKPOINT_DIR_AVATAR_V15") or DEFAULT_CHECKPOINT_DIR_AVATAR_V15


def checkpoint_for_task(task_type: str, model_type: str = None) -> str:
    """Pick the right weights dir for a task, version-aware for avatar tasks."""
    if task_type in ("avatar_single", "avatar_multi"):
        return avatar_checkpoint_dir(model_type or "avatar-v1.5")
    return CHECKPOINT_DIR_VIDEO

# --- distributed / runtime ---
NUM_GPUS = int(os.environ.get("LONGCAT_NUM_GPUS", "1"))            # number of GPUs per task
CONTEXT_PARALLEL_SIZE = int(os.environ.get("LONGCAT_CONTEXT_PARALLEL_SIZE", str(NUM_GPUS)))
ENABLE_COMPILE = os.environ.get("LONGCAT_ENABLE_COMPILE", "0") == "1"
GPU_CONCURRENCY = int(os.environ.get("LONGCAT_GPU_CONCURRENCY", "1"))  # how many tasks may use GPUs at once

# --- storage ---
WORK_DIR = Path(os.environ.get("LONGCAT_WORK_DIR", str(REPO_ROOT / "api_work")))
UPLOAD_DIR = WORK_DIR / "uploads"
OUTPUT_DIR = WORK_DIR / "outputs"
LOG_DIR = WORK_DIR / "logs"
TASK_DB = WORK_DIR / "tasks.json"

# --- server ---
HOST = os.environ.get("LONGCAT_HOST", "0.0.0.0")
PORT = int(os.environ.get("LONGCAT_PORT", "8000"))

# --- cleanup ---
# upload files older than this many seconds are purged on start (0 = disabled)
UPLOAD_TTL_SECONDS = int(os.environ.get("LONGCAT_UPLOAD_TTL_SECONDS", str(24 * 3600)))

ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_AUDIO_EXT = {".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg"}


def ensure_dirs():
    for d in (WORK_DIR, UPLOAD_DIR, OUTPUT_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --- weight readiness checks (fail fast with a friendly message) ---
# Shared base model supplies tokenizer / text_encoder / vae / scheduler for every
# task. Avatar revisions additionally need revision-specific subfolders.
BASE_MODEL_SUBFOLDERS = ["tokenizer", "text_encoder", "vae", "scheduler"]

AVATAR_REQUIRED_SUBFOLDERS = {
    "avatar-v1.0": ["avatar_single", "avatar_multi", "chinese-wav2vec2-base", "vocal_separator"],
    "avatar-v1.5": ["base_model", "whisper-large-v3", "vocal_separator", "scheduler"],
}

# HuggingFace repo ids, used only to build the download hint in error messages.
HF_REPO_VIDEO = "meituan-longcat/LongCat-Video"
HF_REPO_AVATAR_V1 = "meituan-longcat/LongCat-Video-Avatar"
HF_REPO_AVATAR_V15 = "meituan-longcat/LongCat-Video-Avatar-1.5"


def _download_hint(repo: str, local_dir: str) -> str:
    return (
        f"  下载: huggingface-cli download {repo} --local-dir {local_dir} "
        f"--local-dir-use-symlinks False\n"
        f"  (大陆服务器加: export HF_ENDPOINT=https://hf-mirror.com)"
    )


def check_weights(model_type: str = None, task_type: str = None):
    """Validate that the required weights are present on disk.

    Returns ``(ok, problems)`` where ``problems`` is a list of human-readable
    strings describing exactly what is missing and how to fix it. This is used
    both for per-request preflight and for the startup readiness check so a
    misconfigured weight layout surfaces immediately instead of crashing deep
    inside a torchrun subprocess.
    """
    problems: list = []

    # 1) shared base video model
    base_dir = CHECKPOINT_DIR_VIDEO
    if not os.path.isdir(base_dir):
        problems.append(
            f"[基础视频模型] 目录不存在: {base_dir}\n"
            f"  {_download_hint(HF_REPO_VIDEO, base_dir)}\n"
            f"  或设置环境变量 LONGCAT_CHECKPOINT_DIR_VIDEO 指向已下载目录"
        )
    else:
        missing = [s for s in BASE_MODEL_SUBFOLDERS if not os.path.isdir(os.path.join(base_dir, s))]
        if missing:
            problems.append(f"[基础视频模型] {base_dir} 缺少子目录: {missing}")

    # 2) avatar-specific revision
    if task_type in ("avatar_single", "avatar_multi"):
        mt = model_type or "avatar-v1.5"
        avatar_dir = avatar_checkpoint_dir(mt)
        if not os.path.isdir(avatar_dir):
            repo = HF_REPO_AVATAR_V1 if mt == "avatar-v1.0" else HF_REPO_AVATAR_V15
            problems.append(
                f"[数字人 {mt}] 权重目录不存在: {avatar_dir}\n"
                f"  {_download_hint(repo, avatar_dir)}\n"
                f"  或设置 LONGCAT_CHECKPOINT_DIR_AVATAR_V1 / LONGCAT_CHECKPOINT_DIR_AVATAR_V15"
            )
        else:
            required = AVATAR_REQUIRED_SUBFOLDERS.get(mt, [])
            # only the dit subfolder matching this task is mandatory
            if mt == "avatar-v1.0":
                dit = "avatar_single" if task_type == "avatar_single" else "avatar_multi"
                required = [s for s in required if s in ("chinese-wav2vec2-base", "vocal_separator", dit)]
            missing = [s for s in required if not os.path.isdir(os.path.join(avatar_dir, s))]
            if missing:
                problems.append(f"[数字人 {mt}] {avatar_dir} 缺少子目录: {missing}")

    return (len(problems) == 0, problems)


# --- script dispatch table ---
SCRIPTS = {
    "text_to_video": SCRIPTS_DIR / "run_text_to_video.py",
    "image_to_video": SCRIPTS_DIR / "run_image_to_video.py",
    "video_continuation": SCRIPTS_DIR / "run_video_continuation.py",
    "avatar_single": SCRIPTS_DIR / "run_avatar_single.py",
    "avatar_multi": SCRIPTS_DIR / "run_avatar_multi.py",
}

# --- optional: H5 embedding + simple login gate ---
# Flip on for production: LONGCAT_EMBED_H5=1 serves the H5 at "/",
# LONGCAT_AUTH=1 requires login (cookie) before any endpoint.
AUTH_ENABLED = os.environ.get("LONGCAT_AUTH", "0") == "1"
AUTH_USER = os.environ.get("LONGCAT_USER", "admin")
AUTH_PASS = os.environ.get("LONGCAT_PASS", "admin")
AUTH_TOKEN = os.environ.get("LONGCAT_AUTH_TOKEN", "longcat-demo-token")  # secret stored in the HttpOnly cookie
EMBED_H5 = os.environ.get("LONGCAT_EMBED_H5", "0") == "1"
H5_DIR = REPO_ROOT / "h5"
HEALTH_PATH = "/health"
