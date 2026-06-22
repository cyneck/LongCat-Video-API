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
CHECKPOINT_DIR_AVATAR = os.environ.get(
    "LONGCAT_CHECKPOINT_DIR_AVATAR",
    str(REPO_ROOT / "weights" / "LongCat-Video-Avatar-1.5"),
)

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


# --- script dispatch table ---
SCRIPTS = {
    "text_to_video": SCRIPTS_DIR / "run_text_to_video.py",
    "image_to_video": SCRIPTS_DIR / "run_image_to_video.py",
    "video_continuation": SCRIPTS_DIR / "run_video_continuation.py",
    "avatar_single": SCRIPTS_DIR / "run_avatar_single.py",
    "avatar_multi": SCRIPTS_DIR / "run_avatar_multi.py",
}

# checkpoint used by each task type
TASK_CHECKPOINT = {
    "text_to_video": CHECKPOINT_DIR_VIDEO,
    "image_to_video": CHECKPOINT_DIR_VIDEO,
    "video_continuation": CHECKPOINT_DIR_VIDEO,
    "avatar_single": CHECKPOINT_DIR_AVATAR,
    "avatar_multi": CHECKPOINT_DIR_AVATAR,
}
