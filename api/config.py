"""Configuration for the LongCat-Video FastAPI service.

Override any value via environment variables (e.g. LONGCAT_CHECKPOINT_DIR,
LONGCAT_NUM_GPUS, LONGCAT_WORK_DIR, LONGCAT_HOST, LONGCAT_PORT).
"""
import os
from pathlib import Path

# Auto-load a local .env if python-dotenv is available. No-op otherwise, so the
# service also works when variables are exported directly in the shell / systemd.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


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

# --- in-process (resident) inference worker ---
# When enabled (default), the API launches a long-lived torchrun worker at
# startup that loads the avatar model ONCE and serves avatar-single requests
# in-process (no per-request cold start). Set LONGCAT_INPROCESS=0 to fall back
# to the original per-request subprocess dispatch.
INPROCESS = os.environ.get("LONGCAT_INPROCESS", "1") == "1"
WORKER_HOST = os.environ.get("LONGCAT_WORKER_HOST", "127.0.0.1")
WORKER_PORT = int(os.environ.get("LONGCAT_WORKER_PORT", "29500"))
MODEL_TYPE = os.environ.get("LONGCAT_MODEL_TYPE", "avatar-v1.5")
AVATAR_MODE = os.environ.get("LONGCAT_AVATAR_MODE", "single")
USE_INT8 = os.environ.get("LONGCAT_USE_INT8", "0") == "1"
# populated at runtime by server.lifespan
WORKER_CLIENT = None
WORKER_PROC = None

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
