"""Configuration for the LongCat-Video FastAPI service.

Centralised, file-based configuration with environment-variable overrides.

Resolution order (highest → lowest):
    1. Environment variable (``LONGCAT_*``)  — keeps container/systemd deploys working
    2. ``config.toml`` (path via ``LONGCAT_CONFIG``, default ``<repo>/config.toml``)
    3. Built-in default in this file

Why a config file: previously several "safe" defaults (the A100-40G runtime profile
and the auth credentials) were hard-coded as literals in ``schemas.py`` / here. They
are now declared ONCE in ``config.toml`` (see ``config.toml.example``) and read by the
whole process, so tuning the service is a single-file, global change.

After loading, values that came from ``config.toml`` (not from the environment) are
written back into ``os.environ`` via ``setdefault`` so that the torchrun worker
subprocesses launched per request inherit the same settings — that is what makes the
file a *global* source of truth rather than something only the API process sees.
"""
import os
import tomllib
import secrets
from pathlib import Path

# ---------------------------------------------------------------------------
# repository layout
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"

# ---------------------------------------------------------------------------
# config.toml loading
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(os.environ.get("LONGCAT_CONFIG", str(REPO_ROOT / "config.toml")))
_cfg = {}
if _CONFIG_PATH.is_file():
    try:
        with open(_CONFIG_PATH, "rb") as _f:
            _cfg = tomllib.load(_f)
    except Exception as _e:  # noqa: BLE001 - never let a bad config file crash import
        print(f"[longcat][config] 读取 {_CONFIG_PATH} 失败，回退到环境变量/内置默认: {_e}", flush=True)


def _section(name: str) -> dict:
    return _cfg.get(name, {}) or {}


def _env_or_cfg(section: str, key: str, env_name: str, default):
    """env var wins; else config.toml; else built-in default."""
    if env_name and env_name in os.environ:
        return os.environ[env_name]
    return _section(section).get(key, default)


def _as_bool(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _as_int(v, default: int) -> int:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# model weights
# ---------------------------------------------------------------------------
CHECKPOINT_DIR_VIDEO = _env_or_cfg(
    "model", "checkpoint_dir_video", "LONGCAT_CHECKPOINT_DIR_VIDEO",
    str(REPO_ROOT / "weights" / "LongCat-Video"),
)

DEFAULT_CHECKPOINT_DIR_AVATAR_V1 = _env_or_cfg(
    "model", "checkpoint_dir_avatar_v1", "LONGCAT_CHECKPOINT_DIR_AVATAR_V1",
    str(REPO_ROOT / "weights" / "LongCat-Video-Avatar"),
)
DEFAULT_CHECKPOINT_DIR_AVATAR_V15 = _env_or_cfg(
    "model", "checkpoint_dir_avatar_v15", "LONGCAT_CHECKPOINT_DIR_AVATAR_V15",
    str(REPO_ROOT / "weights" / "LongCat-Video-Avatar-1.5"),
)

# legacy single-override (still supported)
CHECKPOINT_DIR_AVATAR = _env_or_cfg(
    "model", "checkpoint_dir_avatar", "LONGCAT_CHECKPOINT_DIR_AVATAR",
    DEFAULT_CHECKPOINT_DIR_AVATAR_V15,
)


def avatar_checkpoint_dir(model_type: str) -> str:
    """Resolve the avatar checkpoint directory for a given model_type.

    Reads LONGCAT_CHECKPOINT_DIR_AVATAR (legacy, overrides both revisions)
    then LONGCAT_CHECKPOINT_DIR_AVATAR_V1 / _V15 (per-revision, possibly from
    config.toml via the setdefault propagation below) at call time, falling
    back to the default weights/ layout.
    """
    override = os.environ.get("LONGCAT_CHECKPOINT_DIR_AVATAR")
    if override:
        return override
    if model_type == "avatar-v1.0":
        return os.environ.get("LONGCAT_CHECKPOINT_DIR_AVATAR_V1") or DEFAULT_CHECKPOINT_DIR_AVATAR_V1
    return os.environ.get("LONGCAT_CHECKPOINT_DIR_AVATAR_V15") or DEFAULT_CHECKPOINT_DIR_AVATAR_V15


def checkpoint_for_task(task_type: str, model_type: str = None) -> str:
    if task_type in ("avatar_single", "avatar_multi"):
        return avatar_checkpoint_dir(model_type or "avatar-v1.5")
    return CHECKPOINT_DIR_VIDEO


# ---------------------------------------------------------------------------
# distributed / runtime
# ---------------------------------------------------------------------------
NUM_GPUS = _as_int(_env_or_cfg("runtime", "num_gpus", "LONGCAT_NUM_GPUS", "1"), 1)
_cp = _as_int(_env_or_cfg("runtime", "context_parallel_size", "LONGCAT_CONTEXT_PARALLEL_SIZE", "0"), 0)
CONTEXT_PARALLEL_SIZE = _cp if _cp and _cp > 0 else NUM_GPUS
ENABLE_COMPILE = _as_bool(_env_or_cfg("runtime", "enable_compile", "LONGCAT_ENABLE_COMPILE", "0"))
GPU_CONCURRENCY = _as_int(_env_or_cfg("runtime", "gpu_concurrency", "LONGCAT_GPU_CONCURRENCY", "1"), 1)

# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------
WORK_DIR = Path(_env_or_cfg("runtime", "work_dir", "LONGCAT_WORK_DIR", str(REPO_ROOT / "api_work")))
UPLOAD_DIR = WORK_DIR / "uploads"
OUTPUT_DIR = WORK_DIR / "outputs"
LOG_DIR = WORK_DIR / "logs"
TASK_DB = WORK_DIR / "tasks.json"
UPLOAD_TTL_SECONDS = _as_int(
    _env_or_cfg("runtime", "upload_ttl_seconds", "LONGCAT_UPLOAD_TTL_SECONDS", str(24 * 3600)), 24 * 3600
)

ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_AUDIO_EXT = {".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg"}


def ensure_dirs():
    for d in (WORK_DIR, UPLOAD_DIR, OUTPUT_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------
HOST = _env_or_cfg("server", "host", "LONGCAT_HOST", "0.0.0.0")
PORT = _as_int(_env_or_cfg("server", "port", "LONGCAT_PORT", "8000"), 8000)

# ---------------------------------------------------------------------------
# auth — fail-closed
#   * No built-in password / token any more. When auth is enabled the operator
#     MUST supply one (via env or config.toml); otherwise the service refuses to
#     start. A missing token is auto-generated randomly and printed at startup.
# ---------------------------------------------------------------------------
AUTH_ENABLED = _as_bool(_env_or_cfg("server", "auth_enabled", "LONGCAT_AUTH", "0"))
AUTH_USER = _env_or_cfg("server", "auth_user", "LONGCAT_USER", "admin")
AUTH_PASS = _env_or_cfg("server", "auth_pass", "LONGCAT_PASS", "")      # empty => must be set when auth on
AUTH_TOKEN = _env_or_cfg("server", "auth_token", "LONGCAT_AUTH_TOKEN", "")  # empty => random at startup
EMBED_H5 = _as_bool(_env_or_cfg("server", "embed_h5", "LONGCAT_EMBED_H5", "0"))
H5_DIR = REPO_ROOT / "h5"
HEALTH_PATH = "/health"


def validate_auth():
    """Fail-closed auth check. Call once at startup (after env/CLI overrides).

    Raises RuntimeError if auth is enabled but no password was provided.
    Auto-generates a random cookie secret when none is configured.
    """
    global AUTH_TOKEN
    if not AUTH_ENABLED:
        return
    if not AUTH_PASS:
        raise RuntimeError(
            "鉴权已开启 (LONGCAT_AUTH=1 / config.toml [server].auth_enabled=true) "
            "但未设置密码。请通过 LONGCAT_PASS 或 config.toml [server].auth_pass 设置强密码，"
            "否则服务拒绝启动（fail-closed）。"
        )
    if not AUTH_TOKEN:
        AUTH_TOKEN = secrets.token_hex(16)
        print(
            f"[longcat][auth] 未配置 auth_token，已自动生成随机 cookie secret: {AUTH_TOKEN}\n"
            f"[longcat][auth]   如需固定 secret，请设置 LONGCAT_AUTH_TOKEN 或 config.toml [server].auth_token",
            flush=True,
        )


# ---------------------------------------------------------------------------
# A100-40G runtime profile — values come from config.toml, NOT hard-coded here.
#   The API normalises avatar-v1.5 requests to these SAFE CEILINGS (clamp down
#   only) so a 40GB card does not OOM, while still honouring caller intent where
#   it is memory-safe. See api/schemas.py::apply_a100_40g_profile.
# ---------------------------------------------------------------------------
A100_40G_PROFILE_ENABLED = _as_bool(
    _env_or_cfg("profile", "a100_40g_enabled", "LONGCAT_A100_40G_PROFILE", "1")
)
A100_40G = {
    "max_resolution": _env_or_cfg("profile", "max_resolution", "LONGCAT_A100_40G_MAX_RESOLUTION", "480p"),
    "max_num_segments": _as_int(
        _env_or_cfg("profile", "max_num_segments", "LONGCAT_A100_40G_MAX_SEGMENTS", "1"), 1
    ),
    "forced_num_inference_steps": _as_int(
        _env_or_cfg("profile", "forced_num_inference_steps", "LONGCAT_A100_40G_STEPS", "8"), 8
    ),
    "force_use_int8": _as_bool(_env_or_cfg("profile", "force_use_int8", "LONGCAT_A100_40G_FORCE_INT8", "1")),
    "force_use_distill": _as_bool(_env_or_cfg("profile", "force_use_distill", "LONGCAT_A100_40G_FORCE_DISTILL", "1")),
    "text_guidance_scale": float(
        _env_or_cfg("profile", "text_guidance_scale", "LONGCAT_A100_40G_TEXT_GUIDANCE", "1.0")
    ),
    "audio_guidance_scale": float(
        _env_or_cfg("profile", "audio_guidance_scale", "LONGCAT_A100_40G_AUDIO_GUIDANCE", "1.0")
    ),
}


# ---------------------------------------------------------------------------
# propagate config.toml values into the environment so torchrun subprocesses
# inherit them (global unified management). Env vars already set are preserved.
# ---------------------------------------------------------------------------
_ENV_PROPAGATE = [
    ("LONGCAT_HOST", HOST),
    ("LONGCAT_PORT", str(PORT)),
    ("LONGCAT_NUM_GPUS", str(NUM_GPUS)),
    ("LONGCAT_CONTEXT_PARALLEL_SIZE", str(CONTEXT_PARALLEL_SIZE)),
    ("LONGCAT_ENABLE_COMPILE", "1" if ENABLE_COMPILE else "0"),
    ("LONGCAT_GPU_CONCURRENCY", str(GPU_CONCURRENCY)),
    ("LONGCAT_WORK_DIR", str(WORK_DIR)),
    ("LONGCAT_UPLOAD_TTL_SECONDS", str(UPLOAD_TTL_SECONDS)),
    ("LONGCAT_CHECKPOINT_DIR_VIDEO", CHECKPOINT_DIR_VIDEO),
    ("LONGCAT_CHECKPOINT_DIR_AVATAR_V1", DEFAULT_CHECKPOINT_DIR_AVATAR_V1),
    ("LONGCAT_CHECKPOINT_DIR_AVATAR_V15", DEFAULT_CHECKPOINT_DIR_AVATAR_V15),
    ("LONGCAT_A100_40G_PROFILE", "1" if A100_40G_PROFILE_ENABLED else "0"),
]
for _env_name, _val in _ENV_PROPAGATE:
    if _val:  # never push an empty value (let a child keep its own default/env)
        os.environ.setdefault(_env_name, str(_val))


# ---------------------------------------------------------------------------
# weight readiness checks (fail fast with a friendly message)
# ---------------------------------------------------------------------------
BASE_MODEL_SUBFOLDERS = ["tokenizer", "text_encoder", "vae", "scheduler"]

AVATAR_REQUIRED_SUBFOLDERS = {
    "avatar-v1.0": ["avatar_single", "avatar_multi", "chinese-wav2vec2-base", "vocal_separator"],
    "avatar-v1.5": ["base_model", "whisper-large-v3", "vocal_separator", "scheduler"],
}

HF_REPO_VIDEO = "meituan-longcat/LongCat-Video"
HF_REPO_AVATAR_V1 = "meituan-longcat/LongCat-Video-Avatar"
HF_REPO_AVATAR_V15 = "meituan-longcat/LongCat-Video-Avatar-1.5"


def _download_hint(repo: str, local_dir: str) -> str:
    return (
        f"  下载: huggingface-cli download {repo} --local-dir {local_dir} "
        f"--local-dir-use-symlinks False\n"
        f"  (大陆服务器加: export HF_ENDPOINT=https://hf-mirror.com)"
    )


def check_weights(model_type: str = None, task_type: str = None, skip_missing_dirs: bool = False):
    """Validate that the required weights are present on disk.

    Returns ``(ok, problems)`` where ``problems`` is a list of human-readable
    strings describing exactly what is missing and how to fix it. Used both for
    per-request preflight and the startup readiness check.
    """
    problems: list = []

    base_dir = CHECKPOINT_DIR_VIDEO
    if not os.path.isdir(base_dir):
        if not skip_missing_dirs:
            problems.append(
                f"[基础视频模型] 目录不存在: {base_dir}\n"
                f"  {_download_hint(HF_REPO_VIDEO, base_dir)}\n"
                f"  或设置环境变量 LONGCAT_CHECKPOINT_DIR_VIDEO 指向已下载目录"
            )
    else:
        missing = [s for s in BASE_MODEL_SUBFOLDERS if not os.path.isdir(os.path.join(base_dir, s))]
        if missing:
            problems.append(f"[基础视频模型] {base_dir} 缺少子目录: {missing}")

    if task_type in ("avatar_single", "avatar_multi"):
        mt = model_type or "avatar-v1.5"
        avatar_dir = avatar_checkpoint_dir(mt)
        if not os.path.isdir(avatar_dir):
            if not skip_missing_dirs:
                repo = HF_REPO_AVATAR_V1 if mt == "avatar-v1.0" else HF_REPO_AVATAR_V15
                problems.append(
                    f"[数字人 {mt}] 权重目录不存在: {avatar_dir}\n"
                    f"  {_download_hint(repo, avatar_dir)}\n"
                    f"  或设置 LONGCAT_CHECKPOINT_DIR_AVATAR_V1 / LONGCAT_CHECKPOINT_DIR_AVATAR_V15"
                )
        else:
            required = AVATAR_REQUIRED_SUBFOLDERS.get(mt, [])
            if mt == "avatar-v1.0":
                dit = "avatar_single" if task_type == "avatar_single" else "avatar_multi"
                required = [s for s in required if s in ("chinese-wav2vec2-base", "vocal_separator", dit)]
            missing = [s for s in required if not os.path.isdir(os.path.join(avatar_dir, s))]
            if missing:
                problems.append(f"[数字人 {mt}] {avatar_dir} 缺少子目录: {missing}")

    return (len(problems) == 0, problems)


# ---------------------------------------------------------------------------
# script dispatch table
# ---------------------------------------------------------------------------
SCRIPTS = {
    "text_to_video": SCRIPTS_DIR / "run_text_to_video.py",
    "image_to_video": SCRIPTS_DIR / "run_image_to_video.py",
    "video_continuation": SCRIPTS_DIR / "run_video_continuation.py",
    "avatar_single": SCRIPTS_DIR / "run_avatar_single.py",
    "avatar_multi": SCRIPTS_DIR / "run_avatar_multi.py",
}
