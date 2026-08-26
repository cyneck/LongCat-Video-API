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
    """env var wins; else config.toml; else built-in default.

    An empty string (in env or config.toml) is treated as "unset", so leaving a
    field blank falls back to the built-in default instead of becoming an empty
    path/value. (e.g. ``checkpoint_dir_video = ""`` in config.toml -> use the
    default ``weights/LongCat-Video`` layout, not an empty directory.)
    """
    if env_name and env_name in os.environ and str(os.environ[env_name]).strip() != "":
        return os.environ[env_name]
    val = _section(section).get(key, None)
    if val is None or str(val).strip() == "":
        return default
    return val


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
# low-VRAM 安全档（内部曾叫 A100-40G；现按真实显存自动判定，不再硬编码“40G 假设”）
#   * 改进点：之前该档“默认开启 + 数值从代码挪到 config.toml”，但仍等于假设
#     “所有部署都是 40G 卡”，与“不要硬编码硬件假设”相悖——任何部署都会强制 INT8+蒸馏
#     并打印“请求已被 profile 调整”日志。现改为：
#        - 显式 env / config.toml 优先（运维可按真实硬件明确开启或关闭）；
#        - 未显式设置时，启动时探测真实最小 GPU 显存：≤ LOW_VRAM_THRESHOLD_GB 才启用
#          安全档；> 阈值（如 A100-80G / H100）不强制 INT8/蒸馏，尊重调用方创意意图，
#          也不会打印那条日志。
#   * 安全档只做“向下钳制”（分辨率/段数封顶、强制 INT8+蒸馏防 OOM），从不改写创意参数。
#     详见 api/schemas.py::apply_a100_40g_profile。
#   * 注：env / config.toml key 仍沿用 `a100_40g_*`（LONGCAT_A100_40G_*）以兼容既有部署。
# ---------------------------------------------------------------------------
LOW_VRAM_THRESHOLD_GB = 50.0
LOW_VRAM_PROFILE_SOURCE = "default"


def _detect_low_vram(threshold_gb: float = LOW_VRAM_THRESHOLD_GB) -> bool:
    """按真实 GPU 显存判定是否低显存——绝不做“当前是 40G 就假设所有都是 40G”的硬编码。

    返回 True 表示需要启用内存安全档（强制 INT8 + 蒸馏等）。探测失败（无 CUDA /
    torch 不可导入）时保守返回 False 并打印提示，由运维显式设置 env/config 接管。
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        min_gb = min(
            torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            for i in range(torch.cuda.device_count())
        )
        return min_gb <= threshold_gb
    except Exception as _e:  # noqa: BLE001
        print(
            f"[longcat][config] GPU 显存探测失败({_e})，低显存安全档默认关闭；"
            f"若确为低显存部署请显式设置 LONGCAT_A100_40G_PROFILE=1",
            flush=True,
        )
        return False


def _resolve_low_vram_profile_enabled() -> bool:
    """解析安全档开关：env > config.toml > 自动探测真实显存（不再默认开启）。"""
    global LOW_VRAM_PROFILE_SOURCE
    env_val = os.environ.get("LONGCAT_A100_40G_PROFILE", "").strip()
    if env_val != "":
        LOW_VRAM_PROFILE_SOURCE = "env(LONGCAT_A100_40G_PROFILE)"
        return _as_bool(env_val)
    cfg_val = _section("profile").get("a100_40g_enabled", None)
    if cfg_val is not None and str(cfg_val).strip() != "":
        LOW_VRAM_PROFILE_SOURCE = "config.toml[profile].a100_40g_enabled"
        return _as_bool(cfg_val)
    LOW_VRAM_PROFILE_SOURCE = f"auto-detect(显存≤{LOW_VRAM_THRESHOLD_GB:.0f}GB)"
    return _detect_low_vram()


LOW_VRAM_PROFILE_ENABLED = _resolve_low_vram_profile_enabled()
LOW_VRAM = {
    "max_resolution": _env_or_cfg("profile", "max_resolution", "LONGCAT_A100_40G_MAX_RESOLUTION", "480p"),
    # max_num_segments: 0 = 不限制。段数走滑动窗口串行（每段独立前向、用完即释放显存/内存），
    # 不额外占用显存，仅影响总生成耗时；生产若担心超长音频任务耗时爆炸，可设为上限值。
    "max_num_segments": _as_int(
        _env_or_cfg("profile", "max_num_segments", "LONGCAT_A100_40G_MAX_SEGMENTS", "0"), 0
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

print(
    f"[longcat][config] 低显存安全档(legacy: A100-40G): "
    f"{'启用' if LOW_VRAM_PROFILE_ENABLED else '关闭'}（来源={LOW_VRAM_PROFILE_SOURCE}）",
    flush=True,
)


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
    ("LONGCAT_A100_40G_PROFILE", "1" if LOW_VRAM_PROFILE_ENABLED else "0"),
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
