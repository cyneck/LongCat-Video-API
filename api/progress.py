"""Machine-readable task progress protocol for inference workers.

Only lines emitted by :func:`progress_event` are part of the task-progress
contract. Ordinary debug/timing logs are intentionally ignored by the parser.

Wire format (one JSON object per line)::

    [longcat][progress] {"v":1,"percent":45,"stage":"audio_ready",...}

Stage names are stable API identifiers; ``detail`` is user-facing text and may
change without breaking consumers.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

PROGRESS_PREFIX = "[longcat][progress] "
PROGRESS_VERSION = 1

STAGE_LABELS = {
    "queued": "排队中",
    "starting": "启动任务",
    "model_loading": "加载模型",
    "model_ready": "模型就绪",
    "audio_separation": "人声分离",
    "audio_features": "音频特征",
    "audio_ready": "音频特征完成",
    "video_generation": "生成视频",
    "muxing": "合并封装",
    "completed": "已完成",
    "failed": "任务失败",
}


@dataclass(frozen=True)
class ProgressSnapshot:
    percent: int
    stage: str
    detail: str = ""
    current_segment: int = 0
    total_segments: int = 0


def progress_event(
    percent: int,
    stage: str,
    detail: str = "",
    *,
    current_segment: int = 0,
    total_segments: int = 0,
    rank: int = 0,
    **extra: Any,
) -> None:
    """Emit one standardized progress event; only global rank 0 may emit it."""
    if rank != 0:
        return
    payload = {
        "v": PROGRESS_VERSION,
        "percent": max(0, min(100, int(percent))),
        "stage": str(stage),
        "detail": str(detail or ""),
        "current_segment": max(0, int(current_segment or 0)),
        "total_segments": max(0, int(total_segments or 0)),
    }
    if extra:
        payload["meta"] = extra
    print(PROGRESS_PREFIX + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def parse_progress(log_text: str, fallback: ProgressSnapshot | None = None) -> ProgressSnapshot:
    """Parse only standardized progress events and return the newest monotonic state."""
    best = fallback or ProgressSnapshot(0, "queued", "等待 GPU 执行资源")
    for line in (log_text or "").splitlines():
        pos = line.find(PROGRESS_PREFIX)
        if pos < 0:
            continue
        raw = line[pos + len(PROGRESS_PREFIX):].strip()
        try:
            event = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if event.get("v") != PROGRESS_VERSION:
            continue
        try:
            percent = max(0, min(100, int(event.get("percent", 0))))
        except (TypeError, ValueError):
            continue
        if percent < best.percent:
            continue
        best = ProgressSnapshot(
            percent=percent,
            stage=str(event.get("stage") or best.stage),
            detail=str(event.get("detail") or ""),
            current_segment=max(0, int(event.get("current_segment") or 0)),
            total_segments=max(0, int(event.get("total_segments") or 0)),
        )
    return best


def _segment_percent(current: int, total: int) -> int:
    total = max(1, int(total))
    current = max(1, min(int(current), total))
    return min(90, 45 + round(45 * current / total))


def install_worker_progress_hooks(module) -> None:
    """Attach the shared reporter to an Avatar worker without parsing debug logs."""
    state = {"total_segments": 0}
    original_count = getattr(module, "_generated_frame_count", None)
    original_timing = getattr(module, "_log_timing", None)

    if callable(original_count):
        def counted(frame_plan, cond_frames):
            state["total_segments"] = len(frame_plan or [])
            return original_count(frame_plan, cond_frames)
        module._generated_frame_count = counted

    if callable(original_timing):
        def timed(name, started_at, rank=0):
            elapsed = original_timing(name, started_at, rank)
            if name == "model_load":
                progress_event(25, "model_ready", "模型加载完成，准备音频预处理", rank=rank)
            elif name == "separator_init":
                progress_event(30, "audio_separation", "人声分离模型已就绪", rank=rank)
            elif name == "vocal_separation":
                progress_event(35, "audio_features", "人声分离完成，正在提取音频特征", rank=rank)
            elif name == "audio_embedding":
                progress_event(45, "audio_ready", "音频特征已完成，准备生成视频", rank=rank)
            elif name.startswith("segment_"):
                m = re.fullmatch(r"segment_(\d+)", name)
                if m:
                    current = int(m.group(1))
                    total = state["total_segments"] or current
                    progress_event(
                        _segment_percent(current, total),
                        "video_generation",
                        f"第 {current}/{total} 段已完成",
                        current_segment=current,
                        total_segments=total,
                        rank=rank,
                    )
            elif name == "mux":
                progress_event(95, "muxing", "视频片段已生成，正在拼接并封装音频", rank=rank)
            return elapsed
        module._log_timing = timed

    # torchrun imports this wrapper once per rank. Respect RANK for the initial
    # event too so only global rank 0 writes task-level progress.
    try:
        initial_rank = int(os.environ.get("RANK", "0"))
    except ValueError:
        initial_rank = 0
    progress_event(10, "model_loading", "正在加载模型与推理组件", rank=initial_rank)
