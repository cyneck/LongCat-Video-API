"""Translate inference logs into coarse, user-facing task progress.

Progress is intentionally stage-based rather than pretending to know denoising ETA.
The parser is monotonic and understands the log markers emitted by current Avatar
single/multi workers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


_SEGMENT_START_RE = re.compile(r"Generating segment\s+(\d+)/(\d+)", re.I)
_SEGMENT_DONE_RE = re.compile(r"\[longcat\]\[timing\]\s+segment_(\d+)=", re.I)


@dataclass(frozen=True)
class ProgressSnapshot:
    percent: int
    stage: str
    detail: str = ""
    current_segment: int = 0
    total_segments: int = 0


def _segment_percent(current: int, total: int, completed: bool = False) -> int:
    total = max(1, total)
    current = max(1, min(current, total))
    # Generation owns 45%..90%. Starting a segment reports its left edge;
    # finishing it reports its right edge.
    pos = current if completed else current - 1
    return min(90, 45 + round(45 * pos / total))


def parse_progress(log_text: str, fallback: ProgressSnapshot | None = None) -> ProgressSnapshot:
    """Return the latest stage represented in ``log_text``.

    The result never moves backwards relative to ``fallback``.
    """
    prev = fallback or ProgressSnapshot(5, "启动任务", "正在启动推理进程")
    best = prev

    def set_progress(percent: int, stage: str, detail: str = "", current: int = 0, total: int = 0):
        nonlocal best
        if percent >= best.percent:
            best = ProgressSnapshot(percent, stage, detail, current, total)

    text = log_text or ""
    if not text:
        return best

    if any(marker in text for marker in ("Loading checkpoint shards", "Loading INT8 quantized DiT", "[INFO] Loading INT8")):
        set_progress(10, "加载模型", "正在加载 DiT / VAE / 文本与音频组件")

    if "[longcat][timing] model_load=" in text:
        set_progress(25, "模型就绪", "模型已加载，准备音频预处理")

    if any(marker in text for marker in ("Loading model Kim_Vocal", "vocal_separator", "Starting audio separation", "Audio separation")):
        set_progress(30, "人声分离", "正在提取数字人驱动人声")

    if "[longcat][timing] vocal_separation=" in text:
        set_progress(35, "人声分离完成", "正在提取音频特征")

    if "[longcat][audio]" in text or "[longcat][timing] audio_embedding=" in text:
        set_progress(45, "音频特征完成", "准备开始视频生成")

    for m in _SEGMENT_START_RE.finditer(text):
        current, total = int(m.group(1)), int(m.group(2))
        set_progress(
            _segment_percent(current, total, completed=False),
            "生成视频",
            f"正在生成第 {current}/{total} 段",
            current,
            total,
        )

    # A completed segment takes precedence over its corresponding start marker.
    # Determine total from the latest start marker when available.
    starts = list(_SEGMENT_START_RE.finditer(text))
    inferred_total = int(starts[-1].group(2)) if starts else 0
    for m in _SEGMENT_DONE_RE.finditer(text):
        current = int(m.group(1))
        total = inferred_total or max(current, best.total_segments or current)
        set_progress(
            _segment_percent(current, total, completed=True),
            "生成视频",
            f"第 {current}/{total} 段已完成",
            current,
            total,
        )

    if any(marker in text for marker in ("ffmpeg", "[longcat][timing] mux=")):
        set_progress(95, "合并封装", "正在拼接视频并封装音频")

    return best
