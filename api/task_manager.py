"""Async task manager.

Each generation request is materialised as a ``task_input.json`` + ``--output_dir``
and executed via a ``torchrun`` subprocess. A bounded semaphore serialises GPU
access so tasks don't OOM each other. While a subprocess is running, its existing
stage log markers are translated into monotonic user-facing progress.
"""
import json
import asyncio
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from . import config
from .progress import ProgressSnapshot, parse_progress


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Task:
    task_id: str
    task_type: str
    status: TaskStatus = TaskStatus.PENDING
    output_dir: str = ""
    input_path: str = ""
    created_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    return_code: Optional[int] = None
    error: str = ""
    request_params: dict = field(default_factory=dict)
    execution_params: dict = field(default_factory=dict)
    outputs: list = field(default_factory=list)
    result: dict = field(default_factory=dict)
    # Stage-based progress. These fields are persisted so API/H5/history survive
    # API restarts and historical records remain backward compatible.
    progress_percent: int = 0
    progress_stage: str = "排队中"
    progress_detail: str = "等待 GPU 执行资源"
    current_segment: int = 0
    total_segments: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["progress"] = {
            "percent": self.progress_percent,
            "stage": self.progress_stage,
            "detail": self.progress_detail,
            "current_segment": self.current_segment,
            "total_segments": self.total_segments,
        }
        return d


class TaskManager:
    def __init__(self):
        config.ensure_dirs()
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()
        self._gpu_sem = asyncio.Semaphore(config.GPU_CONCURRENCY)
        self._load()

    def _load(self):
        if config.TASK_DB.exists():
            try:
                data = json.loads(config.TASK_DB.read_text(encoding="utf-8"))
                for t in data.get("tasks", []):
                    # ``progress`` is an API convenience object, not a dataclass
                    # constructor field. Historical tasks simply use defaults.
                    known = {k: v for k, v in t.items() if k in Task.__dataclass_fields__}
                    task = Task(**known)
                    if isinstance(task.status, str):
                        task.status = TaskStatus(task.status)
                    if not task.execution_params and task.input_path:
                        try:
                            task.execution_params = json.loads(
                                Path(task.input_path).read_text(encoding="utf-8")
                            )
                        except Exception:
                            pass
                    self._tasks[task.task_id] = task
            except Exception:
                pass

    def _persist(self):
        with self._lock:
            data = {"tasks": [t.to_dict() for t in self._tasks.values()]}
        tmp = config.TASK_DB.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(config.TASK_DB)

    def get(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def list(self, limit: int = 100) -> list:
        items = sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)
        return [t.to_dict() for t in items[:limit]]

    def create(
        self,
        task_type: str,
        task_input: dict,
        request_params: Optional[dict] = None,
    ) -> Task:
        task_id = uuid.uuid4().hex[:16]
        output_dir = config.OUTPUT_DIR / task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = output_dir / "task_input.json"
        input_path.write_text(json.dumps(task_input, ensure_ascii=False, indent=2), encoding="utf-8")

        task = Task(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.PENDING,
            output_dir=str(output_dir),
            input_path=str(input_path),
            created_at=time.time(),
            request_params=dict(request_params or task_input),
            execution_params=dict(task_input),
            progress_percent=0,
            progress_stage="排队中",
            progress_detail="等待 GPU 执行资源",
        )
        with self._lock:
            self._tasks[task_id] = task
        self._persist()
        return task

    def delete(self, task_id: str) -> Task:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(task_id)
            if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                raise RuntimeError(f"cannot delete task while status={task.status.value}")
            del self._tasks[task_id]

        self._persist()
        out_dir = Path(task.output_dir).resolve()
        output_root = config.OUTPUT_DIR.resolve()
        try:
            out_dir.relative_to(output_root)
        except ValueError as exc:
            raise RuntimeError(f"refusing to delete output path outside task root: {out_dir}") from exc
        if out_dir.exists():
            shutil.rmtree(out_dir)

        log_path = (config.LOG_DIR / f"{task.task_id}.log").resolve()
        try:
            log_path.relative_to(config.LOG_DIR.resolve())
        except ValueError as exc:
            raise RuntimeError(f"refusing to delete log outside log root: {log_path}") from exc
        if log_path.exists():
            log_path.unlink()
        return task

    def schedule(self, task: Task):
        loop = asyncio.get_running_loop()
        loop.create_task(self._run(task))

    def _set_progress(self, task: Task, snap: ProgressSnapshot) -> bool:
        new_values = (
            max(0, min(100, int(snap.percent))),
            snap.stage,
            snap.detail,
            int(snap.current_segment),
            int(snap.total_segments),
        )
        old_values = (
            task.progress_percent,
            task.progress_stage,
            task.progress_detail,
            task.current_segment,
            task.total_segments,
        )
        if new_values == old_values:
            return False
        (
            task.progress_percent,
            task.progress_stage,
            task.progress_detail,
            task.current_segment,
            task.total_segments,
        ) = new_values
        return True

    def _refresh_progress_from_log(self, task: Task, log_path: Path):
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return
        fallback = ProgressSnapshot(
            task.progress_percent,
            task.progress_stage,
            task.progress_detail,
            task.current_segment,
            task.total_segments,
        )
        snap = parse_progress(text, fallback=fallback)
        if self._set_progress(task, snap):
            self._persist()

    async def _wait_with_progress(self, proc, task: Task, log_path: Path) -> int:
        waiter = asyncio.create_task(proc.wait())
        while not waiter.done():
            await asyncio.sleep(1.0)
            self._refresh_progress_from_log(task, log_path)
        code = await waiter
        self._refresh_progress_from_log(task, log_path)
        return code

    async def _run(self, task: Task):
        async with self._gpu_sem:
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            self._set_progress(task, ProgressSnapshot(5, "启动任务", "正在启动推理进程"))
            self._persist()

            script = config.SCRIPTS.get(task.task_type)
            if not script:
                task.status = TaskStatus.FAILED
                task.error = f"Unknown task type: {task.task_type}"
                task.finished_at = time.time()
                self._set_progress(task, ProgressSnapshot(task.progress_percent, "任务失败", task.error))
                self._persist()
                return

            model_type = None
            if task.task_type in ("avatar_single", "avatar_multi"):
                model_type = task.execution_params.get("model_type")
                if model_type is None:
                    try:
                        data = json.loads(Path(task.input_path).read_text(encoding="utf-8"))
                        model_type = data.get("model_type")
                    except Exception:
                        model_type = None
            checkpoint = config.checkpoint_for_task(task.task_type, model_type)

            log_path = config.LOG_DIR / f"{task.task_id}.log"
            cmd = self._build_cmd(script, checkpoint, task)
            env = os.environ.copy()
            existing_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join(
                [str(config.REPO_ROOT)] + ([existing_pp] if existing_pp else [])
            )
            env["LONGCAT_CHECKPOINT_DIR_VIDEO"] = config.CHECKPOINT_DIR_VIDEO

            try:
                with open(log_path, "w", encoding="utf-8") as logf:
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=logf,
                        stderr=logf,
                        cwd=str(config.REPO_ROOT),
                        env=env,
                    )
                    task.return_code = await self._wait_with_progress(proc, task, log_path)
            except Exception as e:
                task.return_code = -1
                task.error = f"failed to launch subprocess: {e}"

            task.finished_at = time.time()
            if task.return_code == 0:
                task.status = TaskStatus.DONE
                task.outputs = self._collect_outputs(task)
                task.result = {
                    "status": "success",
                    "outputs": task.outputs,
                    "output_count": len(task.outputs),
                }
                self._set_progress(task, ProgressSnapshot(100, "已完成", "视频生成完成"))
            else:
                task.status = TaskStatus.FAILED
                task.error = self._tail_error(log_path)
                task.result = {"status": "failed", "error": task.error}
                self._set_progress(
                    task,
                    ProgressSnapshot(task.progress_percent, "任务失败", "请查看运行日志"),
                )
            self._persist()

    def _build_cmd(self, script: Path, checkpoint: str, task: Task) -> list:
        nproc = max(1, config.NUM_GPUS)
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc}",
            "--nnodes=1",
            "--master_port=29501",
            str(script),
            "--task_input", task.input_path,
            "--output_dir", task.output_dir,
            "--checkpoint_dir", checkpoint,
            "--context_parallel_size", str(config.CONTEXT_PARALLEL_SIZE),
        ]
        if config.ENABLE_COMPILE and task.task_type in (
            "text_to_video", "image_to_video", "video_continuation"
        ):
            cmd.append("--enable_compile")
        return cmd

    def _collect_outputs(self, task: Task) -> list:
        out_dir = Path(task.output_dir)
        results = []
        for p in sorted(out_dir.glob("*.mp4")):
            stat = p.stat()
            results.append({
                "filename": p.name,
                "size": stat.st_size,
                "created_at": stat.st_mtime,
                "download_url": f"/tasks/{task.task_id}/files/{p.name}",
            })
        return results

    def _tail_error(self, log_path: Path, n: int = 40) -> str:
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-n:])
        except Exception as e:
            return f"(unreadable log {log_path}: {e})"


manager = TaskManager()
