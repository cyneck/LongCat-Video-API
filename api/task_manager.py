"""Async task manager.

Each generation request is materialised as a ``task_input.json`` + ``--output_dir``
and executed via a ``torchrun`` subprocess (the underlying scripts require a
distributed launch because they call ``dist.init_process_group``). A bounded
semaphore serialises GPU access so tasks don't OOM each other.
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
    # Exact JSON payload received from the client, before Pydantic/profile/path
    # normalization. This is the audit record of what the caller actually sent.
    request_params: dict = field(default_factory=dict)
    # Normalized payload actually written to task_input.json and consumed by the
    # inference subprocess. Keeping both removes ambiguity during troubleshooting.
    execution_params: dict = field(default_factory=dict)
    outputs: list = field(default_factory=list)  # list of produced file metadata
    result: dict = field(default_factory=dict)   # extensible generation result summary

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class TaskManager:
    def __init__(self):
        config.ensure_dirs()
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()
        # serialise GPU usage across tasks
        self._gpu_sem = asyncio.Semaphore(config.GPU_CONCURRENCY)
        self._load()

    # ---- persistence (simple json store) ----
    def _load(self):
        if config.TASK_DB.exists():
            try:
                data = json.loads(config.TASK_DB.read_text(encoding="utf-8"))
                for t in data.get("tasks", []):
                    # Backward compatible with old tasks.json records: newly added
                    # dataclass fields use their defaults when absent.
                    known = {k: v for k, v in t.items() if k in Task.__dataclass_fields__}
                    task = Task(**known)
                    if isinstance(task.status, str):
                        task.status = TaskStatus(task.status)
                    # Recover execution parameters for historical tasks when the
                    # per-task input file still exists.
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

    # ---- public API ----
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
        # absolute paths help the subprocess resolve uploads reliably
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
        )
        with self._lock:
            self._tasks[task_id] = task
        self._persist()
        return task

    def delete(self, task_id: str) -> Task:
        """Delete a completed/failed task and its generated artifacts/log.

        Uploaded source assets are deliberately retained because they may be
        referenced by other tasks. Pending/running tasks cannot be deleted to
        avoid racing an active subprocess or leaving GPU work orphaned.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(task_id)
            if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                raise RuntimeError(f"cannot delete task while status={task.status.value}")
            del self._tasks[task_id]

        # Persist metadata removal first. If file cleanup later fails, the task is
        # no longer exposed through the API, and cleanup can be retried manually.
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
        """Enqueue the task on the running event loop (must be called from
        within an async endpoint, i.e. on the main event-loop thread)."""
        loop = asyncio.get_running_loop()
        loop.create_task(self._run(task))

    # ---- execution ----
    async def _run(self, task: Task):
        async with self._gpu_sem:
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            self._persist()

            script = config.SCRIPTS.get(task.task_type)
            if not script:
                task.status = TaskStatus.FAILED
                task.error = f"Unknown task type: {task.task_type}"
                task.finished_at = time.time()
                self._persist()
                return

            # Resolve the checkpoint dir in a version-aware way for avatar tasks.
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

            # The inference scripts live under api/scripts/ but import the
            # repo-root package `longcat_video`. Python only puts the script's
            # directory on sys.path[0], so add REPO_ROOT to PYTHONPATH.
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
                    task.return_code = await proc.wait()
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
            else:
                task.status = TaskStatus.FAILED
                task.error = self._tail_error(log_path)
                task.result = {
                    "status": "failed",
                    "error": task.error,
                }
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


# module-level singleton
manager = TaskManager()
