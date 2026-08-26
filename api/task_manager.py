"""Async task manager.

Each generation request is materialised as a ``task_input.json`` + ``--output_dir``
and executed via a ``torchrun`` subprocess (the underlying scripts require a
distributed launch because they call ``dist.init_process_group``). A bounded
semaphore serialises GPU access so tasks don't OOM each other.
"""
import json
import asyncio
import os
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
    outputs: list = field(default_factory=list)  # list of {filename, size, download_url}

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
                    task = Task(**{k: v for k, v in t.items() if k in Task.__dataclass_fields__})
                    if isinstance(task.status, str):
                        task.status = TaskStatus(task.status)
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

    def create(self, task_type: str, task_input: dict) -> Task:
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
        )
        with self._lock:
            self._tasks[task_id] = task
        self._persist()
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
            checkpoint = config.TASK_CHECKPOINT.get(task.task_type)
            if not script or not checkpoint:
                task.status = TaskStatus.FAILED
                task.error = f"Unknown task type: {task.task_type}"
                task.finished_at = time.time()
                self._persist()
                return

            # ---- in-process (resident worker) fast path ----------------------
            # avatar-single requests are served by the long-lived worker that
            # already has the model in memory (no reload per request). The
            # worker returns a list of {filename, size}; map them to download
            # entries the same way the subprocess path does.
            if config.INPROCESS and task.task_type == "avatar_single" and config.WORKER_CLIENT is not None:
                try:
                    with open(task.input_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                    payload = {
                        "task_id": task.task_id,
                        "task_input": cfg,
                        "output_dir": task.output_dir,
                        "checkpoint_dir": checkpoint,
                    }
                    resp = await asyncio.to_thread(config.WORKER_CLIENT.submit, payload)
                    if resp.get("status") == "done":
                        task.status = TaskStatus.DONE
                        task.outputs = self._attach_urls(task, resp.get("outputs", []))
                    else:
                        task.status = TaskStatus.FAILED
                        task.error = resp.get("error", "worker returned status=failed")
                except Exception as e:  # noqa: BLE001
                    task.status = TaskStatus.FAILED
                    task.error = f"worker submit failed: {e}"
                task.finished_at = time.time()
                self._persist()
                return

            log_path = config.LOG_DIR / f"{task.task_id}.log"
            cmd = self._build_cmd(script, checkpoint, task)

            # The inference scripts live under api/scripts/ but import the
            # repo-root package `longcat_video`. Python only puts the *script's*
            # own directory on sys.path[0], so we must add REPO_ROOT to
            # PYTHONPATH for the subprocess (and the torchrun workers it spawns).
            env = os.environ.copy()
            existing_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join(
                [str(config.REPO_ROOT)] + ([existing_pp] if existing_pp else [])
            )

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
            else:
                task.status = TaskStatus.FAILED
                task.error = self._tail_error(log_path)
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
        # only video tasks honour --enable_compile (avatar scripts have no such arg)
        if config.ENABLE_COMPILE and task.task_type in ("text_to_video", "image_to_video", "video_continuation"):
            cmd.append("--enable_compile")
        return cmd

    def _collect_outputs(self, task: Task) -> list:
        out_dir = Path(task.output_dir)
        results = []
        for p in sorted(out_dir.glob("*.mp4")):
            results.append({
                "filename": p.name,
                "size": p.stat().st_size,
                "download_url": f"/tasks/{task.task_id}/files/{p.name}",
            })
        return results

    def _attach_urls(self, task: Task, outputs: list) -> list:
        """Map worker-returned {filename, size} entries to download entries."""
        results = []
        for o in outputs:
            fn = o.get("filename")
            if not fn:
                continue
            results.append({
                "filename": fn,
                "size": o.get("size", 0),
                "download_url": f"/tasks/{task.task_id}/files/{fn}",
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
