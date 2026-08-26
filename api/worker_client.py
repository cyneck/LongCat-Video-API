"""Thin HTTP client used by the FastAPI service to talk to the resident
``inference_worker`` (a long-lived torchrun process that keeps the avatar
model loaded in memory).

The worker speaks a tiny JSON protocol:
    GET  /health   -> {"status": "ok", "model_loaded": bool, "mode": str, "busy": bool}
    GET  /status   -> same as /health
    POST /run      -> body {"task_id", "task_input", "output_dir", "checkpoint_dir"}
                      -> {"status": "done",   "outputs": [{filename, size}, ...]}
                      -> {"status": "failed", "error": str}
"""
import json
import urllib.request


class WorkerClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 29500, timeout: int = 7200):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    def health(self) -> dict:
        try:
            with urllib.request.urlopen(self.base + "/health", timeout=10) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001 - report as unreachable
            return {"status": "unreachable", "error": str(e)}

    def submit(self, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base + "/run",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8"))
