"""Resident inference worker for the avatar-single model.

Goal of the refactor
--------------------
Previously every ``/generate/avatar-single`` request forked a fresh
``torchrun`` subprocess that *re-loaded* the ~40GB avatar model from disk
(cold start on every request). This worker instead starts **once** at API
startup (launched by ``api.server`` via its lifespan), loads the model a
single time, and then serves requests over a local HTTP socket:

    rank0  -> runs an HTTP server (GET /health, POST /run)
    ranks>0 -> sit in a broadcast-driven loop, run inference when asked

All ranks still participate in the context-parallel forward pass exactly like
the original ``run_avatar_single`` script; the only difference is that the
model is already resident, so there is no reload per request.

Run directly (for debugging) with:
    torchrun --nproc_per_node=1 -m api.inference_worker \
        --checkpoint_dir ./weights/LongCat-Video-Avatar-1.5 \
        --context_parallel_size 1 --model_type avatar-v1.5 --avatar_mode single
"""
import argparse
import datetime
import json
import os
import sys
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import torch
import torch.distributed as dist

from longcat_video.context_parallel import context_parallel_util

from api.inference_common import load_avatar_models
from api.scripts.run_avatar_single import run_inference


# --------------------------------------------------------------------------- #
# globals populated at startup
# --------------------------------------------------------------------------- #
MODELS = None
CHECKPOINT_DIR = None
CONTEXT_PARALLEL_SIZE = 1
AVATAR_MODE = "single"
WORKER_BUSY = False
LAST_TASK_ID = None


def init_dist(context_parallel_size: int):
    """Initialise the distributed process group + context parallel. Mirrors the
    preamble of the original inference scripts (which are launched by torchrun
    and therefore already have RANK / WORLD_SIZE / MASTER_* set)."""
    rank = int(os.environ["RANK"])
    num_gpus = torch.cuda.device_count()
    local_rank = rank % num_gpus
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", timeout=datetime.timedelta(seconds=3600 * 24)
    )
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    context_parallel_util.init_context_parallel(
        context_parallel_size=context_parallel_size,
        global_rank=global_rank,
        world_size=num_processes,
    )
    return local_rank, global_rank, num_processes


def _broadcast_run(cfg: dict, output_dir: str):
    """Collective: rank0 sends the task, every rank receives it, then every
    rank runs the (context-parallel) inference. Returns the outputs ONLY on
    rank0 (followers return None)."""
    global WORKER_BUSY, LAST_TASK_ID
    payload = json.dumps({"cmd": "run", "cfg": cfg, "output_dir": output_dir})
    obj_list = [payload]
    dist.broadcast_object_list(obj_list, src=0)
    cmd = json.loads(obj_list[0])
    run_inference(MODELS, cmd["cfg"], types.SimpleNamespace(output_dir=cmd["output_dir"]))
    if dist.get_rank() == 0:
        return collect_outputs(cmd["output_dir"])
    return None


def dispatch_task(cfg: dict, output_dir: str):
    """Entry point called by rank0's HTTP handler."""
    global WORKER_BUSY, LAST_TASK_ID
    if WORKER_BUSY:
        return {"status": "failed", "error": "worker busy; please retry shortly"}
    WORKER_BUSY = True
    try:
        outputs = _broadcast_run(cfg, output_dir)
        return {"status": "done", "outputs": outputs or []}
    except Exception as e:  # noqa: BLE001 - report back to the API caller
        return {"status": "failed", "error": f"{type(e).__name__}: {e}"}
    finally:
        WORKER_BUSY = False


def follower_loop():
    """Non-rank0 ranks wait for a broadcast command and execute it. The only
    command for now is ``run``; a ``stop`` command cleanly exits."""
    while True:
        obj_list = [None]
        dist.broadcast_object_list(obj_list, src=0)
        cmd = json.loads(obj_list[0])
        if cmd.get("cmd") == "stop":
            break
        try:
            run_inference(
                MODELS, cmd["cfg"],
                types.SimpleNamespace(output_dir=cmd["output_dir"]),
            )
        except Exception as e:  # noqa: BLE001 - don't kill the worker on a bad task
            print(f"[follower] task failed: {e}")


def collect_outputs(output_dir: str):
    results = []
    for p in sorted(Path(output_dir).glob("*.mp4")):
        results.append({"filename": p.name, "size": p.stat().st_size})
    return results


# --------------------------------------------------------------------------- #
# HTTP server (rank0 only)
# --------------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, obj: dict):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):  # silence default logging
        pass

    def do_GET(self):
        if self.path in ("/health", "/status"):
            self._send_json(200, {
                "status": "ok",
                "model_loaded": MODELS is not None,
                "mode": AVATAR_MODE,
                "busy": WORKER_BUSY,
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/run":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                cfg = body.get("task_input", {})
                output_dir = body.get("output_dir")
                global LAST_TASK_ID
                LAST_TASK_ID = body.get("task_id")
                if not output_dir:
                    self._send_json(400, {"status": "failed", "error": "output_dir required"})
                    return
                os.makedirs(output_dir, exist_ok=True)
                result = dispatch_task(cfg, output_dir)
                self._send_json(200, result)
            except Exception as e:  # noqa: BLE001
                self._send_json(200, {"status": "failed", "error": f"{type(e).__name__}: {e}"})
        else:
            self._send_json(404, {"error": "not found"})


def serve(host: str, port: int):
    httpd = HTTPServer((host, port), _Handler)
    print(f"[worker rank0] HTTP server listening on {host}:{port}")
    httpd.serve_forever()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=str, required=True)
    p.add_argument("--context_parallel_size", type=int, default=1)
    p.add_argument("--model_type", type=str, default="avatar-v1.5")
    p.add_argument("--avatar_mode", type=str, default="single", choices=["single", "multi"])
    p.add_argument("--use_int8", action="store_true",
                   help="Load INT8 quantized DiT model for reduced VRAM usage")
    p.add_argument("--use_distill", action="store_true")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=29500)
    return p.parse_args()


def main():
    args = parse_args()
    if args.avatar_mode != "single":
        # v1 supports in-process only for avatar-single; multi / video fall
        # back to the subprocess path in the API.
        print("[worker] ERROR: only --avatar_mode single is supported in-process.")
        raise SystemExit(1)

    global CHECKPOINT_DIR, CONTEXT_PARALLEL_SIZE, AVATAR_MODE, MODELS
    CHECKPOINT_DIR = args.checkpoint_dir
    CONTEXT_PARALLEL_SIZE = args.context_parallel_size
    AVATAR_MODE = args.avatar_mode

    local_rank, global_rank, _ = init_dist(args.context_parallel_size)
    cp_size = context_parallel_util.get_cp_size()
    cp_split_hw = context_parallel_util.get_optimal_split(cp_size)

    MODELS = load_avatar_models(
        checkpoint_dir=args.checkpoint_dir,
        model_type=args.model_type,
        use_int8=args.use_int8,
        use_distill=args.use_distill,
        local_rank=local_rank,
        cp_split_hw=cp_split_hw,
        avatar_mode=args.avatar_mode,
    )
    print(f"[worker rank{global_rank}] models loaded, ready to serve.")

    if global_rank == 0:
        serve(args.host, args.port)
    else:
        follower_loop()


if __name__ == "__main__":
    main()
