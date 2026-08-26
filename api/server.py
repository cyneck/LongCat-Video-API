"""FastAPI service that wraps the LongCat-Video inference scripts."""
import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from . import config
from . import schemas
from .task_manager import manager


app = FastAPI(title="LongCat-Video API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup_weight_check():
    import logging
    logger = logging.getLogger("longcat")
    ok_v, pv = config.check_weights(None, None)
    if not ok_v:
        logger.warning("LongCat 基础视频模型检查未通过:\n%s", "\n".join(pv))
    for mt in ("avatar-v1.0", "avatar-v1.5"):
        ok, problems = config.check_weights(mt, "avatar_single", skip_missing_dirs=True)
        if not ok:
            logger.warning("LongCat 权重检查未通过 [%s]:\n%s", mt, "\n".join(problems))


@app.middleware("http")
async def auth_gate(request: Request, call_next):
    if not config.AUTH_ENABLED:
        return await call_next(request)
    path = request.url.path
    if path in (config.HEALTH_PATH, "/login", "/auth/login"):
        return await call_next(request)
    if request.cookies.get("lc_token") != config.AUTH_TOKEN:
        if path == "/" or path.startswith("/h5"):
            return RedirectResponse("/login")
        return JSONResponse(status_code=401, content={"detail": "unauthorized"})
    return await call_next(request)


def _service_info():
    return {
        "service": "LongCat-Video API",
        "version": "0.1.0",
        "num_gpus": config.NUM_GPUS,
        "gpu_concurrency": config.GPU_CONCURRENCY,
        "auth": config.AUTH_ENABLED,
        "a100_40g_profile": config.LOW_VRAM_PROFILE_ENABLED,
        "force_use_int8": config.LOW_VRAM.get("force_use_int8", False),
        "force_use_distill": config.LOW_VRAM.get("force_use_distill", False),
        "endpoints": [
            "/files/image", "/files/video", "/files/audio", "/files/json",
            "/generate/text-to-video", "/generate/image-to-video",
            "/generate/video-continuation", "/generate/avatar-single",
            "/generate/avatar-multi", "/tasks", "/tasks/{id}",
            "DELETE /tasks/{id}", "/tasks/{id}/files/{filename}",
            "/tasks/{id}/log",
        ],
    }


@app.get(config.HEALTH_PATH)
def health():
    return _service_info()


def _save_upload(upload: UploadFile, allowed: set, subdir: str) -> str:
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"unsupported extension '{ext}'. allowed: {sorted(allowed)}")
    dest_dir = config.UPLOAD_DIR / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{uuid.uuid4().hex[:12]}{ext}"
    dest = dest_dir / stored_name
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return str(dest.resolve())


def _resolve_upload_path(p: str) -> str:
    abs_p = Path(p).resolve()
    try:
        abs_p.relative_to(config.UPLOAD_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"path not under uploads dir: {p}")
    if not abs_p.exists():
        raise HTTPException(status_code=400, detail=f"referenced file does not exist: {p}")
    return str(abs_p)


async def _raw_request_params(request: Request, fallback: dict) -> dict:
    try:
        payload = await request.json()
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return dict(fallback)


def _enqueue(task_type: str, task_input: dict, request_params: dict | None = None):
    task = manager.create(task_type, task_input, request_params=request_params)
    manager.schedule(task)
    return JSONResponse(status_code=202, content={"task_id": task.task_id, "status": task.status.value})


@app.get("/")
def root():
    index_path = config.H5_DIR / "index.html"
    progress_path = config.H5_DIR / "progress.js"
    if config.EMBED_H5 and index_path.exists():
        html = index_path.read_text(encoding="utf-8")
        if progress_path.exists() and "/h5/progress.js" not in html:
            html = html.replace("</body>", '<script src="/h5/progress.js"></script>\n</body>')
        return HTMLResponse(
            html,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )
    return _service_info()


@app.get("/h5/progress.js")
def h5_progress_script():
    path = config.H5_DIR / "progress.js"
    if not path.exists():
        raise HTTPException(status_code=404, detail="progress renderer not found")
    return FileResponse(
        str(path),
        media_type="application/javascript",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/login")
def login_page():
    if not config.EMBED_H5 or not (config.H5_DIR / "login.html").exists():
        return HTMLResponse("<p>未启用 H5 嵌入（请设置 LONGCAT_EMBED_H5=1）。</p>")
    return FileResponse(str(config.H5_DIR / "login.html"))


@app.post("/auth/login")
def do_login(username: str = Form(...), password: str = Form(...)):
    if username == config.AUTH_USER and password == config.AUTH_PASS:
        resp = RedirectResponse("/", status_code=303)
        resp.set_cookie("lc_token", config.AUTH_TOKEN, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 7)
        return resp
    return HTMLResponse(
        "<p style='color:#e54848'>账号或密码错误</p><p><a href='/login'>返回重试</a></p>",
        status_code=401,
    )


@app.post("/files/image")
async def upload_image(file: UploadFile = File(...)):
    path = _save_upload(file, config.ALLOWED_IMAGE_EXT, "images")
    return {"path": path, "filename": Path(path).name, "size": Path(path).stat().st_size}


@app.post("/files/video")
async def upload_video(file: UploadFile = File(...)):
    path = _save_upload(file, config.ALLOWED_VIDEO_EXT, "videos")
    return {"path": path, "filename": Path(path).name, "size": Path(path).stat().st_size}


@app.post("/files/audio")
async def upload_audio(file: UploadFile = File(...)):
    path = _save_upload(file, config.ALLOWED_AUDIO_EXT, "audios")
    return {"path": path, "filename": Path(path).name, "size": Path(path).stat().st_size}


@app.post("/files/json")
async def upload_json(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        import json
        data = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid json: {e}")
    dest_dir = config.UPLOAD_DIR / "jsons"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{uuid.uuid4().hex[:12]}.json"
    dest.write_bytes(raw)
    return {"path": str(dest.resolve()), "filename": dest.name, "preview": data}


@app.post("/generate/text-to-video")
async def gen_t2v(request: Request, req: schemas.TextToVideoRequest):
    normalized = req.model_dump()
    request_params = await _raw_request_params(request, normalized)
    return _enqueue("text_to_video", dict(normalized), request_params)


@app.post("/generate/image-to-video")
async def gen_i2v(request: Request, req: schemas.ImageToVideoRequest):
    normalized = req.model_dump()
    request_params = await _raw_request_params(request, normalized)
    execution = dict(normalized)
    execution["cond_image"] = _resolve_upload_path(execution["cond_image"])
    return _enqueue("image_to_video", execution, request_params)


@app.post("/generate/video-continuation")
async def gen_vc(request: Request, req: schemas.VideoContinuationRequest):
    normalized = req.model_dump()
    request_params = await _raw_request_params(request, normalized)
    execution = dict(normalized)
    execution["cond_video"] = _resolve_upload_path(execution["cond_video"])
    return _enqueue("video_continuation", execution, request_params)


@app.post("/generate/avatar-single")
async def gen_avatar_single(request: Request, req: schemas.AvatarSingleRequest):
    normalized = req.model_dump()
    request_params = await _raw_request_params(request, normalized)
    execution = dict(normalized)
    execution["cond_audio"] = {k: _resolve_upload_path(v) for k, v in normalized["cond_audio"].items()}
    if normalized.get("cond_image"):
        execution["cond_image"] = _resolve_upload_path(normalized["cond_image"])
    if execution["stage_1"] == "ai2v" and not execution.get("cond_image"):
        raise HTTPException(status_code=400, detail="cond_image is required when stage_1='ai2v'")
    ok, problems = config.check_weights(execution["model_type"], "avatar_single")
    if not ok:
        raise HTTPException(status_code=400, detail="数字人权重未就绪，无法执行:\n" + "\n".join(problems))
    return _enqueue("avatar_single", execution, request_params)


@app.post("/generate/avatar-multi")
async def gen_avatar_multi(request: Request, req: schemas.AvatarMultiRequest):
    normalized = req.model_dump()
    request_params = await _raw_request_params(request, normalized)
    execution = dict(normalized)
    execution["cond_image"] = _resolve_upload_path(normalized["cond_image"])
    execution["cond_audio"] = {k: _resolve_upload_path(v) for k, v in normalized["cond_audio"].items()}
    if not any(execution["cond_audio"].values()):
        raise HTTPException(status_code=400, detail="at least one of person1/person2 audio is required")
    ok, problems = config.check_weights(execution["model_type"], "avatar_multi")
    if not ok:
        raise HTTPException(status_code=400, detail="数字人权重未就绪，无法执行:\n" + "\n".join(problems))
    return _enqueue("avatar_multi", execution, request_params)


@app.get("/tasks")
def list_tasks(limit: int = 100):
    return {"tasks": manager.list(limit)}


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    task = manager.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return task.to_dict()


@app.delete("/tasks/{task_id}")
def delete_task(task_id: str):
    try:
        task = manager.delete(task_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="task not found")
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {
        "deleted": True,
        "task_id": task.task_id,
        "task_type": task.task_type,
        "status": task.status.value,
        "deleted_artifacts": True,
        "uploaded_sources_retained": True,
    }


@app.get("/tasks/{task_id}/files/{filename}")
def download_task_file(task_id: str, filename: str):
    task = manager.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="bad filename")
    fpath = Path(task.output_dir) / filename
    if not fpath.exists():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(str(fpath), filename=filename)


@app.get("/tasks/{task_id}/log")
def task_log(task_id: str):
    task = manager.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    log_path = config.LOG_DIR / f"{task.task_id}.log"
    if not log_path.exists():
        return {"log": ""}
    return {"log": log_path.read_text(encoding="utf-8", errors="replace")}


def main():
    import argparse
    import uvicorn

    p = argparse.ArgumentParser(
        prog="python -m api.server",
        description="LongCat-Video API 服务（鉴权参数与 LONGCAT_* 环境变量等价，命令行优先）",
    )
    p.add_argument("--host", default=None, help="监听地址（默认取 LONGCAT_HOST / 0.0.0.0）")
    p.add_argument("--port", type=int, default=None, help="监听端口（默认取 LONGCAT_PORT / 8000）")
    p.add_argument("--auth", action="store_true", help="开启鉴权门禁（等价 LONGCAT_AUTH=1）")
    p.add_argument("--user", default=None, help="鉴权账号（默认 admin）")
    p.add_argument("--pass", "--password", dest="password", default=None, help="鉴权密码（默认 admin）")
    p.add_argument("--token", default=None, help="登录 cookie token（默认 longcat-demo-token）")
    p.add_argument("--embed-h5", action="store_true", help="嵌入 H5 页面与登录页（等价 LONGCAT_EMBED_H5=1）")
    p.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = p.parse_args()

    if args.auth:
        os.environ["LONGCAT_AUTH"] = "1"
    if args.embed_h5:
        os.environ["LONGCAT_EMBED_H5"] = "1"
    if args.user is not None:
        os.environ["LONGCAT_USER"] = args.user
    if args.password is not None:
        os.environ["LONGCAT_PASS"] = args.password
    if args.token is not None:
        os.environ["LONGCAT_AUTH_TOKEN"] = args.token

    config.AUTH_ENABLED = os.environ.get("LONGCAT_AUTH", "0") == "1"
    config.EMBED_H5 = os.environ.get("LONGCAT_EMBED_H5", "0") == "1"
    config.AUTH_USER = os.environ.get("LONGCAT_USER", config.AUTH_USER)
    config.AUTH_PASS = os.environ.get("LONGCAT_PASS", config.AUTH_PASS)
    config.AUTH_TOKEN = os.environ.get("LONGCAT_AUTH_TOKEN", config.AUTH_TOKEN)

    try:
        config.validate_auth()
    except RuntimeError as exc:
        import sys
        print(f"[longcat][auth] 启动失败: {exc}", file=sys.stderr, flush=True)
        sys.exit(2)

    if config.AUTH_ENABLED:
        print(
            f"[longcat] 鉴权已开启: 账号={config.AUTH_USER!r} 密码={config.AUTH_PASS!r}\n"
            f"[longcat]   - H5 访问入口 /login（需 LONGCAT_EMBED_H5=1 或 --embed-h5）\n"
            f"[longcat]   - API 调用: POST /auth/login 拿 cookie（账号密码表单）后携带访问",
            flush=True,
        )

    uvicorn.run(
        "api.server:app",
        host=args.host or config.HOST,
        port=args.port or config.PORT,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
