"""FastAPI service that wraps the LongCat-Video inference scripts.

Run with:
    uvicorn api.server:app --host 0.0.0.0 --port 8000
or
    python -m api.server

Endpoints:
    GET  /                       health / info
    POST /files/image            upload an image
    POST /files/video            upload a video
    POST /files/audio            upload an audio clip
    POST /files/json             upload a raw avatar input json (advanced)
    POST /generate/text-to-video
    POST /generate/image-to-video
    POST /generate/video-continuation
    POST /generate/avatar-single
    POST /generate/avatar-multi
    GET  /tasks                  list tasks
    GET  /tasks/{id}             task status / result
    GET  /tasks/{id}/files/{fn}  download a produced file
    GET  /tasks/{id}/log         tail the subprocess log
"""
import os
import shutil
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request, Form
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from . import config
from . import schemas
from .task_manager import manager, TaskStatus


app = FastAPI(title="LongCat-Video API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# optional: simple login gate + H5 embedding
# --------------------------------------------------------------------------- #
@app.middleware("http")
async def auth_gate(request: Request, call_next):
    """Block every route (except health/login) behind a cookie when AUTH_ENABLED."""
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
        "endpoints": [
            "/files/image", "/files/video", "/files/audio", "/files/json",
            "/generate/text-to-video", "/generate/image-to-video",
            "/generate/video-continuation", "/generate/avatar-single",
            "/generate/avatar-multi",
            "/tasks", "/tasks/{id}", "/tasks/{id}/files/{filename}", "/tasks/{id}/log",
        ],
    }


@app.get(config.HEALTH_PATH)
def health():
    return _service_info()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
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
    # return absolute path so subprocess can resolve it from repo root
    return str(dest.resolve())


def _resolve_upload_path(p: str) -> str:
    """Validate that a referenced upload path lives under the upload dir."""
    abs_p = Path(p).resolve()
    try:
        abs_p.relative_to(config.UPLOAD_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail=f"path not under uploads dir: {p}")
    if not abs_p.exists():
        raise HTTPException(status_code=400, detail=f"referenced file does not exist: {p}")
    return str(abs_p)


def _enqueue(task_type: str, task_input: dict):
    task = manager.create(task_type, task_input)
    manager.schedule(task)
    return JSONResponse(status_code=202, content={"task_id": task.task_id, "status": task.status.value})


# --------------------------------------------------------------------------- #
# health / info
# --------------------------------------------------------------------------- #
@app.get("/")
def root():
    if config.EMBED_H5 and (config.H5_DIR / "index.html").exists():
        return FileResponse(str(config.H5_DIR / "index.html"))
    return _service_info()


@app.get("/login")
def login_page():
    if not config.EMBED_H5 or not (config.H5_DIR / "login.html").exists():
        return HTMLResponse("<p>未启用 H5 嵌入（请设置 LONGCAT_EMBED_H5=1）。</p>")
    return FileResponse(str(config.H5_DIR / "login.html"))


@app.post("/auth/login")
def do_login(username: str = Form(...), password: str = Form(...)):
    if username == config.AUTH_USER and password == config.AUTH_PASS:
        resp = RedirectResponse("/", status_code=303)
        resp.set_cookie(
            "lc_token", config.AUTH_TOKEN,
            httponly=True, samesite="lax", max_age=60 * 60 * 24 * 7,
        )
        return resp
    return HTMLResponse(
        "<p style='color:#e54848'>账号或密码错误</p><p><a href='/login'>返回重试</a></p>",
        status_code=401,
    )


# --------------------------------------------------------------------------- #
# file uploads
# --------------------------------------------------------------------------- #
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
    """Advanced: upload a pre-built avatar input json and use it directly
    with the avatar generation endpoints by passing its content via the body.
    """
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


# --------------------------------------------------------------------------- #
# generation endpoints
# --------------------------------------------------------------------------- #
@app.post("/generate/text-to-video")
async def gen_t2v(req: schemas.TextToVideoRequest):
    return _enqueue("text_to_video", req.model_dump())


@app.post("/generate/image-to-video")
async def gen_i2v(req: schemas.ImageToVideoRequest):
    req.cond_image = _resolve_upload_path(req.cond_image)
    return _enqueue("image_to_video", req.model_dump())


@app.post("/generate/video-continuation")
async def gen_vc(req: schemas.VideoContinuationRequest):
    req.cond_video = _resolve_upload_path(req.cond_video)
    return _enqueue("video_continuation", req.model_dump())


@app.post("/generate/avatar-single")
async def gen_avatar_single(req: schemas.AvatarSingleRequest):
    resolved = {k: _resolve_upload_path(v) for k, v in req.cond_audio.items()}
    req.cond_audio = resolved
    if req.cond_image:
        req.cond_image = _resolve_upload_path(req.cond_image)
    if req.stage_1 == "ai2v" and not req.cond_image:
        raise HTTPException(status_code=400, detail="cond_image is required when stage_1='ai2v'")
    return _enqueue("avatar_single", req.model_dump())


@app.post("/generate/avatar-multi")
async def gen_avatar_multi(req: schemas.AvatarMultiRequest):
    req.cond_image = _resolve_upload_path(req.cond_image)
    req.cond_audio = {k: _resolve_upload_path(v) for k, v in req.cond_audio.items()}
    if not any(req.cond_audio.values()):
        raise HTTPException(status_code=400, detail="at least one of person1/person2 audio is required")
    return _enqueue("avatar_multi", req.model_dump())


# --------------------------------------------------------------------------- #
# task status / download
# --------------------------------------------------------------------------- #
@app.get("/tasks")
def list_tasks(limit: int = 100):
    return {"tasks": manager.list(limit)}


@app.get("/tasks/{task_id}")
def get_task(task_id: str):
    task = manager.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return task.to_dict()


@app.get("/tasks/{task_id}/files/{filename}")
def download_task_file(task_id: str, filename: str):
    task = manager.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    # prevent path traversal
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
    import uvicorn
    uvicorn.run("api.server:app", host=config.HOST, port=config.PORT, reload=False)


if __name__ == "__main__":
    main()
