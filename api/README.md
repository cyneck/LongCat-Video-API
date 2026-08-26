# LongCat-Video API

把 LongCat-Video 系列推理脚本（文生视频 / 图生视频 / 视频续写 / 单/多音频数字人）封装为 HTTP 接口，支持上传图片、视频、音频作为合成素材，异步执行生成任务并下载产物。

- 交互式文档（Swagger UI）：`http://<host>:8000/docs`
- ReDoc：`http://<host>:8000/redoc`

---

## 目录

- [工作模型](#工作模型)
- [快速开始](#快速开始)
- [配置项](#配置项)
- [通用约定](#通用约定)
- [文件上传接口](#文件上传接口)
- [生成接口](#生成接口)
  - [文本生成视频](#1-文本生成视频)
  - [图像生成视频](#2-图像生成视频)
  - [视频续写](#3-视频续写)
  - [单音频数字人](#4-单音频数字人)
  - [多音频数字人](#5-多音频数字人)
- [任务查询与下载接口](#任务查询与下载接口)
- [错误码](#错误码)
- [端到端调用示例](#端到端调用示例)
- [产出文件说明](#产出文件说明)
- [注意事项](#注意事项)

---

## 工作模型

底层 `run_demo_*.py` 依赖 `torchrun` 启动（脚本内部调用 `dist.init_process_group(backend="nccl")` 并读取 `RANK` 环境变量），因此**无法**在 FastAPI 同进程内直接运行 pipeline。本服务的处理链路为：

```
客户端                       FastAPI 服务                         torchrun 子进程
  │  1. POST /files/* ─────────►  落盘到 uploads/，返回 path
  │  2. POST /generate/* ───────►  校验素材路径 → 生成 task_input.json
  │                                → 入队（GPU 串行）→ 返回 task_id (202)
  │                                3. torchrun api/scripts/run_*.py ────►  推理 → 写 mp4 到 outputs/<task_id>/
  │  4. GET /tasks/{id} ────────►  返回状态 + 产出清单
  │  5. GET /tasks/{id}/files/..►  下载 mp4
```

- **GPU 串行**：同一时刻只跑 `LONGCAT_GPU_CONCURRENCY` 个任务，避免显存冲突，其余排队。
- **状态持久化**：任务记录写入 `api_work/tasks.json`，服务重启后历史任务可查。
- **素材路径校验**：生成接口里引用的图片/视频/音频路径必须来自 `/files/*` 上传返回的 `path`，拒绝越界路径。

## 快速开始

```shell
# 1. 安装（在已装好 requirements.txt / requirements_avatar.txt 的环境基础上）
pip install -r requirements_api.txt

# 2. 配置：复制模板，按服务器实际修改路径 / 端口
cp .env.example .env
#   重点：LONGCAT_CHECKPOINT_DIR_VIDEO / _AVATAR 用绝对路径，避免 ../LongCat-Video 回退失效
#   也可直接 export 覆盖（与 .env 等效）：
export LONGCAT_NUM_GPUS=2
export LONGCAT_CHECKPOINT_DIR_VIDEO=./weights/LongCat-Video
export LONGCAT_CHECKPOINT_DIR_AVATAR=./weights/LongCat-Video-Avatar-1.5

# 3. 启动（start.sh 会自动 source .env 并注入 PYTHONPATH）
./start.sh
# 或：uvicorn api.server:app --host 0.0.0.0 --port 8000
# 或：python -m api.server
```

打开 `http://localhost:8000/docs` 即可在浏览器里调试全部接口。

> **生产部署（GPU 服务器 + H5 对外服务 + 登录网关）请看根目录 [`DEPLOYMENT.md`](../DEPLOYMENT.md)**，里面包含依赖安装修正（flash-attn 预编译 wheel / ABI 坑）、权重下载、systemd 守护、Nginx 反代与排错。

## H5 嵌入与登录网关

本服务内置一个移动端「数字人创作」H5 页面（`h5/index.html`）和一个登录页（`h5/login.html`），通过环境变量一键启用，无需单独部署前端。

- **开启 H5**：`LONGCAT_EMBED_H5=1` → `GET /` 返回 `h5/index.html`。页面使用相对地址（`/health`、`/files/*`、`/generate/*`），因此无论服务跑在 `8000` / `8080` 还是 Nginx 反代域名下都无需改代码。
- **开启登录网关**：`LONGCAT_AUTH=1` → 中间件拦截所有接口（除 `/health`、`/login`、`/auth/login`）。未登录访问 `/` 或 `/h5*` 会 `307` 跳到 `/login`；访问 API 会返回 `401`。
- **登录流程**：浏览器访问 `/login` → 提交 `LONGCAT_USER` / `LONGCAT_PASS` → 服务端在校验通过后向客户端种下 HttpOnly cookie（`lc_token`）→ 重定向回 `/` 加载 H5。cookie 有效期 7 天。

启动示例（容器暴露 8080、需登录）：

```shell
export LONGCAT_PORT=8080 LONGCAT_EMBED_H5=1 LONGCAT_AUTH=1
export LONGCAT_USER=admin LONGCAT_PASS='改成强密码'
uvicorn api.server:app --host 0.0.0.0 --port 8080
```

> 注意：`LONGCAT_AUTH_TOKEN` 是 cookie 里存的实际令牌串，默认 `longcat-demo-token`。若不修改，知道令牌的人可绕过登录页直接带 cookie 访问；生产环境建议改成一个随机串。

## 配置项

均通过环境变量覆盖，定义见 `api/config.py`。

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `LONGCAT_CHECKPOINT_DIR_VIDEO` | `weights/LongCat-Video` | 基础视频模型权重目录 |
| `LONGCAT_CHECKPOINT_DIR_AVATAR` | `weights/LongCat-Video-Avatar-1.5` | 数字人权重目录 |
| `LONGCAT_NUM_GPUS` | `1` | 每个任务使用的 GPU 数（=torchrun `nproc_per_node`） |
| `LONGCAT_CONTEXT_PARALLEL_SIZE` | 同 `NUM_GPUS` | context parallel 大小 |
| `LONGCAT_GPU_CONCURRENCY` | `1` | 同时执行的生成任务数（默认串行） |
| `LONGCAT_ENABLE_COMPILE` | `0` | 是否对视频任务启用 `torch.compile`（avatar 脚本不支持） |
| `LONGCAT_WORK_DIR` | `./api_work` | 上传 / 输出 / 日志根目录 |
| `LONGCAT_HOST` / `LONGCAT_PORT` | `0.0.0.0` / `8000` | 监听地址与端口（代码无硬编码，改 `LONGCAT_PORT` 即可换端口） |
| `LONGCAT_EMBED_H5` | `0` | 置 `1` 时 `/` 直接返回 `h5/index.html`（数字人创作页） |
| `LONGCAT_AUTH` | `0` | 置 `1` 时开启登录网关，未登录拦截除 `/health`、`/login`、`/auth/login` 外的全部接口 |
| `LONGCAT_USER` / `LONGCAT_PASS` | `admin` / `admin` | 登录账号 / 密码（生产请改掉默认值） |
| `LONGCAT_AUTH_TOKEN` | `longcat-demo-token` | 登录后写入 HttpOnly cookie 的令牌值，可自定义 |

> **avatar 版本切换**：默认指向 `LongCat-Video-Avatar-1.5`。若用 v1.0，把 `LONGCAT_CHECKPOINT_DIR_AVATAR` 指到 `weights/LongCat-Video-Avatar`，并在请求里设 `"model_type":"avatar-v1.0"`。

## 通用约定

### 任务状态机

```
pending → running → done
                  └→ failed
```

- `pending`：已入队，等待 GPU 空闲
- `running`：torchrun 子进程执行中
- `done`：成功，`outputs` 字段列出可下载文件
- `failed`：失败，`error` 字段含日志尾部，可查 `/tasks/{id}/log` 看完整日志

### 请求/响应格式

- 上传接口：`multipart/form-data`
- 生成接口：`application/json`
- 查询接口：`GET`，返回 `application/json`
- 生成接口成功返回 `202 Accepted`，响应体含 `task_id`

### 素材引用方式

上传后返回的 `path` 是一个**绝对路径**。调用生成接口时把该 `path` 填入对应字段（如 `cond_image`、`cond_video`、`cond_audio.person1`）。服务端会校验该路径位于 `api_work/uploads/` 之内且确实存在。

---

## 文件上传接口

所有上传文件按类型分别存到 `api_work/uploads/{images,videos,audios,jsons}/`，返回的 `path` 供后续生成接口引用。

### POST `/files/image`

上传图片素材。

- 允许扩展名：`.png` `.jpg` `.jpeg` `.webp` `.bmp`
- 表单字段：`file`

**请求示例**

```bash
curl -F "file=@girl.png" http://localhost:8000/files/image
```

**响应 200**

```json
{
  "path": "D:/code/LongCat-Video/api_work/uploads/images/3ebe2af6adfc.png",
  "filename": "3ebe2af6adfc.png",
  "size": 1024
}
```

### POST `/files/video`

上传视频素材（用于视频续写）。

- 允许扩展名：`.mp4` `.mov` `.avi` `.mkv` `.webm`
- 表单字段：`file`

```bash
curl -F "file=@clip.mp4" http://localhost:8000/files/video
```

### POST `/files/audio`

上传音频素材（用于数字人）。

- 允许扩展名：`.mp3` `.wav` `.flac` `.aac` `.m4a` `.ogg`
- 表单字段：`file`

```bash
curl -F "file=@speech.mp3" http://localhost:8000/files/audio
```

### POST `/files/json`

高级用法：上传一个预构建的 avatar 输入 JSON（结构与 `assets/avatar/*.json` 一致），返回 `path` 与解析后的 `preview`。

```bash
curl -F "file=@my_avatar_input.json" http://localhost:8000/files/json
```

**响应 200**

```json
{
  "path": "D:/code/LongCat-Video/api_work/uploads/jsons/abc123.json",
  "filename": "abc123.json",
  "preview": { "prompt": "...", "cond_image": "...", "cond_audio": { "person1": "..." } }
}
```

---

## 生成接口

### 1. 文本生成视频

`POST /generate/text-to-video`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `prompt` | string | *必填* | 正向提示词 |
| `negative_prompt` | string | 内置默认 | 负向提示词 |
| `height` | int | 480 | 画面高度 |
| `width` | int | 832 | 画面宽度 |
| `num_frames` | int | 93 | 帧数 |
| `num_inference_steps` | int | 50 | 去噪步数 |
| `guidance_scale` | float | 4.0 | CFG 引导强度 |
| `spatial_refine_only` | bool | false | 仅做空间精修 |
| `seed` | int | 42 | 随机种子 |

```bash
curl -X POST http://localhost:8000/generate/text-to-video \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A cat playing piano in a jazz bar, cinematic lighting","num_inference_steps":50}'
```

**响应 202**

```json
{ "task_id": "d3d13b94f4fd4382", "status": "pending" }
```

### 2. 图像生成视频

`POST /generate/image-to-video`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `cond_image` | string | *必填* | 上传图片返回的 `path` |
| `prompt` | string | *必填* | 正向提示词 |
| `negative_prompt` | string | 内置默认 | 负向提示词 |
| `resolution` | string | `480p` | `480p` 或 `720p` |
| `num_frames` | int | 93 | 帧数 |
| `num_inference_steps` | int | 50 | 去噪步数 |
| `guidance_scale` | float | 4.0 | CFG 引导强度 |
| `spatial_refine_only` | bool | false | 仅做空间精修 |
| `seed` | int | 42 | 随机种子 |

```bash
curl -X POST http://localhost:8000/generate/image-to-video \
  -H "Content-Type: application/json" \
  -d "{\"cond_image\":\"$IMG\",\"prompt\":\"she picks up a coffee cup and sips\"}"
```

### 3. 视频续写

`POST /generate/video-continuation`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `cond_video` | string | *必填* | 上传视频返回的 `path` |
| `prompt` | string | *必填* | 正向提示词 |
| `negative_prompt` | string | 内置默认 | 负向提示词 |
| `resolution` | string | `480p` | `480p` 或 `720p` |
| `num_frames` | int | 93 | 帧数 |
| `num_cond_frames` | int | 13 | 条件帧数 |
| `num_inference_steps` | int | 50 | 去噪步数 |
| `guidance_scale` | float | 4.0 | CFG 引导强度 |
| `spatial_refine_only` | bool | false | 仅做空间精修 |
| `seed` | int | 42 | 随机种子 |

```bash
curl -X POST http://localhost:8000/generate/video-continuation \
  -H "Content-Type: application/json" \
  -d "{\"cond_video\":\"$VID\",\"prompt\":\"the rider accelerates along the road\"}"
```

### 4. 单音频数字人

`POST /generate/avatar-single`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `prompt` | string | *必填* | 场景描述（越详细越好） |
| `cond_audio` | object | *必填* | `{"person1":"<audio path>"}` |
| `cond_image` | string | null | 参考人像图，`stage_1=ai2v` 时必填 |
| `stage_1` | string | `ai2v` | `at2v`（纯音频+文本）或 `ai2v`（音频+图像） |
| `resolution` | string | `480p` | `480p` 或 `720p` |
| `num_segments` | int | 1 | 生成长视频的段数 |
| `num_inference_steps` | int | 50 | 去噪步数（v1.5+distill 自动设为 8） |
| `text_guidance_scale` | float | 4.0 | 文本 CFG |
| `audio_guidance_scale` | float | 4.0 | 音频 CFG（3–5 口型更准） |
| `ref_img_index` | int | 10 | 参考帧索引 |
| `mask_frame_range` | int | 3 | 掩码帧范围 |
| `model_type` | string | `avatar-v1.5` | `avatar-v1.0` 或 `avatar-v1.5` |
| `use_distill` | bool | false | 蒸馏加速（v1.5 必须开） |
| `use_int8` | bool | false | INT8 量化（仅 v1.5） |
| `seed` | int | 42 | 随机种子 |

```bash
curl -X POST http://localhost:8000/generate/avatar-single \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"a man speaking on stage under dramatic lighting\",\"cond_image\":\"$IMG\",\"cond_audio\":{\"person1\":\"$AUD\"},\"stage_1\":\"ai2v\",\"model_type\":\"avatar-v1.5\",\"use_distill\":true,\"use_int8\":true}"
```

> v1.5 推荐组合：`model_type=avatar-v1.5` + `use_distill=true` + `use_int8=true`。

### 5. 多音频数字人

`POST /generate/avatar-multi`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `prompt` | string | *必填* | 场景描述 |
| `cond_image` | string | *必填* | 含两人的参考图 |
| `cond_audio` | object | *必填* | `{"person1":"<path>","person2":"<path>"}`，至少一个 |
| `audio_type` | string | `para` | `para`（合并，需等长）或 `add`（拼接，无需等长） |
| `bbox` | object | null | 可选，`{"person1":[y0,x0,y1,x1],"person2":[...],"others":[...]}` |
| `resolution` | string | `480p` | `480p` 或 `720p` |
| `num_segments` | int | 1 | 段数 |
| `num_inference_steps` | int | 50 | 去噪步数 |
| `text_guidance_scale` | float | 4.0 | 文本 CFG |
| `audio_guidance_scale` | float | 4.0 | 音频 CFG |
| `ref_img_index` | int | 10 | 参考帧索引 |
| `mask_frame_range` | int | 3 | 掩码帧范围 |
| `model_type` | string | `avatar-v1.5` | `avatar-v1.0` 或 `avatar-v1.5` |
| `use_distill` | bool | false | 蒸馏加速 |
| `use_int8` | bool | false | INT8 量化（仅 v1.5） |
| `seed` | int | 42 | 随机种子 |

```bash
curl -X POST http://localhost:8000/generate/avatar-multi \
  -H "Content-Type: application/json" \
  -d "{\"prompt\":\"two people singing face to face\",\"cond_image\":\"$IMG\",\"cond_audio\":{\"person1\":\"$AUD1\",\"person2\":\"$AUD2\"},\"audio_type\":\"para\",\"model_type\":\"avatar-v1.5\",\"use_distill\":true,\"use_int8\":true}"
```

---

## 任务查询与下载接口

### GET `/tasks`

列出最近的任务（默认按创建时间倒序）。

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `limit` | int | 100 | 返回条数上限 |

```bash
curl http://localhost:8000/tasks
```

```json
{
  "tasks": [
    {
      "task_id": "d3d13b94f4fd4382",
      "task_type": "text_to_video",
      "status": "done",
      "output_dir": ".../outputs/d3d13b94f4fd4382",
      "created_at": 1750000000.0,
      "started_at": 1750000001.0,
      "finished_at": 1750000120.0,
      "return_code": 0,
      "error": "",
      "outputs": [
        { "filename": "output.mp4", "size": 5242880,
          "download_url": "/tasks/d3d13b94f4fd4382/files/output.mp4" }
      ]
    }
  ]
}
```

### GET `/tasks/{task_id}`

查询单个任务的最新状态与产出清单。返回结构与上条 `tasks[]` 元素一致。任务不存在返回 `404`。

```bash
curl http://localhost:8000/tasks/d3d13b94f4fd4382
```

### GET `/tasks/{task_id}/files/{filename}`

下载该任务产出的文件。`filename` 必须是 `outputs` 里列出的文件名，禁止包含 `/`、`\`、`..`。文件不存在返回 `404`。

```bash
curl -o result.mp4 http://localhost:8000/tasks/d3d13b94f4fd4382/files/output.mp4
```

### GET `/tasks/{task_id}/log`

返回该任务子进程的完整 stdout/stderr 日志（推理进度、报错堆栈都在这里）。

```bash
curl http://localhost:8000/tasks/d3d13b94f4fd4382/log
```

```json
{ "log": "...full subprocess output..." }
```

---

## 错误码

| HTTP 状态 | 触发场景 |
|---|---|
| `202` | 生成任务已成功入队 |
| `200` | 上传/查询/下载成功 |
| `400` | 请求体校验失败；扩展名不允许；素材路径越界或不存在；业务校验失败（如 `ai2v` 未传 `cond_image`、multi 未传任何音频） |
| `404` | 任务或文件不存在 |
| `422` | JSON 字段类型/格式不符（Pydantic 校验） |
| `500` | 服务内部异常 |

生成接口返回 `202` 只代表任务入队成功；**推理是否成功**要看 `GET /tasks/{id}` 的 `status` 是否为 `done`，失败时查 `error` 字段或 `/tasks/{id}/log`。

---

## 端到端调用示例

以「图生视频」为例，完整流程：

```bash
HOST=http://localhost:8000

# 1. 上传图片
IMG=$(curl -s -F "file=@girl.png" $HOST/files/image \
      | python -c "import sys,json;print(json.load(sys.stdin)['path'])")

# 2. 提交生成任务
TASK=$(curl -s -X POST $HOST/generate/image-to-video \
      -H "Content-Type: application/json" \
      -d "{\"cond_image\":\"$IMG\",\"prompt\":\"she sips coffee by the window\"}")
echo $TASK
TID=$(echo $TASK | python -c "import sys,json;print(json.load(sys.stdin)['task_id'])")

# 3. 轮询状态
while :; do
  STATUS=$(curl -s $HOST/tasks/$TID | python -c "import sys,json;print(json.load(sys.stdin)['status'])")
  echo "status: $STATUS"
  [ "$STATUS" = "done" ] && break
  [ "$STATUS" = "failed" ] && { echo "FAILED"; curl -s $HOST/tasks/$TID/log; exit 1; }
  sleep 10
done

# 4. 下载结果
curl -o result.mp4 $HOST/tasks/$TID/files/output.mp4
```

---

## 产出文件说明

每个任务在其 `output_dir`（`api_work/outputs/<task_id>/`）下产出若干 mp4，`GET /tasks/{id}` 的 `outputs` 字段会列出全部可下载文件。

| 任务类型 | 产出文件 |
|---|---|
| text_to_video / image_to_video / video_continuation | `stage1_480p.mp4`、`stage2_distill_480p.mp4`、`output.mp4`（720p 精修最终结果，**下载首选**） |
| avatar_single / avatar_multi | `segment_1.mp4`（首段），长视频另有 `video_continue_N.mp4`；单段时额外生成 `output.mp4` 别名 |

---

## 注意事项

1. **torchrun 端口**：子进程以 `torchrun --master_port=29501` 启动。默认串行执行不会冲突；若调高 `LONGCAT_GPU_CONCURRENCY` 使任务并行，需在 `api/task_manager.py` 里为每个任务分配不同 master_port。
2. **`torch.compile`** 仅对视频三件套脚本生效（avatar 脚本无此参数），由 `LONGCAT_ENABLE_COMPILE=1` 开启。
3. **多 GPU**：把 `LONGCAT_NUM_GPUS` 设为可用卡数，脚本会自动 context-parallel。
4. **存储清理**：上传文件在 `api_work/uploads/`、产出在 `api_work/outputs/`、日志在 `api_work/logs/`，请定期清理；建议把 `api_work/` 加入 `.gitignore`。
5. **运行环境**：子进程实际推理需要 `torch` + `flash-attn` + 对应权重就绪（见根目录 `README.md` 的安装步骤）。服务本身只依赖 `fastapi` / `uvicorn` / `python-multipart`。
6. **同步阻塞**：生成接口里的素材路径校验是同步文件系统调用（仅 `exists`/`resolve`），开销极小；真正的推理在子进程里，不阻塞主事件循环。
