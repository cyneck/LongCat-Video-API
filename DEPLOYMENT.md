# LongCat-Video API · 生产部署指南

本文面向「把 LongCat-Video-API 跑在带 GPU 的 Linux 服务器 / 容器里，并通过内置 H5 页面对外提供服务」的场景。
普通本地调试（仅跑 API、不涉及权重和推理）请看 [`api/README.md`](api/README.md) 的「快速开始」。

> 配套能力：本项目已内置 **H5 数字人创作页面**（`h5/index.html`）与 **登录网关**，通过环境变量一键开启，无需额外部署前端服务。

---

## 0. 前提条件（务必先核对）

| 项 | 要求 | 说明 |
|---|---|---|
| GPU 架构 | **Ampere 及以上**（A100 / H100 / L4 / RTX 30 / 40 系） | flash-attn 内核要求计算能力 ≥ 8.0；T4（7.5）、V100（7.0）不适合作为本项目推理 GPU |
| CUDA 驱动 | 12.x（建议 ≥ 12.4） | `nvidia-smi` 显示的驱动需支持对应 CUDA |
| Python | 3.11 / 3.12 | 本指南以 **3.12** 为例，需与 flash-attn wheel 的 `cp312` 对应 |
| 推荐显存 | **Avatar 1.5：A100 40GB 起，推荐 INT8 + distill** | 40GB 属于单卡保守运行档位；720p / 多段 / 多人会显著增加显存压力 |
| 推荐系统内存 | **≥ 40GB，生产建议 64GB+** | 40GB RAM 可以部署，但模型加载和 OS 文件缓存空间较紧 |
| 模型权重 | 约数十 GB | `weights/LongCat-Video` + `weights/LongCat-Video-Avatar-1.5` |
| 系统库 | `libsndfile1`、`ffmpeg` | Debian/Ubuntu 用 apt 安装 |

确认 GPU 与 CUDA：

```bash
nvidia-smi
nvcc --version        # 容器里可能没装 nvcc，看 nvidia-smi 右上角 CUDA Version 即可
python --version
free -h
```

### 0.1 单卡 A100 40GB 默认运行档位

当前 API 对 `avatar-v1.5` 默认启用面向 **1×A100 40GB** 的低显存 profile：

```text
model_type            avatar-v1.5
resolution            480p
num_segments          1
num_inference_steps   8
use_distill           true
use_int8              true
text_guidance_scale   1.0
audio_guidance_scale  1.0
GPU concurrency       1
```

即使旧版 H5 / 客户端仍显式提交 `50 steps / distill=false / int8=false`，后端也会把 Avatar 1.5 归一化到上述低显存档位，避免误走 BF16 高显存路径。

若部署在 A100 80GB / H100 / 多卡环境，需要恢复完全由调用方控制，可设置：

```bash
export LONGCAT_A100_40G_PROFILE=0
```

> 注意：上游 Avatar 推理脚本仍使用 93 帧生成形状。A100 40GB 是目标兼容配置，但实际峰值显存仍应在目标服务器上用单任务 smoke test 验证。首轮不要直接跑 720p、多 segment 或 GPU 并发。

---

## 1. 安装依赖

仓库里有三个依赖文件：

- `requirements_api.txt` — API 服务（FastAPI / Uvicorn / python-multipart）
- `requirements.txt` — 视频 / 数字人推理主依赖（PyTorch / Transformers / Diffusers / flash-attn / OpenCV 等）
- `requirements_avatar.txt` — 数字人专属依赖（librosa / ONNX Runtime / audio-separator 等）

> `requirements.txt` 中的 `flash-attn==2.7.4.post1` 若直接从 PyPI 安装，通常会触发源码编译。生产部署建议先安装与 Python / PyTorch / CUDA / C++ ABI 匹配的预编译 wheel。

### 1.1 系统库 + API 依赖

```bash
apt-get update && apt-get install -y libsndfile1 ffmpeg
pip install -r requirements_api.txt
```

### 1.2 PyTorch（CUDA 12.4）

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

### 1.3 flash-attn 预编译 wheel

标准 pip PyTorch 使用 old C++ ABI（`_GLIBCXX_USE_CXX11_ABI=0`），所以需要选择 `cxx11abiFALSE`。

Python 3.12 + PyTorch 2.6 + CUDA 12.x 示例：

```bash
WHEEL="flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
wget "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/$WHEEL"
pip install "$WHEEL"
rm -f "$WHEEL"
```

wheel 需同时匹配：

1. CUDA：`cu12` / 对应 CUDA 12.x
2. PyTorch：`torch2.6`
3. ABI：`cxx11abiFALSE`
4. Python：3.12 使用 `cp312`，3.11 使用 `cp311`

ABI 选错常见报错：

```text
ImportError: ... undefined symbol ... NSt7__cxx1112basic_string ...
```

### 1.4 其余推理依赖

```bash
pip install -r requirements.txt
pip install -r requirements_avatar.txt
```

### 1.5 环境冒烟验证

```bash
python - <<'PY'
import torch, flash_attn, transformers, diffusers
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("flash_attn:", flash_attn.__version__)
print("transformers:", transformers.__version__)
print("diffusers:", diffusers.__version__)
PY
```

`torch.cuda.is_available()` 必须为 `True`。

---

## 2. 下载模型权重

### 2.1 推荐：完整下载两个 Hugging Face 模型仓库

Avatar 1.5 **不需要额外单独下载 OpenAI Whisper**。项目读取的是 Avatar 权重仓库内部的：

```text
LongCat-Video-Avatar-1.5/
├── base_model/
├── base_model_int8/           # A100 40GB 默认使用
├── whisper-large-v3/          # Avatar 1.5 音频编码器
├── scheduler/
├── lora/
│   └── dmd_lora.safetensors   # 8-step distill
└── vocal_separator/
    └── Kim_Vocal_2.onnx       # 人声分离
```

因此最稳妥的方法是完整下载：

```bash
cd /opt/ml/cctv/LongCat-Video-API   # 改成你的实际仓库目录
mkdir -p weights

pip install -U huggingface_hub

hf download meituan-longcat/LongCat-Video \
  --local-dir ./weights/LongCat-Video

hf download meituan-longcat/LongCat-Video-Avatar-1.5 \
  --local-dir ./weights/LongCat-Video-Avatar-1.5
```

> 新版 `huggingface_hub` 推荐使用 `hf download`。旧的 `huggingface-cli download` 在部分环境仍可用，但本文统一使用 `hf` CLI。

Avatar 1.5 的 Whisper 在这里仅作为音频特征编码器使用，并不是运行时先做 ASR 再生成视频；模型代码直接读取 `whisper-large-v3` encoder hidden states。

### 2.2 中国大陆服务器

如果直连 Hugging Face 超时，可使用镜像 endpoint：

```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com

hf download meituan-longcat/LongCat-Video \
  --local-dir ./weights/LongCat-Video

hf download meituan-longcat/LongCat-Video-Avatar-1.5 \
  --local-dir ./weights/LongCat-Video-Avatar-1.5
```

`HF_ENDPOINT` 只影响下载阶段，模型落盘后本地推理无需访问 Hugging Face。

> `hf-mirror.com` 是第三方镜像。生产环境如能正常访问 Hugging Face 官方 Hub，优先使用官方源。

如果模型仓库需要认证：

```bash
hf auth login
hf auth whoami
```

### 2.3 下载后目录校验

基础模型必须包含共享的 tokenizer / text encoder / VAE / scheduler：

```bash
test -d weights/LongCat-Video/tokenizer && echo "tokenizer OK"
test -d weights/LongCat-Video/text_encoder && echo "text_encoder OK"
test -d weights/LongCat-Video/vae && echo "VAE OK"
test -d weights/LongCat-Video/scheduler && echo "scheduler OK"
```

A100 40GB 默认 Avatar 1.5 profile 重点检查：

```bash
test -d weights/LongCat-Video-Avatar-1.5/base_model_int8 && echo "INT8 DiT OK"
test -d weights/LongCat-Video-Avatar-1.5/whisper-large-v3 && echo "Whisper OK"
test -f weights/LongCat-Video-Avatar-1.5/lora/dmd_lora.safetensors && echo "DMD LoRA OK"
test -d weights/LongCat-Video-Avatar-1.5/vocal_separator && echo "Vocal separator OK"
test -d weights/LongCat-Video-Avatar-1.5/scheduler && echo "Avatar scheduler OK"
```

全部出现 `OK` 后再启动 API。

可以额外确认磁盘占用：

```bash
du -sh weights/LongCat-Video weights/LongCat-Video-Avatar-1.5
```

### 2.4 权重目录规则

默认读取目录：

- `weights/LongCat-Video` — 基础视频模型；Avatar 也从这里读取 tokenizer / UMT5 text encoder / VAE 等共享组件
- `weights/LongCat-Video-Avatar` — Avatar v1.0
- `weights/LongCat-Video-Avatar-1.5` — Avatar v1.5

可通过以下环境变量覆盖：

```text
LONGCAT_CHECKPOINT_DIR_VIDEO
LONGCAT_CHECKPOINT_DIR_AVATAR_V1
LONGCAT_CHECKPOINT_DIR_AVATAR_V15
LONGCAT_CHECKPOINT_DIR_AVATAR       # 旧兼容变量，会覆盖两个 Avatar 版本
```

v1.0 与 v1.5 是两套独立权重，目录结构不同，不能混用。

Avatar 推理强依赖基础视频模型。基础模型若放在自定义目录，显式设置：

```bash
export LONGCAT_CHECKPOINT_DIR_VIDEO=/path/to/LongCat-Video
export LONGCAT_CHECKPOINT_DIR_AVATAR_V15=/path/to/LongCat-Video-Avatar-1.5
```

服务在提交数字人任务前会执行权重就绪校验；缺失目录或版本错配会返回 HTTP 400，而不是等 torchrun 深层加载后才失败。

---

## 3. 生产配置（环境变量）

常用配置定义在 `api/config.py`：

| 变量 | 默认值 | A100 40GB 建议 | 说明 |
|---|---|---|---|
| `LONGCAT_HOST` | `0.0.0.0` | `0.0.0.0` | 监听地址 |
| `LONGCAT_PORT` | `8000` | 按暴露端口设置 | 服务端口 |
| `LONGCAT_EMBED_H5` | `0` | `1` | `/` 返回内置 H5 |
| `LONGCAT_AUTH` | `0` | `1` | 开启登录网关 |
| `LONGCAT_USER` | `admin` | 自定义 | 登录用户名 |
| `LONGCAT_PASS` | `admin` | 强密码 | 登录密码 |
| `LONGCAT_AUTH_TOKEN` | `longcat-demo-token` | 随机串 | HttpOnly cookie token |
| `LONGCAT_NUM_GPUS` | `1` | `1` | 每个任务使用 GPU 数 |
| `LONGCAT_CONTEXT_PARALLEL_SIZE` | 跟随 GPU 数 | `1` | Context Parallel 大小 |
| `LONGCAT_GPU_CONCURRENCY` | `1` | **`1`** | A100 40GB 不要并发跑多个生成任务 |
| `LONGCAT_ENABLE_COMPILE` | `0` | `0` | 首轮 smoke test 保持关闭 |
| `LONGCAT_A100_40G_PROFILE` | `1` | **`1`** | Avatar 1.5 自动使用 INT8 + 8-step distill 低显存档位 |
| `LONGCAT_WORK_DIR` | `./api_work` | 默认即可 | 上传 / 输出 / 日志目录 |
| `LONGCAT_CHECKPOINT_DIR_VIDEO` | `weights/LongCat-Video` | 按实际路径 | 基础模型 |
| `LONGCAT_CHECKPOINT_DIR_AVATAR_V1` | `weights/LongCat-Video-Avatar` | 可不下载 | Avatar v1.0 |
| `LONGCAT_CHECKPOINT_DIR_AVATAR_V15` | `weights/LongCat-Video-Avatar-1.5` | 按实际路径 | Avatar v1.5 |

### 3.1 A100 40GB 推荐环境变量

```bash
export LONGCAT_NUM_GPUS=1
export LONGCAT_CONTEXT_PARALLEL_SIZE=1
export LONGCAT_GPU_CONCURRENCY=1
export LONGCAT_A100_40G_PROFILE=1
export LONGCAT_ENABLE_COMPILE=0

export LONGCAT_CHECKPOINT_DIR_VIDEO="$PWD/weights/LongCat-Video"
export LONGCAT_CHECKPOINT_DIR_AVATAR_V15="$PWD/weights/LongCat-Video-Avatar-1.5"
```

### 3.2 鉴权配置

```bash
export LONGCAT_AUTH=1
export LONGCAT_USER=admin
export LONGCAT_PASS='改成强密码'
export LONGCAT_EMBED_H5=1
```

等价命令行参数：

```bash
python -m api.server --auth --user admin --pass '改成强密码' --embed-h5
```

---

## 4. 启动服务

### 4.1 前台 / 简单后台启动

```bash
cd /opt/ml/cctv/LongCat-Video-API

export LONGCAT_HOST=0.0.0.0
export LONGCAT_PORT=8080
export LONGCAT_EMBED_H5=1
export LONGCAT_AUTH=1
export LONGCAT_USER=admin
export LONGCAT_PASS='改成强密码'

export LONGCAT_NUM_GPUS=1
export LONGCAT_CONTEXT_PARALLEL_SIZE=1
export LONGCAT_GPU_CONCURRENCY=1
export LONGCAT_A100_40G_PROFILE=1

uvicorn api.server:app --host 0.0.0.0 --port 8080
```

后台运行：

```bash
nohup uvicorn api.server:app --host 0.0.0.0 --port 8080 \
  > api_work/server.log 2>&1 &
```

### 4.2 systemd（推荐生产）

`/etc/systemd/system/longcat-api.service`：

```ini
[Unit]
Description=LongCat-Video API
After=network.target

[Service]
WorkingDirectory=/opt/ml/cctv/LongCat-Video-API
ExecStart=/opt/conda/bin/python3.12 -m api.server
Environment=LONGCAT_HOST=0.0.0.0
Environment=LONGCAT_PORT=8080
Environment=LONGCAT_EMBED_H5=1
Environment=LONGCAT_AUTH=1
Environment=LONGCAT_USER=admin
Environment=LONGCAT_PASS=改成强密码
Environment=LONGCAT_NUM_GPUS=1
Environment=LONGCAT_CONTEXT_PARALLEL_SIZE=1
Environment=LONGCAT_GPU_CONCURRENCY=1
Environment=LONGCAT_A100_40G_PROFILE=1
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable --now longcat-api
journalctl -u longcat-api -f
```

### 4.3 启动验证

```bash
curl -s http://127.0.0.1:8080/health
```

如果启用了鉴权：

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/tasks
curl -s -c /tmp/cj -X POST \
  -d 'username=admin&password=改成强密码' \
  http://127.0.0.1:8080/auth/login
curl -s -b /tmp/cj http://127.0.0.1:8080/tasks
```

---

## 5. A100 40GB 首次推理验证

第一轮只跑：

- Avatar v1.5
- 单人 AI2V
- 480p
- 1 segment
- INT8
- distill / 8-step
- GPU concurrency = 1

生成期间监控：

```bash
watch -n 0.5 nvidia-smi
```

另一个终端监控系统内存：

```bash
watch -n 1 free -h
```

建议记录：

- 峰值 GPU 显存
- 峰值系统 RAM
- 单段生成耗时
- 是否发生 CUDA OOM / Linux OOM Killer

如果 480p / 1 segment 已接近 40GB 显存上限，不要直接开启 720p 或多任务并发。

---

## 6. 访问 H5 页面

1. 浏览器打开 `http://<服务器IP或域名>:8080/`
2. 启用鉴权后会跳转 `/login`
3. 使用 `LONGCAT_USER` / `LONGCAT_PASS` 登录
4. 上传肖像图片 + 语音 → 填写 prompt → 提交生成 → 查看日志 / 下载视频

Swagger UI：

```text
http://<host>:8080/docs
```

---

## 7. Nginx / HTTPS

H5 与 API 同源，只需要反代一个 API 端口：

```nginx
server {
    listen 80;
    server_name your.domain.com;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        client_max_body_size 200m;
        proxy_read_timeout 600s;
    }
}
```

对外发布建议再配置 TLS/HTTPS。

---

## 8. 更新流程

```bash
cd /opt/ml/cctv/LongCat-Video-API
git pull origin main

pip install -r requirements_api.txt
pip install -r requirements.txt
pip install -r requirements_avatar.txt

# systemd
systemctl restart longcat-api
```

模型权重一般无需随着代码每次重新下载；只有上游权重版本发生变化时才需要更新。

---

## 9. 常见问题排查

### Q1. `flash_attn` 报 `undefined symbol ... NSt7__cxx1112basic_string`

ABI 不匹配。卸载并安装 `cxx11abiFALSE` 版本：

```bash
pip uninstall -y flash-attn
# 按 §1.3 安装正确 wheel
```

### Q2. `ModuleNotFoundError: No module named 'longcat_video'`

API 的 task manager 会自动把仓库根目录加入 `PYTHONPATH`。如果直接手工执行 `api/scripts/run_avatar_*.py`，需确保：

```bash
export PYTHONPATH=/opt/ml/cctv/LongCat-Video-API:$PYTHONPATH
```

正常生产环境建议通过 API / task manager 启动推理子进程。

### Q3. `ModuleNotFoundError: No module named 'transformers'`

只安装了 API 依赖：

```bash
pip install -r requirements.txt
```

### Q4. Avatar 1.5 报找不到 Whisper

不要单独把 Whisper 下载到 Hugging Face 默认 cache。项目预期路径是：

```text
weights/LongCat-Video-Avatar-1.5/whisper-large-v3
```

重新完整下载 Avatar 1.5：

```bash
hf download meituan-longcat/LongCat-Video-Avatar-1.5 \
  --local-dir ./weights/LongCat-Video-Avatar-1.5
```

### Q5. A100 40GB 报 CUDA OOM

先确认请求实际使用：

```text
model_type=avatar-v1.5
resolution=480p
num_segments=1
use_int8=true
use_distill=true
num_inference_steps=8
```

并确认：

```bash
export LONGCAT_A100_40G_PROFILE=1
export LONGCAT_GPU_CONCURRENCY=1
```

若仍 OOM，先不要尝试 720p / 多段 / 多人；记录 `nvidia-smi` 峰值，再评估进一步模型 offload 或帧数优化。

### Q6. Linux 进程被杀但没有 CUDA OOM

很可能是 40GB 系统内存触发 OOM Killer：

```bash
dmesg -T | grep -Ei 'oom|killed process'
free -h
```

生产建议把系统内存提升到 64GB 或以上。

### Q7. 推理子进程 `ChildFailedError`

查看任务日志：

```text
GET /tasks/{task_id}/log
```

重点检查：

- 权重目录是否完整
- INT8 / DMD / Whisper 子目录是否存在
- CUDA / flash-attn 是否正常
- 是否 OOM

### Q8. `/` 返回 JSON 而不是 H5

设置：

```bash
export LONGCAT_EMBED_H5=1
```

### Q9. GPU 不可用

确认：

```bash
nvidia-smi
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
```

容器环境需正确传递 GPU device。

---

## 10. 目录与产物

```text
api_work/uploads/{images,videos,audios,jsons}/
api_work/outputs/<task_id>/
api_work/logs/<task_id>.log
api_work/tasks.json
```

建议将 `api_work/` 保持在 `.gitignore` 中。
