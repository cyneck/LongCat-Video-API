# LongCat-Video API · 生产部署指南

本文面向「把 LongCat-Video-API 跑在带 GPU 的 Linux 服务器 / 容器里，并通过内置 H5 页面对外提供服务」的场景。
普通本地调试（仅跑 API、不涉及权重和推理）请看 [`api/README.md`](api/README.md) 的「快速开始」。

> 配套能力：本项目已内置 **H5 数字人创作页面**（`h5/index.html`）与 **登录网关**，通过环境变量一键开启，无需额外部署前端服务。

---

## 0. 前提条件（务必先核对）

| 项 | 要求 | 说明 |
|---|---|---|
| GPU 架构 | **Ampere 及以上**（A100 / H100 / L4 / RTX 30 / 40 系） | flash-attn 内核要求计算能力 ≥ 8.0；**T4（7.5）、V100（7.0）会在推理时挂掉**，即使 import 成功 |
| CUDA 驱动 | 12.x（建议 ≥ 12.4） | `nvidia-smi` 看到的驱动版本需支持对应 CUDA |
| Python | 3.11 / 3.12 | 本指南以 **3.12** 为例（需与 flash-attn wheel 的 `cp312` 对应） |
| 显存 | 单卡 ≥ 24GB 较稳妥（13.6B 模型） | 具体取决于分辨率 / 段数 |
| 模型权重 | ~40GB 磁盘 | `weights/LongCat-Video` + `weights/LongCat-Video-Avatar-1.5` |
| 系统库 | `librosa` 依赖的 `libsndfile1` | Debian/Ubuntu：`apt-get install -y libsndfile1` |

确认 GPU 与 CUDA：

```bash
nvidia-smi
nvcc --version        # 容器里可能没装 nvcc，看 nvidia-smi 右上角的 CUDA Version 即可
python --version
```

---

## 1. 安装依赖（已修正坑版）

仓库里有三个依赖文件，分工如下：

- `requirements_api.txt` — 仅 API 服务本身（fastapi / uvicorn / python-multipart），**必须装**
- `requirements.txt` — 视频 / 数字人推理主依赖（torch / transformers / diffusers / flash-attn / OpenCV 等）
- `requirements_avatar.txt` — 数字人专属依赖（librosa / onnxruntime / audio-separator 等）+ **系统库用 apt 装，不再是 pip 包**

> ⚠️ `requirements.txt` 里的 `flash-attn==2.7.4.post1` 在 PyPI 只有源码包，直接 `pip install -r requirements.txt` 会触发 **10~30 分钟源码编译且容易失败**。
> 正确做法：**先单独装预编译 wheel，再装其余依赖**（flash-attn 已满足后 pip 会自动跳过）。

### 1.1 装系统库 + API 服务依赖

```bash
# 系统库（声音处理）
apt-get update && apt-get install -y libsndfile1 ffmpeg

# API 服务本身
pip install -r requirements_api.txt
```

### 1.2 装 PyTorch（CUDA 12.4 版）

```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

> 为什么锁定 2.6.0：flash-attn 官方预编译 wheel 对 2.6 有现成版本；更高版本（如 2.13）暂无对应 wheel，会逼你源码编译踩坑。

### 1.3 装 flash-attn（预编译 wheel，重点是 **cxx11abiFALSE**）

标准 pip 安装的 PyTorch 用的是 **old C++ ABI**（`_GLIBCXX_USE_CXX11_ABI=0`），因此必须选名字带 **`cxx11abiFALSE`** 的 wheel。装错 `cxx11abiTRUE` 会出现：

```
ImportError: ... undefined symbol: _ZN3c105ErrorC2E ... NSt7__cxx1112basic_string ...
```

这就是 ABI 不匹配，不是版本问题，重装 FALSE 变体即可。

下载并安装（Python 3.12 + torch 2.6 + CUDA 12.x）：

```bash
# 直接用官方 release 的预编译 wheel（cu12 通用 CUDA 12.x，无需精确到 12.4）
WHEEL="flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
wget "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/$WHEEL"

# 若 GitHub 下载慢/被墙，走镜像（把整条 URL 前缀换成 gh-proxy.org/）
# wget "https://gh-proxy.org/https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/$WHEEL"

pip install "$WHEEL"
rm -f "$WHEEL"
```

> 选择 wheel 的三要素，缺一不可：
> 1. `cu12`（对应 CUDA 12.x）或 `cu124`
> 2. `torch2.6`（与上面装的 torch 2.6.0 对齐）
> 3. `cxx11abiFALSE`（与标准 pip torch 的 old ABI 对齐）
> 4. `cp312`（与你的 Python 3.12 对齐；若是 3.11 就找 `cp311`）

### 1.4 装其余推理依赖

```bash
# 此时 requirements.txt 里的 flash-attn 已满足，pip 跳过编译
pip install -r requirements.txt
pip install -r requirements_avatar.txt
```

### 1.5 冒烟验证

```bash
python -c "import torch,flash_attn,transformers; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
print('flash_attn', flash_attn.__version__)"
```

看到版本号且无报错即环境就绪。**注意**：`torch.cuda.is_available()` 为 `True` 才算 GPU 可用。

---

## 2. 下载模型权重

```bash
pip install "huggingface_hub[cli]"

huggingface-cli download meituan-longcat/LongCat-Video \
  --local-dir ./weights/LongCat-Video

huggingface-cli download meituan-longcat/LongCat-Video-Avatar-1.5 \
  --local-dir ./weights/LongCat-Video-Avatar-1.5
```

### 2.1 国内下载（HF 被墙/超时首选）

直连 HuggingFace 在大陆服务器经常超时或断流，改用官方镜像 `hf-mirror.com`，**两条命令前加一行环境变量即可**：

```bash
pip install "huggingface_hub[cli]"

export HF_ENDPOINT=https://hf-mirror.com

hf-download() { huggingface-cli download "$1" --local-dir "$2"; }

hf-download meituan-longcat/LongCat-Video        ./weights/LongCat-Video
hf-download meituan-longcat/LongCat-Video-Avatar-1.5 ./weights/LongCat-Video-Avatar-1.5
```

> 镜像只加速「下载」环节，`HF_ENDPOINT` 仅影响 `huggingface-cli` 拉取地址；权重落盘后推理不再访问网络。
> 若仍慢，可加 `--local-dir-use-symlinks False` 减少软链开销，或配合 `aria2` 多线程（见 `hf-mirror.com` 文档）。

- 权重默认读取目录：`weights/LongCat-Video`（视频）与 `weights/LongCat-Video-Avatar-1.5`（数字人），可用环境变量 `LONGCAT_CHECKPOINT_DIR_VIDEO` / `LONGCAT_CHECKPOINT_DIR_AVATAR` 覆盖。
- 若用旧版数字人 v1.0，把 `LONGCAT_CHECKPOINT_DIR_AVATAR` 指向 `weights/LongCat-Video-Avatar`，并在请求里设 `"model_type":"avatar-v1.0"`。

> ⚠️ **数字人强依赖基础视频模型**：`api/scripts/run_avatar_*.py` 里 tokenizer / text_encoder / vae / scheduler 来自**基础视频模型**。代码优先读环境变量 `LONGCAT_CHECKPOINT_DIR_VIDEO`，未设置时回退到 avatar 目录的**兄弟目录** `weights/LongCat-Video`（`checkpoint_dir/../LongCat-Video`）。
> - 默认布局：两个权重都下载到 `weights/` 下且同级（avatar 自动找到兄弟 `LongCat-Video`），即可直接跑。
> - 若基础模型放在别处，设 `LONGCAT_CHECKPOINT_DIR_VIDEO=/path/to/LongCat-Video` 即可，**不再要求必须是兄弟目录**。
> - 缺它会报 `OSError: Incorrect path_or_model_id: '.../LongCat-Video-Avatar-1.5/../LongCat-Video'`（或你自定义的 base 路径）。
> 验证：`ls $LONGCAT_CHECKPOINT_DIR_VIDEO/tokenizer $LONGCAT_CHECKPOINT_DIR_VIDEO/text_encoder $LONGCAT_CHECKPOINT_DIR_VIDEO/vae $LONGCAT_CHECKPOINT_DIR_VIDEO/scheduler` 都应存在。

---

## 3. 生产配置（环境变量）

全部通过环境变量设置，定义在 [`api/config.py`](api/config.py)。常用项：

| 变量 | 默认值 | 生产建议 | 说明 |
|---|---|---|---|
| `LONGCAT_HOST` | `0.0.0.0` | `0.0.0.0` | 监听地址 |
| `LONGCAT_PORT` | `8000` | 与容器暴露端口一致（如 `8080`） | 服务端口，**代码无硬编码，改这个即可** |
| `LONGCAT_EMBED_H5` | `0` | `1` | 开启后 `/` 直接返回 H5 页面 |
| `LONGCAT_AUTH` | `0` | `1` | 开启登录网关，未登录拦截所有接口 |
| `LONGCAT_USER` | `admin` | **改成你的账号** | 登录用户名 |
| `LONGCAT_PASS` | `admin` | **改成强密码** | 登录密码 |
| `LONGCAT_AUTH_TOKEN` | `longcat-demo-token` | 可自定义（HttpOnly cookie 值） | 令牌随机串 |
| `LONGCAT_NUM_GPUS` | `1` | 按可用卡数 | 每个任务的 GPU 数（=torchrun `nproc_per_node`） |
| `LONGCAT_GPU_CONCURRENCY` | `1` | 默认串行 | 同时跑几个任务，多任务需分配不同 `master_port` |
| `LONGCAT_WORK_DIR` | `./api_work` | 保持默认 | 上传 / 输出 / 日志根目录（已建议加入 `.gitignore`） |
| `LONGCAT_CHECKPOINT_DIR_VIDEO` | `weights/LongCat-Video` | 按实际改 | 视频权重目录；**数字人脚本也从此读取共享的 tokenizer/text_encoder/vae/scheduler**（未设则回退到 avatar 目录的兄弟 `LongCat-Video`） |
| `LONGCAT_CHECKPOINT_DIR_AVATAR` | `weights/LongCat-Video-Avatar-1.5` | 按实际改 | 数字人权重目录 |

> H5 页面用的是**相对地址**（`fetch("/health")`、上传 `/files/*`），所以改端口 / 反代域名对前端完全透明，无需改代码。

---

## 4. 启动服务

### 4.1 前台 / 简单后台启动

```bash
cd /opt/ml/cctv/LongCat-Video-API   # 改成你的仓库目录

export LONGCAT_HOST=0.0.0.0
export LONGCAT_PORT=8080            # 与容器暴露端口一致
export LONGCAT_EMBED_H5=1
export LONGCAT_AUTH=1
export LONGCAT_USER=admin
export LONGCAT_PASS='改成强密码'

# 前台运行（Ctrl+C 停止）
uvicorn api.server:app --host 0.0.0.0 --port 8080

# 或后台运行（日志落盘）
nohup uvicorn api.server:app --host 0.0.0.0 --port 8080 \
  > api_work/server.log 2>&1 &
```

> 容器只暴露 `8080` 时，把 `LONGCAT_PORT` 和 `--port` 都设为 `8080`，服务进程内部就监听 8080，与容器端口对齐。

### 4.2 用 systemd 守护（推荐生产）

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
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
systemctl daemon-reload
systemctl enable --now longcat-api
journalctl -u longcat-api -f     # 看日志
```

---

## 5. 访问 H5 页面

1. 浏览器打开 `http://<服务器IP或域名>:8080/`
2. 未登录会自动跳到 `http://<host>:8080/login`
3. 输入 `LONGCAT_USER` / `LONGCAT_PASS`，登录后种下 HttpOnly cookie，自动回到 `/` 加载 H5
4. 在 H5 里：上传图谱（图片）+ 语音（音频）→ 填提示词 → 提交 → 轮询进度 → 下载成片

交互式 API 文档仍在 `http://<host>:8080/docs`（Swagger UI）。

---

## 6. 反向代理 / HTTPS（对外发布时建议）

H5 与 API 同源，用 Nginx 反代一个 8080 即可，无需额外配 CORS：

```nginx
server {
    listen 80;
    server_name your.domain.com;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        client_max_body_size 200m;   # 上传图片/音频较大，调高限制
        proxy_read_timeout 600s;     # 推理耗时，调高超时
    }
}
```

之后用 `certbot` 申请证书即可转 HTTPS。

---

## 7. 更新流程（拉仓库新代码 → 重启）

```bash
cd /opt/ml/cctv/LongCat-Video-API
git pull origin main

# 如果依赖有变动（requirements*.txt 改动），重装
pip install -r requirements_api.txt
pip install -r requirements.txt
pip install -r requirements_avatar.txt

# 重启服务
# 后台进程：kill 掉旧的 uvicorn 再 nohup 起；systemd：systemctl restart longcat-api
```

---

## 8. 常见问题排查

**Q1. `import flash_attn` 报 `undefined symbol ... NSt7__cxx1112basic_string`**
ABI 不匹配。你装成了 `cxx11abiTRUE`，标准 pip torch 用 old ABI。解决：
```bash
pip uninstall -y flash-attn
# 重装 1.3 节的 cxx11abiFALSE wheel
```

**Q1b. 推理子进程报 `ModuleNotFoundError: No module named 'longcat_video'`**
推理脚本在 `api/scripts/` 下，只把脚本自身目录放进 `sys.path`，而 `longcat_video` 包在仓库根，导致 torchrun 子进程找不到它。仓库已在 `api/task_manager.py` 拉起子进程时自动把仓库根加入 `PYTHONPATH`（见第 1 节依赖 / 第 4 节启动）。若你仍遇到：
- 先 `git pull` 确保已包含该修复（`api/task_manager.py` 的 `env["PYTHONPATH"]` 设置）；
- 或临时手动验证：`PYTHONPATH=/opt/ml/cctv/LongCat-Video-API torchrun ... api/scripts/run_avatar_single.py ...`。
- 直接 `python api/scripts/run_avatar_single.py`（不经过服务）也会报同样的错，因为它依赖服务注入的 `PYTHONPATH`。



**Q2. `ModuleNotFoundError: No module named 'transformers'`**
只装了 `requirements_api.txt`。补装：
```bash
pip install -r requirements.txt
```

**Q3. 启动后访问 `/` 看到的是 JSON 而不是 H5**
没开 `LONGCAT_EMBED_H5=1`，或 `h5/index.html` 不存在。确认环境变量生效且文件在仓库 `h5/` 下。

**Q4. 访问被 401 / 一直跳登录页**
开 `LONGCAT_AUTH=1` 后，账号密码默认 `admin/admin`，或你设的 `LONGCAT_USER`/`LONGCAT_PASS`。`/health`、`/login`、`/auth/login` 免登录，其余都拦。

**Q5. 推理卡住 / torchrun 报错 `ChildFailedError`**
- 确认权重目录就位且 `LONGCAT_CHECKPOINT_DIR_*` 指向正确。
- 多任务并行（`LONGCAT_GPU_CONCURRENCY>1`）时，需在 `api/task_manager.py` 给每个任务分配不同 `torchrun --master_port`，否则端口争抢。
- 看具体日志：`GET /tasks/{task_id}/log`。

**Q6. 端口被占用**
确认 `LONGCAT_PORT` 与 `--port` 一致，且没有其他进程占用该端口（`ss -ltnp | grep 8080`）。

**Q7. GPU 不可用（`torch.cuda.is_available()` 为 False）**
检查驱动 / CUDA；容器需带 `--gpus all` 或对应 device 映射；确认 `nvidia-smi` 能看到卡。

---

## 9. 目录与产物

- 上传：`api_work/uploads/{images,videos,audios,jsons}/`
- 产出：`api_work/outputs/<task_id>/`（mp4）
- 日志：`api_work/logs/<task_id>.log` 与 `api_work/tasks.json`（任务状态库，重启后可查）
- 建议把 `api_work/` 加入 `.gitignore`
