#!/usr/bin/env bash
# 启动 LongCat-Video-API：source .env 后拉起 python -m api.server，
# 并注入 PYTHONPATH 确保 import longcat_video 可用。
set -a
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
[ -f "$SCRIPT_DIR/.env" ] && . "$SCRIPT_DIR/.env"
set +a

# 仓库根目录加入 PYTHONPATH，确保 torchrun / api 进程都能 import longcat_video
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH}"

exec python -m api.server
