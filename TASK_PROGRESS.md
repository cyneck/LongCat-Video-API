# Task Progress Event Protocol

LongCat-Video-API 使用独立的机器可读进度事件协议。**任务进度不得依赖普通 debug/timing 日志文本。**

## 1. 标准格式

每条任务进度必须由 `api.progress.progress_event()` 发出，stdout 线上格式固定为：

```text
[longcat][progress] {"v":1,"percent":45,"stage":"audio_ready","detail":"音频特征已完成，准备生成视频","current_segment":0,"total_segments":4}
```

协议前缀固定：

```text
[longcat][progress] 
```

前缀后的内容必须是单行 JSON。

## 2. 字段

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `v` | int | 当前为 `1` | 协议版本 |
| `percent` | int | `0..100` | 总任务进度，必须单调不下降 |
| `stage` | string | 稳定枚举 | 机器读取的阶段 ID，不使用中文文案作为协议值 |
| `detail` | string | 可变 | 用户可读的阶段描述，可调整文案 |
| `current_segment` | int | `>=0` | 当前视频 segment；非视频阶段为 0 |
| `total_segments` | int | `>=0` | 总 segment 数；未知时为 0 |
| `meta` | object | 可选 | 扩展诊断元数据；消费者必须允许未知字段 |

## 3. Stage 枚举与默认百分比

| stage | UI 名称 | 进度 |
|---|---|---:|
| `queued` | 排队中 | 0% |
| `starting` | 启动任务 | 5% |
| `model_loading` | 加载模型 | 10% |
| `model_ready` | 模型就绪 | 25% |
| `audio_separation` | 人声分离 | 30% |
| `audio_features` | 音频特征 | 35% |
| `audio_ready` | 音频特征完成 | 45% |
| `video_generation` | 生成视频 | 45–90% |
| `muxing` | 合并封装 | 95% |
| `completed` | 已完成 | 100% |
| `failed` | 任务失败 | 保留失败前最后进度 |

视频生成区间 `45..90` 按实际 `frame_plan` 的 segment 数均分。每段开始和完成都会发事件。

## 4. 生产约束

1. 推理脚本不得手写 `[longcat][progress]` 文本，统一调用 `progress_event()` / `install_worker_progress_hooks()`。
2. 普通 `print()`、`[longcat][timing]`、第三方 tqdm、PyTorch/FFmpeg/ONNX 日志都不是进度协议。
3. TaskManager 只能解析 `PROGRESS_PREFIX + JSON`，禁止用自然语言正则猜测任务进度。
4. 多卡 `torchrun` 只允许 global rank 0 发任务级进度，防止重复和乱序。
5. `percent` 必须单调不下降。消费者遇到低于当前百分比的旧事件必须忽略。
6. `stage` 是 API 契约，应保持稳定；UI 文案通过 `stage_label` / `detail` 展示。
7. 未知协议版本必须忽略，而不是尝试兼容性猜测。
8. 新阶段需要先更新协议文档与 `STAGE_LABELS`，再由 worker 发事件。

## 5. API

`GET /tasks/{task_id}` 与 `GET /tasks` 返回：

```json
{
  "progress": {
    "percent": 67,
    "stage": "video_generation",
    "stage_label": "生成视频",
    "detail": "正在生成第 2/4 段",
    "current_segment": 2,
    "total_segments": 4
  }
}
```

H5 只消费该对象展示进度，不再生成固定 70% 的假进度。
