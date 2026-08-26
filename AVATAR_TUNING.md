# Avatar 自然嘴型与提示词调优指南

本文针对 LongCat-Video-Avatar v1.5 在真人数字人场景中可能出现的 **嘴巴张幅偏大、表情过强、头部动作过度** 等现象，给出本 API 的推荐参数与提示词写法。

## 1. `audio_drive_gain`：优先调这个参数

API 的单人和多人数字人请求都支持：

```json
{
  "audio_drive_gain": 0.85
}
```

默认值为 `0.85`，允许范围 `0.5 ~ 1.2`。

它作用在 Whisper/Wav2Vec 已提取出的 **audio embedding** 上，而不是修改 WAV 音量，因此：

- 不会改变最终视频里的声音响度；
- 不会被 Whisper 前的 LUFS loudness normalization 抵消；
- 在 Avatar v1.5 distill 8-step 模式下，即使 `audio_guidance_scale` 被固定为 `1.0`，仍然可以独立调节嘴部/表情驱动强度。

推荐从以下三档做 A/B：

| `audio_drive_gain` | 建议用途 | 预期效果 |
|---:|---|---|
| `1.00` | 接近模型原始驱动 | 口型最强，可能更夸张 |
| `0.85` | **默认推荐** | 在同步性和自然度之间折中 |
| `0.70` | 大嘴/表情过强明显时 | 更克制，但过低可能降低口型同步感 |

不建议一开始低于 `0.70`。如果人物几乎不张嘴或同步变弱，应往 `0.8 ~ 0.9` 回调。

多人模式中，该增益只作用于 person1/person2 的说话人 embedding；`others` 使用的背景静音 embedding 不做衰减。

## 2. 为什么不直接调低 WAV 音量

Avatar v1.5 的 Whisper 音频特征提取前会先做 loudness normalization。单纯将分离后的波形乘 `0.8`，随后仍可能被归一化回目标响度，因此不能稳定控制嘴型。

本项目把增益放在 audio encoder **之后**：

```text
原始音频
  ↓
人声分离
  ↓
LUFS normalization
  ↓
Whisper / Wav2Vec
  ↓
audio embedding
  ↓
audio_drive_gain = 0.85   ← 在这里生效
  ↓
Avatar DiT
```

最终 mux 仍使用原始上传音频。

## 3. Avatar v1.5 提示词原则

数字人 prompt 不宜只写：

```text
A person is talking.
```

这种描述给模型留下较大的表情和身体动作自由度。若目标是企业数字人、课程讲解、产品介绍，应明确告诉模型：

1. **说话方式**：calmly / naturally / conversationally；
2. **嘴部动作**：subtle natural lip movements；
3. **表情**：relaxed / neutral / gentle；
4. **头部动作**：minimal head movement；
5. **身份稳定性**：stable facial features / consistent identity；
6. **镜头稳定**：steady camera / medium shot / frontal view。

尽量描述“希望出现什么”，不要堆大量相互冲突的否定词。

## 4. 单人数字人提示词示例

### 企业讲解 / 新闻播报

```text
A professional presenter speaking calmly to the camera in a medium shot,
with subtle natural lip movements, a relaxed facial expression,
minimal head movement, natural blinking, stable facial features,
and a steady camera. The speaking style is composed and conversational.
```

### 中文语义版本

```text
一位专业讲解者正平静自然地面对镜头讲话，中景构图，嘴部动作轻微自然，
面部表情放松克制，头部动作较少，自然眨眼，人物五官和身份保持稳定，
镜头稳定，整体表达自然、不夸张。
```

### 培训 / 课程讲师

```text
A teacher explaining a topic clearly and calmly to the camera,
with restrained facial expressions, subtle lip movement,
small natural head gestures, stable identity, natural blinking,
and a clean stationary background.
```

### 轻微微笑但避免夸张

```text
A person speaking naturally with a gentle slight smile,
subtle mouth movement, relaxed cheeks and jaw,
minimal expressive exaggeration, stable facial features,
and a steady frontal camera.
```

## 5. 多人对谈提示词示例

```text
Two people having a calm natural conversation in a medium shot.
Each speaker uses subtle natural lip movements and restrained facial expressions.
Their faces and identities remain stable, head movements are small and natural,
and the camera remains steady throughout the conversation.
```

中文版本：

```text
两位人物进行平静自然的对谈，中景构图。说话者嘴部动作轻微自然，表情克制，
两人的五官和身份保持稳定，头部仅有小幅自然动作，镜头全程稳定，
避免夸张的张嘴、表情和身体动作。
```

多人场景建议明确谁在画面左侧/右侧，以及服装或人物特征，帮助模型保持角色一致：

```text
Two people are seated side by side. The person on the left wears a dark blue shirt,
and the person on the right wears a light gray shirt. They have a calm conversation...
```

## 6. 建议的调参顺序

发现“大嘴巴”时，不要一次修改多个变量。推荐按以下顺序：

```text
同一张图 + 同一段音频 + 同一 seed
    ↓
audio_drive_gain = 1.00
    ↓
audio_drive_gain = 0.85
    ↓
audio_drive_gain = 0.70
```

先选最自然的一档，再优化 prompt。

推荐基线：

```json
{
  "model_type": "avatar-v1.5",
  "resolution": "480p",
  "num_segments": "auto",
  "num_inference_steps": 8,
  "use_distill": true,
  "use_int8": true,
  "text_guidance_scale": 1.0,
  "audio_guidance_scale": 1.0,
  "audio_drive_gain": 0.85,
  "seed": 42
}
```

在当前 A100 40GB profile + v1.5 distill 模式下，`text_guidance_scale` / `audio_guidance_scale` 会归一化到 `1.0`；若要控制嘴部驱动力，优先使用 `audio_drive_gain`。

## 7. 参考图片也很重要

若参数和 prompt 都正常，但仍容易出现嘴部形变，建议检查参考图：

- 优先使用嘴巴自然闭合或轻微放松的照片；
- 避免原图本身大幅张嘴、夸张笑容或明显露齿；
- 避免脸部过度贴近画面边缘；
- 清晰正脸或轻微侧脸通常比强侧脸稳定；
- 光照均匀、面部无遮挡更有利于身份和嘴部稳定。

## 8. 日志确认

新版本运行时会输出：

```text
[longcat][audio] audio_drive_gain=0.85 applied to audio embedding (output audio volume unchanged)
```

多人任务则会显示 speaker embeddings 已应用增益。可以据此确认任务实际使用了新参数。
