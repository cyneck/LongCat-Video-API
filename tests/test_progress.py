from api.progress import PROGRESS_PREFIX, ProgressSnapshot, parse_progress


def event(percent, stage, detail="", current=0, total=0, version=1):
    return (
        f'{PROGRESS_PREFIX}{{"v":{version},"percent":{percent},"stage":"{stage}",'
        f'"detail":"{detail}","current_segment":{current},"total_segments":{total}}}'
    )


def test_plain_logs_never_change_progress():
    fallback = ProgressSnapshot(5, "starting", "启动")
    text = "Loading checkpoint shards\nGenerating segment 1/4\n[longcat][timing] segment_1=12.3s"
    assert parse_progress(text, fallback) == fallback


def test_latest_standard_event_wins():
    text = "\n".join([
        event(10, "model_loading"),
        event(45, "audio_ready"),
        event(56, "video_generation", current=1, total=4),
    ])
    p = parse_progress(text)
    assert p.percent == 56
    assert p.stage == "video_generation"
    assert p.current_segment == 1
    assert p.total_segments == 4


def test_progress_is_monotonic():
    fallback = ProgressSnapshot(70, "video_generation", "segment 2", 2, 4)
    text = "\n".join([event(30, "audio_separation"), event(60, "video_generation")])
    assert parse_progress(text, fallback) == fallback


def test_bad_json_and_unknown_version_are_ignored():
    fallback = ProgressSnapshot(10, "model_loading", "")
    text = "\n".join([
        PROGRESS_PREFIX + "{bad json}",
        event(90, "video_generation", version=2),
    ])
    assert parse_progress(text, fallback) == fallback
