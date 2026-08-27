from pathlib import Path

from api.schemas import AvatarMultiRequest, AvatarSingleRequest


def test_v1_single_normalizes_v15_only_flags():
    request = AvatarSingleRequest(
        prompt="speaker",
        cond_audio={"person1": "/tmp/audio.wav"},
        model_type="avatar-v1.0",
        use_int8=True,
        use_distill=True,
    )
    assert request.use_int8 is False
    assert request.use_distill is False


def test_v1_multi_normalizes_v15_only_flags():
    request = AvatarMultiRequest(
        prompt="speakers",
        cond_image="/tmp/image.png",
        cond_audio={"person1": "/tmp/audio.wav"},
        model_type="avatar-v1.0",
        use_int8=True,
        use_distill=True,
    )
    assert request.use_int8 is False
    assert request.use_distill is False


def test_h5_omits_v15_flags_for_v1_requests():
    source = (Path(__file__).parents[1] / "h5" / "index.html").read_text(encoding="utf-8")
    assert 'if (model === "avatar-v1.5")' in source
    assert 'label.hidden = true' in source


def test_pipeline_keeps_text_shape_metadata_after_offload():
    source = (Path(__file__).parents[1] / "longcat_video" / "pipeline_longcat_video_avatar.py").read_text(encoding="utf-8")
    assert "self.text_encoder_d_model = int(text_encoder.config.d_model)" in source
    assert "self.text_encoder = None" in source
    assert "self.text_encoder_d_model" in source
