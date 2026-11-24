"""Gemini frame review tests without external API calls."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from google import genai

from traffic_analysis import gemini_count as gc
from traffic_analysis.schemas import SceneConfig


def test_sends_timestamped_frames_and_writes_separate_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = SceneConfig(camera_id="test", frame_width=100, frame_height=100)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        gc,
        "iter_sampled_frames",
        lambda *args: iter([(0, 0.0, frame, False), (1, 5.0, frame, False)]),
    )
    models = Mock()
    models.generate_content.return_value = SimpleNamespace(
        text='{"events":[],"notes":"uncertain"}'
    )
    monkeypatch.setattr(
        genai, "Client", Mock(return_value=SimpleNamespace(models=models))
    )
    progress = Mock()
    report = gc.run_gemini_count(
        Path("video.mp4"), scene, tmp_path, 0.0, 10.0, 1.0, progress
    )
    assert report.frames_sent == 2
    assert [b.frame_count for b in report.batches] == [1, 1]
    assert models.generate_content.call_count == 2
    parts = models.generate_content.call_args_list[0].kwargs["contents"].parts
    assert "scene" in parts[0].text.lower()
    assert "0.000" in parts[1].text
    assert parts[2].inline_data.mime_type == "image/jpeg"
    assert json.loads((tmp_path / gc.OUTPUT).read_text())["frames_sent"] == 2
    assert progress.call_count == 2


def test_limits_and_missing_key_fail_before_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = SceneConfig(camera_id="test", frame_width=100, frame_height=100)
    client = Mock()
    monkeypatch.setattr(genai, "Client", client)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="60 seconds"):
        gc.run_gemini_count(Path("v"), scene, tmp_path, 0, 61, 10, Mock())
    with pytest.raises(ValueError, match="10 fps"):
        gc.run_gemini_count(Path("v"), scene, tmp_path, 0, 60, 15, Mock())
    with pytest.raises(RuntimeError, match="not configured"):
        gc.run_gemini_count(Path("v"), scene, tmp_path, 0, 60, 10, Mock())
    client.assert_not_called()


def test_rejects_out_of_window_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = SceneConfig(camera_id="test", frame_width=100, frame_height=100)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        gc, "iter_sampled_frames", lambda *args: iter([(0, 0.0, frame, False)])
    )
    models = Mock()
    models.generate_content.return_value = SimpleNamespace(
        text='{"events":[{"kind":"entry","time_s":6,"arm":"north",'
        '"description":"car"}],"notes":""}'
    )
    monkeypatch.setattr(
        genai, "Client", Mock(return_value=SimpleNamespace(models=models))
    )
    with pytest.raises(ValueError, match="outside"):
        gc.run_gemini_count(Path("v"), scene, tmp_path, 0, 5, 1, Mock())
    assert not (tmp_path / gc.OUTPUT).exists()


def test_sixty_seconds_at_ten_fps_sends_600_frames_in_12_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scene = SceneConfig(camera_id="test", frame_width=10, frame_height=10)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        gc,
        "iter_sampled_frames",
        lambda *args: ((i, i / 10, frame, False) for i in range(600)),
    )
    models = Mock()
    models.generate_content.return_value = SimpleNamespace(
        text='{"events":[],"notes":""}'
    )
    monkeypatch.setattr(
        genai, "Client", Mock(return_value=SimpleNamespace(models=models))
    )
    report = gc.run_gemini_count(Path("v"), scene, tmp_path, 0, 60, 10, Mock())
    assert report.frames_sent == 600
    assert len(report.batches) == 12
    assert [b.frame_count for b in report.batches] == [50] * 12
    assert models.generate_content.call_count == 12
