import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from traffic_analysis.ffmpeg_tools import (
    SampleRequest,
    build_frame_command,
    build_sample_command,
    cut_sample,
    parse_ffprobe_output,
)

PROBE_JSON = json.dumps(
    {
        "streams": [
            {
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
                "r_frame_rate": "30000/1001",
                "duration": "3600.000000",
            }
        ],
        "format": {"duration": "3600.1"},
    }
)


def test_parse_probe_reads_fraction_fps_and_duration() -> None:
    info = parse_ffprobe_output(PROBE_JSON, video_id="cam01")
    assert info.id == "cam01"
    assert info.fps == pytest.approx(29.97, abs=0.01)
    assert info.duration_s == pytest.approx(3600.0)
    assert (info.width, info.height) == (1920, 1080)


def test_parse_probe_falls_back_to_format_duration() -> None:
    data = json.loads(PROBE_JSON)
    del data["streams"][0]["duration"]
    info = parse_ffprobe_output(json.dumps(data), video_id="cam01")
    assert info.duration_s == pytest.approx(3600.1)


def test_parse_probe_rejects_missing_video_stream() -> None:
    with pytest.raises(ValueError, match="no video stream"):
        parse_ffprobe_output(json.dumps({"streams": []}), video_id="x")


def test_sample_command_seeks_before_input_and_drops_audio(tmp_path: Path) -> None:
    request = SampleRequest(
        source=tmp_path / "in.mp4",
        start_s=600,
        duration_s=300,
        output=tmp_path / "sample.mp4",
    )
    command = build_sample_command(request)
    assert command[command.index("-ss") + 1] == "600.000"
    assert command.index("-ss") < command.index("-i")
    assert "-an" in command
    assert command[-1] == str(tmp_path / "sample.mp4")


def test_frame_command_extracts_single_frame(tmp_path: Path) -> None:
    command = build_frame_command(tmp_path / "in.mp4", 12.5, tmp_path / "f.png")
    assert command[command.index("-ss") + 1] == "12.500"
    assert command[command.index("-frames:v") + 1] == "1"


def test_sample_request_rejects_negative_start(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        SampleRequest(
            source=tmp_path / "in.mp4",
            start_s=-1,
            duration_s=10,
            output=tmp_path / "out.mp4",
        )


def test_cut_sample_fails_clearly_for_missing_video(tmp_path: Path) -> None:
    request = SampleRequest(
        source=tmp_path / "missing.mp4",
        start_s=0,
        duration_s=10,
        output=tmp_path / "out.mp4",
    )
    with pytest.raises(FileNotFoundError):
        cut_sample(request)
