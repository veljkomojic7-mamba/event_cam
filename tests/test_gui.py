"""Smoke tests for the Streamlit app and its helpers."""

import sys
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from traffic_analysis import gui
from traffic_analysis.analytics import (
    RingFirstSeen,
    RoundaboutStats,
    StatsReport,
    WrongWayExample,
)
from traffic_analysis.jobs import JobSpec, _write_status, create_job, read_status
from traffic_analysis.schemas import Point2D, SceneConfig, Zone, save_model

GUI_FILE = Path(gui.__file__)


def _app(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.setattr(sys, "argv", ["gui.py", "--data-dir", str(data_dir)])
    monkeypatch.chdir(data_dir)
    return AppTest.from_file(str(GUI_FILE), default_timeout=30)


def test_scene_files_skip_files_that_are_not_scenes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    scenes = tmp_path / "scenes"
    scenes.mkdir()
    (scenes / "clip.json").write_text('{"source": "a.mp4", "start_s": 60.0}')
    scene = SceneConfig(camera_id="c", frame_width=10, frame_height=10)
    save_model(scene, scenes / "good.json")
    assert [p.name for p in gui.scene_files(tmp_path)] == ["good.json"]


def test_safe_name_strips_paths_and_odd_characters() -> None:
    assert gui.safe_name("../../etc/passwd") == "passwd"
    assert gui.safe_name("my video (1).mp4") == "my_video_1_.mp4"
    assert gui.safe_name("...") == "upload"


def test_format_duration() -> None:
    assert gui.format_duration(45) == "45 s"
    assert gui.format_duration(125) == "2 min 05 s"
    assert gui.format_duration(3725) == "1 h 02 min"


@pytest.mark.parametrize(
    ("label", "model"),
    [
        ("Standard YOLO11m", "yolo11m.pt"),
        ("Standard YOLO11l (large)", "yolo11l.pt"),
        ("Standard YOLO11x (extra large)", "yolo11x.pt"),
        ("Aerial YOLO11m-OBB (experimental)", "yolo11m-obb.pt"),
        ("Gemini frame review (experimental)", "gemini-frames"),
    ],
)
def test_options_select_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, label: str, model: str
) -> None:
    st = Mock()
    st.selectbox.return_value = label
    columns = [Mock(), Mock(), Mock()]
    columns[0].slider.return_value = 10
    columns[1].slider.return_value = 0.1
    columns[2].selectbox.return_value = 1280
    st.columns.return_value = columns
    st.checkbox.return_value = True
    monkeypatch.setattr(gui, "st", st)

    settings, start, end, calibration = gui.options_section(tmp_path, 60.0)

    assert settings.model == model
    assert (start, end, calibration) == (0.0, None, None)
    assert st.warning.call_count == (
        1 if model in (gui.AERIAL_MODEL, gui.GEMINI_MODEL) else 0
    )


def test_draw_zones_marks_pixels_and_downsizes() -> None:
    scene = SceneConfig(
        camera_id="c",
        frame_width=2000,
        frame_height=1000,
        zones=[
            Zone(
                name="ring",
                role="ring",
                polygon=[
                    Point2D(x=100, y=100),
                    Point2D(x=900, y=100),
                    Point2D(x=500, y=800),
                ],
            )
        ],
    )
    frame = np.zeros((1000, 2000, 3), dtype=np.uint8)
    out = gui.draw_zones(frame, scene, max_width=1000)
    assert out.shape == (500, 1000, 3)
    assert out.any()


def test_new_run_page_without_videos_shows_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _app(tmp_path, monkeypatch).run()
    assert not at.exception
    assert any("No videos" in i.value for i in at.info)


def test_results_page_shows_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = create_job(
        tmp_path / "jobs", JobSpec(video="/v.mp4", scene="/s.json", label="v.mp4")
    )
    _write_status(
        job_dir,
        read_status(job_dir).model_copy(
            update={"state": "done", "progress": 1.0, "updated_at": time.time()}
        ),
    )
    stats = StatsReport(
        video_id="v",
        calibration_id=None,
        total_vehicles=5,
        by_class={"car": 5},
        entries_by_zone={},
        exits_by_zone={},
        line_crossings={},
        od_matrix=[],
        time_bins=[],
        speed_by_class={},
        mean_id_switches=0.0,
        mean_occluded_frac=0.0,
        pedestrian_tracks=0,
        roundabout=RoundaboutStats(
            entered=4,
            exited=3,
            wrong_way_entries=1,
            wrong_way_exits=0,
            entered_by_arm={"n_in": 3, "n_out": 1},
            exited_by_arm={"e_out": 3},
            wrong_way_entries_by_arm={"n_out": 1},
            wrong_way_examples=[WrongWayExample(track_id=9, arm="n_out", t_s=12.0)],
            first_seen_on_ring=[RingFirstSeen(track_id=10, t_s=39.3)],
        ),
        caveats=["check by hand"],
    )
    save_model(stats, job_dir / "out" / "stats.json")

    at = _app(tmp_path, monkeypatch)
    at.session_state["page"] = "Results"
    at.run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Entered"] == "4"
    assert metrics["Wrong-way entries"] == "1"
    assert metrics["Tracks first seen on ring"] == "1"
    assert any("Track 9" in m.value for m in at.markdown)
    assert any("yolo11m.pt" in c.value for c in at.caption)
