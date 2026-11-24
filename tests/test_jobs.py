"""Job folders and worker status used by the GUI."""

import os
import time
from pathlib import Path

import pytest

from traffic_analysis import jobs
from traffic_analysis.jobs import (
    JobSpec,
    JobStatus,
    cancel_job,
    create_job,
    list_jobs,
    load_spec,
    read_status,
)


def _spec() -> JobSpec:
    return JobSpec(video="/v/a.mp4", scene="/s/scene.json", label="a.mp4")


def _set_status(job_dir: Path, **changes: object) -> None:
    status = read_status(job_dir).model_copy(update=changes)
    jobs._write_status(job_dir, status)  # noqa: SLF001


def test_create_job_writes_spec_and_queued_status(tmp_path: Path) -> None:
    job_dir = create_job(tmp_path, _spec())
    assert load_spec(job_dir) == _spec()
    status = read_status(job_dir)
    assert status.state == "queued"
    assert status.progress == 0


def test_list_jobs_newest_first(tmp_path: Path) -> None:
    first = create_job(tmp_path, _spec())
    time.sleep(1.1)
    second = create_job(tmp_path, _spec())
    assert list_jobs(tmp_path) == [second, first]
    assert list_jobs(tmp_path / "missing") == []


def test_running_job_that_stops_updating_is_failed(tmp_path: Path) -> None:
    job_dir = create_job(tmp_path, _spec())
    _set_status(job_dir, state="running", updated_at=time.time())
    assert read_status(job_dir).state == "running"
    _set_status(job_dir, updated_at=time.time() - jobs.STALE_AFTER_S - 5)
    status = read_status(job_dir)
    assert status.state == "failed"
    assert "stopped" in status.message


def test_cancel_marks_cancelled_and_ignores_finished(tmp_path: Path) -> None:
    job_dir = create_job(tmp_path, _spec())
    cancel_job(job_dir)
    assert read_status(job_dir).state == "cancelled"
    done = create_job(tmp_path, _spec())
    _set_status(done, state="done")
    cancel_job(done)
    assert read_status(done).state == "done"


def test_detect_log_lines_become_progress(tmp_path: Path) -> None:
    job_dir = create_job(tmp_path, _spec())
    progress = jobs._Progress(job_dir)  # noqa: SLF001
    handler = jobs._DetectProgressHandler(progress, 100.0, 200.0)  # noqa: SLF001
    import logging

    record = logging.LogRecord(
        "traffic_analysis.detection",
        logging.INFO,
        __file__,
        1,
        "pass %d: %d frames processed (t=%.1fs)",
        (1, 200, 150.0),
        None,
    )
    handler.emit(record)
    status: JobStatus = read_status(job_dir)
    assert status.state == "running"
    assert status.pid == os.getpid()
    assert status.progress == pytest.approx(0.45)


def test_gemini_without_consent_does_not_send_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unittest.mock import Mock

    from traffic_analysis import gemini_count
    from traffic_analysis.schemas import SceneConfig

    spec = JobSpec(video="v", scene="s", backend="gemini")
    job = create_job(tmp_path, spec)
    monkeypatch.setattr(
        jobs,
        "load_scene_config",
        lambda _: SceneConfig(camera_id="c", frame_width=10, frame_height=10),
    )
    monkeypatch.setattr(
        jobs, "probe_video", lambda _: type("Info", (), {"duration_s": 30.0})()
    )
    send = Mock()
    monkeypatch.setattr(gemini_count, "run_gemini_count", send)
    jobs.run_job(job)
    assert read_status(job).state == "failed"
    assert "not explicitly authorized" in read_status(job).message
    send.assert_not_called()
