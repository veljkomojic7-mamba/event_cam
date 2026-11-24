"""Background analysis jobs for the GUI.

One folder per run under the jobs directory:
  job.json     the request (video, scene, settings, time window)
  status.json  state, stage and progress, refreshed while the job runs
  log.txt      everything the worker printed
  out/         the normal pipeline outputs (stats.json, report.md, clips/, ...)

The worker is a separate process, so a closed browser tab or a Streamlit rerun
does not stop a long run.
"""

import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import Field

from traffic_analysis.ffmpeg_tools import probe_video
from traffic_analysis.pipeline import (
    stage_analyze,
    stage_clips,
    stage_detect,
    stage_events,
    stage_report,
)
from traffic_analysis.schemas import (
    AnalysisSettings,
    StrictModel,
    load_scene_config,
    save_model,
)

JOB_FILE = "job.json"
STATUS_FILE = "status.json"
LOG_FILE = "log.txt"
OUT_DIR = "out"
HEARTBEAT_S = 10.0
STALE_AFTER_S = 90.0
DETECT_SHARE = 0.9
STAGES = ("detect", "analyze", "events", "clips", "report")
PROGRESS_PATTERN = re.compile(r"\(t=([\d.]+)s\)")

State = Literal["queued", "running", "done", "failed", "cancelled"]
logger = logging.getLogger(__name__)


class JobSpec(StrictModel):
    """What to run. Paths are absolute so the worker needs no other context."""

    video: str
    scene: str
    calibration: str | None = None
    settings: AnalysisSettings = Field(default_factory=AnalysisSettings)
    start_s: float = Field(default=0.0, ge=0)
    end_s: float | None = Field(default=None, gt=0)
    label: str = ""
    backend: Literal["yolo", "gemini"] = "yolo"
    gemini_consent: bool = False


class JobStatus(StrictModel):
    """Progress of one job, rewritten often while it runs."""

    state: State
    stage: str = ""
    progress: float = Field(default=0.0, ge=0, le=1)
    message: str = ""
    pid: int | None = None
    created_at: float
    updated_at: float
    finished_at: float | None = None


def create_job(jobs_dir: Path, spec: JobSpec) -> Path:
    """Write a new queued job folder and return it."""
    job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    job_dir = jobs_dir / job_id
    job_dir.mkdir(parents=True)
    save_model(spec, job_dir / JOB_FILE)
    now = time.time()
    _write_status(job_dir, JobStatus(state="queued", created_at=now, updated_at=now))
    return job_dir


def start_job(job_dir: Path) -> None:
    """Launch the worker as its own process and return at once."""
    log = (job_dir / LOG_FILE).open("ab")
    subprocess.Popen(
        [sys.executable, "-m", "traffic_analysis.jobs", str(job_dir)],
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    log.close()


def load_spec(job_dir: Path) -> JobSpec:
    return JobSpec.model_validate_json((job_dir / JOB_FILE).read_text(encoding="utf-8"))


def read_status(job_dir: Path) -> JobStatus:
    """Current status. A running job that stopped updating is reported as failed."""
    status = JobStatus.model_validate_json(
        (job_dir / STATUS_FILE).read_text(encoding="utf-8")
    )
    if status.state == "running" and time.time() - status.updated_at > STALE_AFTER_S:
        return status.model_copy(
            update={"state": "failed", "message": "worker stopped responding"}
        )
    return status


def list_jobs(jobs_dir: Path) -> list[Path]:
    """Job folders, newest first."""
    if not jobs_dir.is_dir():
        return []
    found = [p for p in jobs_dir.iterdir() if (p / STATUS_FILE).is_file()]
    return sorted(found, key=lambda p: p.name, reverse=True)


def cancel_job(job_dir: Path) -> None:
    """Stop a running worker and mark the job cancelled."""
    status = read_status(job_dir)
    if status.state not in ("queued", "running"):
        return
    if status.pid is not None:
        try:
            os.kill(status.pid, signal.SIGTERM)
        except OSError:
            logger.info("worker %s already gone", status.pid)
    _write_status(
        job_dir,
        status.model_copy(
            update={
                "state": "cancelled",
                "message": "cancelled by user",
                "updated_at": time.time(),
                "finished_at": time.time(),
            }
        ),
    )


def _write_status(job_dir: Path, status: JobStatus) -> None:
    temp = job_dir / f"{STATUS_FILE}.tmp"
    temp.write_text(status.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temp.replace(job_dir / STATUS_FILE)


class _Progress:
    """Keeps status.json fresh: updates on events and on a heartbeat."""

    def __init__(self, job_dir: Path) -> None:
        self._job_dir = job_dir
        self._lock = threading.Lock()
        self._status = read_status(job_dir).model_copy(
            update={"state": "running", "pid": os.getpid()}
        )
        self._stop = threading.Event()
        self._beat = threading.Thread(target=self._heartbeat, daemon=True)

    def start(self) -> None:
        self._flush()
        self._beat.start()

    def update(self, **changes: object) -> None:
        with self._lock:
            self._status = self._status.model_copy(update=changes)
            self._flush_locked()

    def finish(self, state: State, message: str = "") -> None:
        self._stop.set()
        now = time.time()
        self.update(state=state, message=message, finished_at=now)

    def _flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        self._status = self._status.model_copy(update={"updated_at": time.time()})
        _write_status(self._job_dir, self._status)

    def _heartbeat(self) -> None:
        while not self._stop.wait(HEARTBEAT_S):
            self._flush()


class _DetectProgressHandler(logging.Handler):
    """Turns the detector's periodic log lines into a progress fraction."""

    def __init__(self, progress: _Progress, start_s: float, end_s: float) -> None:
        super().__init__(level=logging.INFO)
        self._progress = progress
        self._start_s = start_s
        self._span = max(end_s - start_s, 1e-6)

    def emit(self, record: logging.LogRecord) -> None:
        match = PROGRESS_PATTERN.search(record.getMessage())
        if match is None:
            return
        done = (float(match.group(1)) - self._start_s) / self._span
        self._progress.update(
            progress=round(DETECT_SHARE * min(max(done, 0.0), 1.0), 4)
        )


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _effective_settings(settings: AnalysisSettings) -> AnalysisSettings:
    """Use half precision on a CUDA GPU unless the request chose a device."""
    if settings.device is None and not settings.half and _cuda_available():
        return settings.model_copy(update={"half": True})
    return settings


def run_job(job_dir: Path) -> None:
    """Run every stage for one job. Writes status as it goes."""
    spec = load_spec(job_dir)
    out_dir = job_dir / OUT_DIR
    progress = _Progress(job_dir)
    progress.start()
    video = Path(spec.video)
    handler: _DetectProgressHandler | None = None
    try:
        settings = _effective_settings(spec.settings)
        scene = load_scene_config(Path(spec.scene))
        calibration = Path(spec.calibration) if spec.calibration else None
        duration = probe_video(video).duration_s
        end_s = min(spec.end_s, duration) if spec.end_s is not None else duration
        if spec.backend == "gemini":
            from traffic_analysis.gemini_count import run_gemini_count

            if not spec.gemini_consent:
                raise ValueError("Gemini frame upload was not explicitly authorized")
            progress.update(stage="gemini", message="Preparing sampled frames")
            run_gemini_count(
                video,
                scene,
                out_dir,
                spec.start_s,
                end_s,
                settings.detect_fps,
                lambda done, message: progress.update(progress=done, message=message),
            )
            progress.finish("done")
            return
        handler = _DetectProgressHandler(progress, spec.start_s, end_s)
        logging.getLogger("traffic_analysis.detection").addHandler(handler)

        progress.update(stage="detect", message=f"half precision: {settings.half}")
        stage_detect(video, out_dir, settings, spec.start_s, end_s)
        progress.update(stage="analyze", progress=DETECT_SHARE)
        stage_analyze(video, scene, out_dir, settings, calibration)
        progress.update(stage="events", progress=0.93)
        stage_events(video, out_dir, settings, calibration, scene)
        progress.update(stage="clips", progress=0.95)
        stage_clips(video, out_dir, settings)
        progress.update(stage="report", progress=0.99)
        stage_report(out_dir, narrate_with_cloud=False, allow_cloud=False)
        progress.update(progress=1.0)
        progress.finish("done")
    except Exception as exc:
        logger.exception("job failed")
        progress.finish("failed", f"{type(exc).__name__}: {exc}")
    finally:
        if handler is not None:
            logging.getLogger("traffic_analysis.detection").removeHandler(handler)


def main(argv: list[str] | None = None) -> int:
    """Worker entry point: ``python -m traffic_analysis.jobs <job_dir>``."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m traffic_analysis.jobs <job_dir>", file=sys.stderr)
        return 2
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_job(Path(args[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
