"""Pipeline stages. Each function reads the previous stage's files and writes its own.

Output layout under the chosen out_dir:
  observations.sqlite  pass-1 tracker output (detect)
  vehicles.json        per-vehicle records (analyze)
  stats.json           aggregates, computed in code (analyze)
  events.json          ranked events and planned pass-2 windows (events)
  clips/               evidence clips and sidecars (clips)
  model_calls.jsonl    log of every cloud call (review)
  report.md            deterministic report (report)
  narrative.md         optional LLM summary, only if grounded (report)
"""

import logging
import os
from pathlib import Path

from pydantic import TypeAdapter

from traffic_analysis.analytics import (
    VEHICLE_CLASSES,
    StatsReport,
    build_stats,
    build_tracks,
    build_vehicle_record,
    wrong_way_events,
)
from traffic_analysis.calibration import build_calibration, homography_matrix
from traffic_analysis.clips import ClipSidecar, cut_event_clips
from traffic_analysis.conflicts import build_paths, detect_events, plan_pass2_windows
from traffic_analysis.detection import DetectionSummary, run_detection
from traffic_analysis.detector import YoloTracker
from traffic_analysis.ffmpeg_tools import probe_video
from traffic_analysis.geometry import FloatArray
from traffic_analysis.report import narrate, render_report
from traffic_analysis.review import (
    ClipReviewer,
    GeminiReviewer,
    NullReviewer,
    ReviewBudget,
    review_event,
)
from traffic_analysis.schemas import (
    AnalysisSettings,
    CalibrationFile,
    CalibrationPoint,
    EventRecord,
    SceneConfig,
    StrictModel,
    VehicleRecord,
    load_calibration,
    save_model,
)
from traffic_analysis.store import clear_pass, load_observations, open_store

logger = logging.getLogger(__name__)

OBSERVATIONS_DB = "observations.sqlite"
VEHICLES_JSON = "vehicles.json"
STATS_JSON = "stats.json"
EVENTS_JSON = "events.json"
CLIPS_DIR = "clips"
MODEL_LOG = "model_calls.jsonl"
REPORT_MD = "report.md"
NARRATIVE_MD = "narrative.md"
PASS_1 = 1


class VehiclesFile(StrictModel):
    """Per-vehicle records for one video."""

    video_id: str
    vehicles: list[VehicleRecord]


class EventsFile(StrictModel):
    """Ranked events and the padded windows planned for a second pass."""

    video_id: str
    events: list[EventRecord]
    pass2_windows: list[tuple[float, float]] = []


def _require_db(out_dir: Path) -> Path:
    db = out_dir / OBSERVATIONS_DB
    if not db.is_file():
        raise FileNotFoundError(f"no observations at {db}; run detect first")
    return db


def stage_detect(
    video: Path,
    out_dir: Path,
    settings: AnalysisSettings,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> DetectionSummary:
    """Pass 1: track vehicles and pedestrians over the whole video."""
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = open_store(out_dir / OBSERVATIONS_DB)
    try:
        clear_pass(conn, PASS_1)
        tracker = YoloTracker(settings)
        return run_detection(
            video, conn, tracker, settings, pass_no=PASS_1, start_s=start_s, end_s=end_s
        )
    finally:
        conn.close()


def stage_calibrate(
    camera_id: str, points_path: Path, out_path: Path
) -> CalibrationFile:
    """Fit the ground-plane homography from a JSON list of image/world point pairs."""
    if not points_path.is_file():
        raise FileNotFoundError(f"points file not found: {points_path}")
    points = TypeAdapter(list[CalibrationPoint]).validate_json(
        points_path.read_text(encoding="utf-8")
    )
    calibration = build_calibration(camera_id, points)
    save_model(calibration, out_path)
    logger.info(
        "calibration %s: rms %.3f m, max %.3f m",
        calibration.calibration_id,
        calibration.rms_error_m,
        calibration.max_error_m,
    )
    return calibration


def _load_homography(calibration_path: Path) -> tuple[FloatArray, CalibrationFile]:
    calibration = load_calibration(calibration_path)
    return homography_matrix(calibration), calibration


def _check_scene_calibration(scene: SceneConfig, calibration: CalibrationFile) -> None:
    if (
        scene.calibration_id is not None
        and scene.calibration_id != calibration.calibration_id
    ):
        raise ValueError(
            f"scene expects calibration {scene.calibration_id}, "
            f"but the file is {calibration.calibration_id}"
        )


def stage_analyze(
    video: Path,
    scene: SceneConfig,
    out_dir: Path,
    settings: AnalysisSettings,
    calibration_path: Path | None,
) -> StatsReport:
    """Build tracks, count per zone and line, compute O-D, time bins and speeds."""
    conn = open_store(_require_db(out_dir))
    try:
        observations = load_observations(conn, PASS_1)
    finally:
        conn.close()
    info = probe_video(video)
    homography: FloatArray | None = None
    calibration_id: str | None = None
    if calibration_path is not None:
        homography, calibration = _load_homography(calibration_path)
        _check_scene_calibration(scene, calibration)
        calibration_id = calibration.calibration_id
    tracks = build_tracks(observations, settings)
    records = [
        build_vehicle_record(track, scene, settings, settings.detect_fps, homography)
        for track in tracks
        if track.class_name in VEHICLE_CLASSES
    ]
    stats = build_stats(info.id, scene, settings, tracks, records, calibration_id)
    save_model(
        VehiclesFile(video_id=info.id, vehicles=records), out_dir / VEHICLES_JSON
    )
    save_model(stats, out_dir / STATS_JSON)
    logger.info("analysed %d vehicles", stats.total_vehicles)
    return stats


def _write_events(
    out_dir: Path,
    video_id: str,
    events: list[EventRecord],
    windows: list[tuple[float, float]],
) -> None:
    save_model(
        EventsFile(video_id=video_id, events=events, pass2_windows=windows),
        out_dir / EVENTS_JSON,
    )


def load_events(out_dir: Path) -> EventsFile:
    path = out_dir / EVENTS_JSON
    if not path.is_file():
        raise FileNotFoundError(f"no events at {path}; run events first")
    return EventsFile.model_validate_json(path.read_text(encoding="utf-8"))


def stage_events(
    video: Path,
    out_dir: Path,
    settings: AnalysisSettings,
    calibration_path: Path | None,
    scene: SceneConfig | None = None,
) -> EventsFile:
    """Rank near-miss, hard-deceleration and wrong-way entry candidates.

    Conflict metrics need a calibration. Wrong-way entries need a scene with a
    ring zone and work without one.
    """
    info = probe_video(video)
    needs_tracks = calibration_path is not None or scene is not None
    if not needs_tracks:
        logger.warning(
            "no calibration or scene: skipping event analysis "
            "(speeds and roundabout zones are unavailable)"
        )
        _write_events(out_dir, info.id, [], [])
        return load_events(out_dir)
    conn = open_store(_require_db(out_dir))
    try:
        observations = load_observations(conn, PASS_1)
    finally:
        conn.close()
    tracks = build_tracks(observations, settings)
    events: list[EventRecord] = []
    windows: list[tuple[float, float]] = []
    if calibration_path is None:
        logger.warning("no calibration: skipping conflict analysis")
    else:
        homography, _ = _load_homography(calibration_path)
        paths = build_paths(tracks, homography, settings)
        events = detect_events(paths, settings)
        windows = plan_pass2_windows(events, settings, info.duration_s)
    wrong_way = wrong_way_events(tracks, scene, settings) if scene else []
    _write_events(out_dir, info.id, events + wrong_way, windows)
    logger.info(
        "%d conflict events ranked; %d wrong-way entries; %d pass-2 windows planned",
        len(events),
        len(wrong_way),
        len(windows),
    )
    return load_events(out_dir)


def stage_clips(video: Path, out_dir: Path, settings: AnalysisSettings) -> EventsFile:
    """Cut one evidence clip per ranked event."""
    info = probe_video(video)
    bundle = load_events(out_dir)
    clipped = cut_event_clips(
        bundle.events, video, info.duration_s, out_dir / CLIPS_DIR, settings
    )
    _write_events(out_dir, bundle.video_id, clipped, bundle.pass2_windows)
    return load_events(out_dir)


def stage_review(
    video: Path,
    out_dir: Path,
    settings: AnalysisSettings,
    allow_cloud: bool,
) -> int:
    """Review clips with the configured reviewer. Cloud review needs --allow-cloud."""
    bundle = load_events(out_dir)
    budget = ReviewBudget(
        max_calls=settings.review_max_calls, max_requests=settings.review_max_requests
    )
    reviewer: ClipReviewer
    if allow_cloud:
        reviewer = GeminiReviewer.from_env(out_dir / MODEL_LOG, budget)
    else:
        logger.info("cloud review is off; pass --allow-cloud to send clips to Gemini")
        reviewer = NullReviewer()
    clips_dir = out_dir / CLIPS_DIR
    reviewed = 0
    updated: list[EventRecord] = []
    for event in bundle.events:
        if event.clip is None or event.type == "wrong_way_entry":
            updated.append(event)
            continue
        sidecar = ClipSidecar.model_validate_json(
            (clips_dir / f"{event.event_id}.json").read_text(encoding="utf-8")
        )
        clip_duration = sidecar.end_s - sidecar.start_s
        ai_review = review_event(
            reviewer,
            event,
            clips_dir / event.clip,
            video,
            sidecar.start_s,
            clip_duration,
            settings,
            budget,
            clips_dir,
        )
        if ai_review.verdict != "unreviewed":
            reviewed += 1
        updated.append(event.model_copy(update={"ai_review": ai_review}))
    _write_events(out_dir, bundle.video_id, updated, bundle.pass2_windows)
    return reviewed


def stage_report(out_dir: Path, narrate_with_cloud: bool, allow_cloud: bool) -> Path:
    """Write the deterministic report, plus a narrative when cloud use is allowed."""
    stats_path = out_dir / STATS_JSON
    if not stats_path.is_file():
        raise FileNotFoundError(f"no stats at {stats_path}; run analyze first")
    stats = StatsReport.model_validate_json(stats_path.read_text(encoding="utf-8"))
    events = load_events(out_dir).events
    report_path = out_dir / REPORT_MD
    report_path.write_text(render_report(stats, events), encoding="utf-8")
    if narrate_with_cloud:
        if not allow_cloud:
            logger.info(
                "narrative skipped: pass --allow-cloud to send the report to Gemini"
            )
        else:
            from google import genai

            key = os.environ.get("GEMINI_API_KEY", "")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            model = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")
            written = narrate(
                genai.Client(api_key=key), model, stats, events, out_dir / NARRATIVE_MD
            )
            if not written:
                logger.warning(
                    "narrative dropped: it contained numbers not in the report"
                )
    return report_path
