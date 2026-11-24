"""Streamlit front end: choose a video and a scene, press Go, read the results.

Start it with ``traffic-analysis gui --data-dir DIR``. Runs happen in separate
worker processes (see ``jobs.py``), so closing the browser does not stop them.
"""

import argparse
import os
import re
import shutil
import tempfile
import time
import zipfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from numpy.typing import NDArray

from traffic_analysis.analytics import StatsReport
from traffic_analysis.ffmpeg_tools import build_frame_command, probe_video, run_ffmpeg
from traffic_analysis.gemini_count import (
    MAX_DURATION_S,
    MAX_FPS,
    GeminiCountReport,
)
from traffic_analysis.gemini_count import (
    OUTPUT as GEMINI_OUTPUT,
)
from traffic_analysis.jobs import (
    LOG_FILE,
    OUT_DIR,
    JobSpec,
    JobStatus,
    cancel_job,
    create_job,
    list_jobs,
    load_spec,
    read_status,
    start_job,
)
from traffic_analysis.pipeline import EventsFile
from traffic_analysis.schemas import (
    AnalysisSettings,
    SceneConfig,
    load_scene_config,
)

VIDEO_SUFFIXES = (".mp4", ".mov", ".mkv", ".avi", ".m4v")
ACTIVE_STATES = ("queued", "running")
REFRESH_S = 3
ROLE_COLORS = {
    "approach": (60, 200, 60),
    "exit": (230, 60, 60),
    "both": (220, 200, 0),
    "ring": (230, 140, 40),
    "other": (170, 170, 170),
}
GPU_S_PER_FRAME = 0.06
CPU_S_PER_FRAME = 1.2
GUI_DEFAULT_CONF = 0.10
AERIAL_MODEL = "yolo11m-obb.pt"
GEMINI_MODEL = "gemini-frames"
MODEL_OPTIONS = {
    "Standard YOLO11m": "yolo11m.pt",
    "Standard YOLO11l (large)": "yolo11l.pt",
    "Standard YOLO11x (extra large)": "yolo11x.pt",
    "Aerial YOLO11m-OBB (experimental)": AERIAL_MODEL,
    "Gemini frame review (experimental)": GEMINI_MODEL,
}
BROWSER_UPLOAD_WARN_MB = 2000
PAGES = ("New run", "Results")


def safe_name(name: str) -> str:
    """A file name that cannot escape its folder."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).name).strip("._")
    return cleaned or "upload"


def list_files(folder: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in suffixes
    )


def save_upload(data: bytes, name: str, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / safe_name(name)
    target.write_bytes(data)
    return target


def draw_zones(
    frame: NDArray[np.uint8], scene: SceneConfig, max_width: int = 1280
) -> NDArray[np.uint8]:
    """Return the frame (RGB) with every zone outlined, filled and labelled."""
    canvas = frame.copy()
    overlay = canvas.copy()
    for zone in scene.zones:
        color = ROLE_COLORS.get(zone.role, ROLE_COLORS["other"])
        points = np.array([[p.x, p.y] for p in zone.polygon], dtype=np.int32)
        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(canvas, [points], True, color, 6)
    blended = np.asarray(cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0), np.uint8)
    if scene.ring_exclusion:
        points = np.array([[p.x, p.y] for p in scene.ring_exclusion], dtype=np.int32)
        cv2.polylines(blended, [points], True, (0, 200, 255), 6)
    for zone in scene.zones:
        cx = int(np.mean([p.x for p in zone.polygon]))
        cy = int(np.mean([p.y for p in zone.polygon]))
        cv2.putText(
            blended,
            f"{zone.name} ({zone.role})",
            (cx - 120, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            (255, 255, 255),
            4,
        )
    scale = min(1.0, max_width / blended.shape[1])
    if scale < 1.0:
        return np.asarray(cv2.resize(blended, None, fx=scale, fy=scale), np.uint8)
    return blended


@st.cache_data(show_spinner="Reading a frame from the video...")
def zone_preview(video: str, scene_json: str, time_s: float) -> NDArray[np.uint8]:
    scene = SceneConfig.model_validate_json(scene_json)
    with tempfile.TemporaryDirectory() as tmp:
        frame_path = Path(tmp) / "frame.png"
        run_ffmpeg(build_frame_command(Path(video), time_s, frame_path))
        bgr = cv2.imread(str(frame_path))
    if bgr is None:
        raise RuntimeError("could not read the frame")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return draw_zones(np.asarray(rgb, dtype=np.uint8), scene)


def gpu_label() -> str:
    try:
        import torch
    except ImportError:
        return "PyTorch is not installed"
    if torch.cuda.is_available():
        return f"GPU: {torch.cuda.get_device_name(0)}"
    return "No NVIDIA GPU found: runs on the CPU and is about 10x slower"


def has_gpu() -> bool:
    return gpu_label().startswith("GPU")


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} h {minutes:02d} min"
    if minutes:
        return f"{minutes} min {secs:02d} s"
    return f"{secs} s"


def is_valid_scene(path: Path) -> bool:
    try:
        load_scene_config(path)
    except (ValueError, OSError):
        return False
    return True


def scene_files(data_dir: Path) -> list[Path]:
    folders = [data_dir / "scenes", Path("configs")]
    found = [p for f in folders for p in list_files(f, (".json",))]
    return [p for p in found if is_valid_scene(p)]


# ---------- New run page ----------


def pick_video(data_dir: Path) -> Path | None:
    st.subheader("1. Video")
    videos_dir = data_dir / "videos"
    existing = list_files(videos_dir, VIDEO_SUFFIXES)
    uploaded = st.file_uploader(
        "Upload a video", type=[s.lstrip(".") for s in VIDEO_SUFFIXES]
    )
    if uploaded is not None:
        if uploaded.size > BROWSER_UPLOAD_WARN_MB * 1024 * 1024:
            st.warning(
                "Large browser uploads are slow and held in memory. For files over "
                "2 GB copy the file with `deploy.ps1 -Action upload` instead."
            )
        target = save_upload(uploaded.getvalue(), uploaded.name, videos_dir)
        st.success(f"Saved {target.name}")
        existing = list_files(videos_dir, VIDEO_SUFFIXES)
    if not existing:
        st.info(
            f"No videos in {videos_dir}. Upload one above, or copy a file there "
            "with `deploy.ps1 -Action upload`."
        )
        return None
    names = [p.name for p in existing]
    default = names.index(safe_name(uploaded.name)) if uploaded is not None else 0
    chosen = st.selectbox("Video to analyse", names, index=default)
    return videos_dir / chosen


def pick_scene(data_dir: Path) -> Path | None:
    st.subheader("2. Scene (zones)")
    uploaded = st.file_uploader("Upload a scene JSON", type=["json"], key="scene_up")
    if uploaded is not None:
        try:
            SceneConfig.model_validate_json(uploaded.getvalue())
        except ValueError:
            st.error(
                f"{uploaded.name} is not a scene file. A scene has camera_id, "
                "frame_width, frame_height and zones; make it with "
                "`traffic-analysis zones` (for example configs/DJI.json)."
            )
            uploaded = None
        else:
            save_upload(uploaded.getvalue(), uploaded.name, data_dir / "scenes")
    scenes = scene_files(data_dir)
    if not scenes:
        st.info("No scene files yet. Draw zones on your PC and upload the JSON.")
        return None
    labels = {str(p): p.name for p in scenes}
    default = 0
    if uploaded is not None:
        wanted = str(data_dir / "scenes" / safe_name(uploaded.name))
        default = list(labels).index(wanted) if wanted in labels else 0
    key = st.selectbox(
        "Scene", list(labels), index=default, format_func=lambda k: labels[k]
    )
    return Path(key)


def check_scene(scene_path: Path) -> SceneConfig | None:
    try:
        scene = load_scene_config(scene_path)
    except (ValueError, OSError) as exc:
        st.error(f"Scene file is not valid: {exc}")
        return None
    roles = {z.role for z in scene.zones}
    if "ring" not in roles:
        st.warning(
            "This scene has no zone with role `ring`, so roundabout entries, exits "
            "and wrong-way entries will not be counted."
        )
    if not roles & {"approach", "both"} or not roles & {"exit", "both"}:
        st.warning("The scene needs both approach and exit zones.")
    return scene


def preview_section(video: Path, scene_path: Path) -> None:
    with st.expander("Preview the zones on a video frame"):
        time_s = st.number_input("Frame time (s)", min_value=0.0, value=10.0, step=5.0)
        if st.button("Show preview"):
            try:
                st.image(
                    zone_preview(str(video), scene_path.read_text("utf-8"), time_s),
                    caption=f"{scene_path.name} at {time_s:.0f} s",
                )
            except (RuntimeError, OSError, ValueError) as exc:
                st.error(f"Preview failed: {exc}")


def options_section(
    data_dir: Path, duration_s: float
) -> tuple[AnalysisSettings, float, float | None, Path | None]:
    st.subheader("3. Options")
    defaults = AnalysisSettings()
    model_label = st.selectbox("Detection model", list(MODEL_OPTIONS))
    model = MODEL_OPTIONS[model_label]
    st.caption(
        "Larger models can take longer and may not improve counts. "
        "Weights may download on first use."
    )
    if model == AERIAL_MODEL:
        st.warning(
            "Experimental aerial model: verify counts against the video. Its small "
            "and large vehicle detections are both reported as cars, not reliable "
            "vehicle types. People and bicycles are not detected by this model."
        )
    elif model == GEMINI_MODEL:
        st.warning(
            "Gemini receives sampled frames, timestamps and scene boundaries. "
            "API charges may apply. This experimental review is limited to 60 "
            "seconds and 10 frames per second; independent 5-second batches may "
            "miss or duplicate movements at their boundaries."
        )
    col1, col2, col3 = st.columns(3)
    detect_fps = col1.slider(
        "Frames analysed per second",
        3,
        15,
        int(defaults.detect_fps),
        help="Lower is faster but tracks break more easily (5 fps lost most cars "
        "in a test).",
    )
    conf = col2.slider(
        "Detection confidence",
        0.05,
        0.5,
        GUI_DEFAULT_CONF,
        0.05,
        help="Lower values allow weaker detections but can increase false positives. "
        "Tracker thresholds also affect which vehicles are followed.",
    )
    imgsz = col3.selectbox("Image size", [960, 1280, 1600], index=1)
    whole = st.checkbox("Analyse the whole video", value=True)
    start_s, end_s = 0.0, None
    if not whole:
        c1, c2 = st.columns(2)
        start_min = c1.number_input("Start (min)", 0.0, duration_s / 60, 0.0, 1.0)
        end_min = c2.number_input(
            "End (min)", 0.1, duration_s / 60, min(5.0, duration_s / 60), 1.0
        )
        start_s, end_s = start_min * 60, end_min * 60
    calibration = None
    cal_files = list_files(data_dir / "calibrations", (".json",))
    if cal_files:
        names = ["none"] + [p.name for p in cal_files]
        pick = st.selectbox("Calibration (needed for speeds)", names)
        if pick != "none":
            calibration = data_dir / "calibrations" / pick
    settings = AnalysisSettings(
        model=model, detect_fps=float(detect_fps), conf=conf, imgsz=imgsz
    )
    return settings, start_s, end_s, calibration


def estimate_text(settings: AnalysisSettings, span_s: float) -> str:
    frames = int(span_s * settings.detect_fps)
    per_frame = GPU_S_PER_FRAME if has_gpu() else CPU_S_PER_FRAME
    return (
        f"About {frames:,} frames. Rough time: {format_duration(frames * per_frame)} "
        f"({gpu_label()}). Actual time depends on the model and video."
    )


def launch_job(
    data_dir: Path,
    video: Path,
    scene_path: Path,
    calibration: Path | None,
    settings: AnalysisSettings,
    start_s: float,
    end_s: float | None,
    gemini_consent: bool = False,
) -> None:
    """Button callback: create the job, start the worker, switch to Results."""
    spec = JobSpec(
        video=str(video.resolve()),
        scene=str(scene_path.resolve()),
        calibration=str(calibration.resolve()) if calibration else None,
        settings=settings,
        start_s=start_s,
        end_s=end_s,
        label=video.name,
        backend="gemini" if settings.model == GEMINI_MODEL else "yolo",
        gemini_consent=gemini_consent,
    )
    job_dir = create_job(data_dir / "jobs", spec)
    start_job(job_dir)
    st.session_state["job"] = job_dir.name
    st.session_state["page"] = "Results"


def page_new_run(data_dir: Path) -> None:
    video = pick_video(data_dir)
    scene_path = pick_scene(data_dir)
    if video is None or scene_path is None:
        return
    scene = check_scene(scene_path)
    try:
        duration_s = probe_video(video).duration_s
    except (RuntimeError, OSError, ValueError) as exc:
        st.error(f"Cannot read the video: {exc}")
        return
    st.caption(f"{video.name}: {format_duration(duration_s)}")
    if scene is not None:
        preview_section(video, scene_path)
    settings, start_s, end_s, calibration = options_section(data_dir, duration_s)
    span_s = (end_s if end_s is not None else duration_s) - start_s
    gemini = settings.model == GEMINI_MODEL
    consent = False
    if gemini:
        st.info(
            f"Up to approximately {int(span_s * settings.detect_fps):,} sampled "
            "frames will be sent in 5-second batches. The video file is not sent."
        )
        if span_s > MAX_DURATION_S or settings.detect_fps > MAX_FPS:
            st.error("Gemini requires a segment at most 60 seconds and at most 10 fps.")
        if not os.environ.get("GEMINI_API_KEY"):
            st.error("GEMINI_API_KEY is not configured on the server.")
        consent = st.checkbox(
            "I authorize sending the sampled frames and scene boundaries to Gemini "
            "and understand API charges may apply."
        )
    else:
        st.info(estimate_text(settings, span_s))
    if settings.conf > 0.3 and settings.model not in (AERIAL_MODEL, GEMINI_MODEL):
        st.warning(
            "Confidence above 0.3 misses most cars on this overhead view and can "
            "give zero counts. Try 0.10."
        )
    st.button(
        "Go",
        type="primary",
        disabled=scene is None
        or span_s <= 0
        or (
            gemini
            and (
                not consent
                or not os.environ.get("GEMINI_API_KEY")
                or span_s > MAX_DURATION_S
                or settings.detect_fps > MAX_FPS
            )
        ),
        on_click=launch_job,
        args=(
            data_dir,
            video,
            scene_path,
            calibration,
            settings,
            start_s,
            end_s,
            consent,
        ),
    )


# ---------- Results page ----------


def tail(path: Path, lines: int = 25) -> str:
    if not path.is_file():
        return ""
    return "\n".join(path.read_text("utf-8", errors="replace").splitlines()[-lines:])


@st.fragment(run_every=REFRESH_S)
def live_panel(job_dir: Path) -> None:
    status = read_status(job_dir)
    if status.state not in ACTIVE_STATES:
        st.rerun()
    st.progress(status.progress, text=f"{status.state}: {status.stage or 'waiting'}")
    if status.message:
        st.caption(status.message)
    if st.button("Cancel this run"):
        cancel_job(job_dir)
        st.rerun()
    with st.expander("Log"):
        st.code(tail(job_dir / LOG_FILE) or "(empty)")


def metric_row(items: list[tuple[str, int | float]]) -> None:
    for column, (label, value) in zip(st.columns(len(items)), items, strict=True):
        column.metric(label, value)


def show_roundabout(stats: StatsReport) -> None:
    ring = stats.roundabout
    if ring is None:
        st.warning(
            "No roundabout counts: the scene has no `ring` zone. Zone counts are "
            "shown in the report below."
        )
        return
    st.subheader("Roundabout")
    metric_row(
        [
            ("Entered", ring.entered),
            ("Exited", ring.exited),
            ("Wrong-way entries", ring.wrong_way_entries),
            ("Wrong-way exits", ring.wrong_way_exits),
        ]
    )
    if ring.first_seen_on_ring is None:
        st.caption("First-seen-on-ring data is unavailable for this run.")
    else:
        st.metric("Tracks first seen on ring", len(ring.first_seen_on_ring))
        st.caption(
            "Not confirmed entries: tracks may start mid-journey or be duplicates. "
            "Verify the ring and inner boundary in the zone preview."
        )
        if ring.first_seen_on_ring:
            with st.expander("First-seen-on-ring times"):
                for item in ring.first_seen_on_ring:
                    st.write(f"Track {item.track_id}: {item.t_s:.1f} s")
    arms = sorted(set(ring.entered_by_arm) | set(ring.exited_by_arm))
    st.dataframe(
        [
            {
                "arm": arm,
                "entered": ring.entered_by_arm.get(arm, 0),
                "exited": ring.exited_by_arm.get(arm, 0),
                "wrong-way entries": ring.wrong_way_entries_by_arm.get(arm, 0),
            }
            for arm in arms
        ],
        hide_index=True,
        width="stretch",
    )


def show_wrong_way(out_dir: Path, stats: StatsReport) -> None:
    st.subheader("Wrong-way entry examples")
    ring = stats.roundabout
    if ring is None or not ring.wrong_way_examples:
        st.write("None found.")
        return
    events_path = out_dir / "events.json"
    clips: dict[int, str] = {}
    if events_path.is_file():
        bundle = EventsFile.model_validate_json(events_path.read_text("utf-8"))
        clips = {
            e.tracks[0]: e.clip
            for e in bundle.events
            if e.type == "wrong_way_entry" and e.clip
        }
    for example in ring.wrong_way_examples:
        st.markdown(
            f"**Track {example.track_id}**: from `{example.arm}` at "
            f"{format_duration(example.t_s)}"
        )
        clip = clips.get(example.track_id)
        if clip and (out_dir / "clips" / clip).is_file():
            st.video(str(out_dir / "clips" / clip))


def show_other_stats(stats: StatsReport) -> None:
    st.subheader("Vehicles")
    metric_row(
        [
            ("Vehicles tracked", stats.total_vehicles),
            ("Pedestrian tracks", stats.pedestrian_tracks),
        ]
    )
    if stats.by_class:
        st.dataframe(
            [{"class": k, "count": v} for k, v in sorted(stats.by_class.items())],
            hide_index=True,
        )
    if stats.od_matrix:
        st.markdown("**Origin to destination**")
        st.dataframe(
            [c.model_dump() for c in stats.od_matrix], hide_index=True, width="stretch"
        )
    if stats.speed_by_class:
        st.markdown("**Speeds (km/h)**")
        st.dataframe(
            [{"class": k, **v.model_dump()} for k, v in stats.speed_by_class.items()],
            hide_index=True,
        )
    if len(stats.time_bins) > 1:
        st.markdown("**Vehicles over time**")
        st.bar_chart(
            {
                f"{int(b.start_s // 60)} min": sum(b.by_class.values())
                for b in stats.time_bins
            }
        )


def zip_results(out_dir: Path) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(out_dir))
    return buffer.getvalue()


def show_results(job_dir: Path) -> None:
    out_dir = job_dir / OUT_DIR
    spec = load_spec(job_dir)
    if spec.backend == "gemini":
        show_gemini_results(out_dir)
        return
    stats_path = out_dir / "stats.json"
    if not stats_path.is_file():
        st.error("This run produced no results.")
        return
    stats = StatsReport.model_validate_json(stats_path.read_text("utf-8"))
    st.caption(f"Video: {stats.video_id}")
    st.caption(f"Detection model: {spec.settings.model}")
    show_roundabout(stats)
    show_wrong_way(out_dir, stats)
    show_other_stats(stats)
    report_path = out_dir / "report.md"
    if report_path.is_file():
        with st.expander("Full report"):
            st.markdown(report_path.read_text("utf-8"))
    st.subheader("Caveats")
    for caveat in stats.caveats:
        st.write(f"- {caveat}")
    if st.button("Prepare a zip of all result files"):
        st.download_button(
            "Download results.zip",
            zip_results(out_dir),
            file_name=f"{job_dir.name}.zip",
            mime="application/zip",
        )


def show_gemini_results(out_dir: Path) -> None:
    path = out_dir / GEMINI_OUTPUT
    if not path.is_file():
        st.error("This Gemini run produced no report.")
        return
    report = GeminiCountReport.model_validate_json(path.read_text("utf-8"))
    st.subheader("Gemini frame review")
    st.caption(f"Model: {report.model} | Frames sent: {report.frames_sent}")
    st.warning(
        "Unverified candidates, not a confirmed count. Each 5-second batch is "
        "independent: events can be missed or duplicated near batch boundaries. "
        "Compare with the video; do not sum these as a verified total."
    )
    for batch in report.batches:
        st.markdown(f"**{batch.start_s:.1f}–{batch.end_s:.1f} seconds**")
        if not batch.answer.events:
            st.write("No candidate crossings reported.")
        for event in batch.answer.events:
            st.write(
                f"{event.time_s:.1f}s — {event.kind}: {event.arm} — {event.description}"
            )
        if batch.answer.notes:
            st.caption(batch.answer.notes)
    st.download_button(
        "Download Gemini report",
        path.read_bytes(),
        file_name="gemini-count.json",
        mime="application/json",
    )


def describe_job(job_dir: Path) -> str:
    try:
        spec, status = load_spec(job_dir), read_status(job_dir)
    except (OSError, ValueError):
        return job_dir.name
    return f"{job_dir.name}  {spec.label}  [{status.state}]"


def page_results(data_dir: Path) -> None:
    jobs = list_jobs(data_dir / "jobs")
    if not jobs:
        st.info("No runs yet. Start one on the New run page.")
        return
    names = [j.name for j in jobs]
    current = st.session_state.get("job")
    index = names.index(current) if current in names else 0
    by_name = {j.name: j for j in jobs}
    chosen = st.selectbox(
        "Run", names, index=index, format_func=lambda n: describe_job(by_name[n])
    )
    st.session_state["job"] = chosen
    job_dir = by_name[chosen]
    status: JobStatus = read_status(job_dir)
    if status.state in ACTIVE_STATES:
        live_panel(job_dir)
    elif status.state == "done":
        show_results(job_dir)
    else:
        st.error(f"Run {status.state}: {status.message}")
        st.code(tail(job_dir / LOG_FILE, 40) or "(empty)")
    if st.button("Delete this run's files", key=f"del-{chosen}"):
        shutil.rmtree(job_dir, ignore_errors=True)
        st.session_state.pop("job", None)
        st.rerun()


# ---------- App ----------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/gui"))
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    data_dir = parse_args().data_dir
    st.set_page_config(page_title="Roundabout traffic analysis", layout="wide")
    st.title("Roundabout traffic analysis")
    st.sidebar.write(gpu_label())
    st.sidebar.caption(f"Data folder: {data_dir.resolve()}")
    page = st.sidebar.radio("Page", PAGES, key="page")
    if page == "New run":
        page_new_run(data_dir)
    else:
        page_results(data_dir)
    st.sidebar.caption(time.strftime("Loaded %H:%M:%S"))


if __name__ == "__main__":
    main()
