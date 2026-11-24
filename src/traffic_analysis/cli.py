"""Command-line entry point for the traffic analysis pipeline."""

import argparse
import importlib.util
import logging
import subprocess
import sys
from pathlib import Path

from pydantic import ValidationError

from traffic_analysis.ffmpeg_tools import (
    SampleRequest,
    build_frame_command,
    cut_sample,
    probe_video,
    run_ffmpeg,
)
from traffic_analysis.overlay import run_editor
from traffic_analysis.pipeline import (
    stage_analyze,
    stage_calibrate,
    stage_clips,
    stage_detect,
    stage_events,
    stage_report,
    stage_review,
)
from traffic_analysis.schemas import load_scene_config, load_settings

LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with one sub-command per stage."""
    parser = argparse.ArgumentParser(
        prog="traffic-analysis", description="Traffic video analysis tools."
    )
    parser.add_argument("--log-level", default="INFO", choices=LOG_LEVELS)
    commands = parser.add_subparsers(dest="command", required=True)

    probe = commands.add_parser("probe", help="Read fps, duration and resolution")
    probe.add_argument("--video", type=Path, required=True)
    probe.add_argument("--output", type=Path, required=True, help="video.json path")

    sample = commands.add_parser("sample", help="Cut a re-encoded sample clip")
    sample.add_argument("--video", type=Path, required=True)
    sample.add_argument("--start", type=float, default=0.0, help="seconds")
    sample.add_argument("--duration", type=float, default=240.0, help="seconds")
    sample.add_argument("--output", type=Path, required=True)

    frame = commands.add_parser("frame", help="Extract one frame as an image")
    frame.add_argument("--video", type=Path, required=True)
    frame.add_argument("--time", type=float, default=0.0, help="seconds")
    frame.add_argument("--output", type=Path, required=True)

    zones = commands.add_parser("zones", help="Draw or edit zones on a frame")
    zones.add_argument("--frame", type=Path, required=True)
    zones.add_argument("--scene", type=Path, required=True, help="scene JSON path")

    def add_common(sub: argparse.ArgumentParser, video: bool = True) -> None:
        if video:
            sub.add_argument("--video", type=Path, required=True)
        sub.add_argument("--out-dir", type=Path, required=True)
        sub.add_argument("--settings", type=Path, default=None, help="settings JSON")

    detect = commands.add_parser("detect", help="Pass 1: track vehicles over the video")
    add_common(detect)
    detect.add_argument("--start", type=float, default=0.0, help="seconds")
    detect.add_argument("--end", type=float, default=None, help="seconds")

    calibrate = commands.add_parser("calibrate", help="Fit a pixel-to-metre homography")
    calibrate.add_argument("--camera-id", required=True)
    calibrate.add_argument(
        "--points", type=Path, required=True, help="JSON point pairs"
    )
    calibrate.add_argument("--out", type=Path, required=True, help="calibration JSON")

    analyze = commands.add_parser("analyze", help="Counts, O-D, time bins, speeds")
    add_common(analyze)
    analyze.add_argument("--scene", type=Path, required=True)
    analyze.add_argument("--calibration", type=Path, default=None)

    events = commands.add_parser(
        "events", help="Rank near-miss, decel and wrong-way entry candidates"
    )
    add_common(events)
    events.add_argument("--calibration", type=Path, default=None)
    events.add_argument(
        "--scene", type=Path, default=None, help="scene JSON (wrong-way entries)"
    )

    clips = commands.add_parser("clips", help="Cut evidence clips for ranked events")
    add_common(clips)

    review = commands.add_parser(
        "review", help="Review clips (cloud needs --allow-cloud)"
    )
    add_common(review)
    review.add_argument("--allow-cloud", action="store_true")

    report = commands.add_parser("report", help="Write report.md from stats")
    add_common(report, video=False)
    report.add_argument("--narrate", action="store_true", help="add an LLM summary")
    report.add_argument("--allow-cloud", action="store_true")

    run = commands.add_parser("run", help="Run every stage in order")
    add_common(run)
    run.add_argument("--scene", type=Path, required=True)
    run.add_argument("--calibration", type=Path, default=None)
    run.add_argument("--allow-cloud", action="store_true")
    run.add_argument("--narrate", action="store_true")

    gui = commands.add_parser("gui", help="Start the browser GUI (needs the gui extra)")
    gui.add_argument("--data-dir", type=Path, default=Path("data/gui"))
    gui.add_argument("--host", default="127.0.0.1")
    gui.add_argument("--port", type=int, default=8501)
    gui.add_argument("--max-upload-mb", type=int, default=4096)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested sub-command. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.command == "probe":
            info = probe_video(args.video)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                info.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            logger.info("%s: %s", args.video.name, info.model_dump_json())
        elif args.command == "sample":
            cut_sample(
                SampleRequest(
                    source=args.video,
                    start_s=args.start,
                    duration_s=args.duration,
                    output=args.output,
                )
            )
        elif args.command == "frame":
            args.output.parent.mkdir(parents=True, exist_ok=True)
            run_ffmpeg(build_frame_command(args.video, args.time, args.output))
        elif args.command == "zones":
            run_editor(args.frame, args.scene)
        elif args.command == "detect":
            settings = load_settings(args.settings)
            summary = stage_detect(
                args.video, args.out_dir, settings, args.start, args.end
            )
            logger.info("detect: %s", summary)
        elif args.command == "calibrate":
            stage_calibrate(args.camera_id, args.points, args.out)
        elif args.command == "analyze":
            settings = load_settings(args.settings)
            scene = load_scene_config(args.scene)
            stage_analyze(args.video, scene, args.out_dir, settings, args.calibration)
        elif args.command == "events":
            settings = load_settings(args.settings)
            event_scene = load_scene_config(args.scene) if args.scene else None
            stage_events(
                args.video, args.out_dir, settings, args.calibration, event_scene
            )
        elif args.command == "clips":
            settings = load_settings(args.settings)
            stage_clips(args.video, args.out_dir, settings)
        elif args.command == "review":
            settings = load_settings(args.settings)
            count = stage_review(args.video, args.out_dir, settings, args.allow_cloud)
            logger.info("review: %d events with a verdict", count)
        elif args.command == "report":
            path = stage_report(args.out_dir, args.narrate, args.allow_cloud)
            logger.info("report written to %s", path)
        elif args.command == "run":
            _run_all(args)
        elif args.command == "gui":
            return _start_gui(args)
    except (
        FileNotFoundError,
        RuntimeError,
        ValueError,
        ValidationError,
        ImportError,
    ) as exc:
        logger.error("%s", exc)
        return 1
    return 0


def _start_gui(args: argparse.Namespace) -> int:
    """Run the Streamlit app in this terminal until it is stopped."""
    if importlib.util.find_spec("streamlit") is None:
        logger.error("streamlit is missing; run: python -m uv sync --extra gui")
        return 1
    script = Path(__file__).with_name("gui.py")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
        "--server.maxUploadSize",
        str(args.max_upload_mb),
        "--browser.gatherUsageStats",
        "false",
        "--",
        "--data-dir",
        str(args.data_dir),
    ]
    return subprocess.call(command)


def _run_all(args: argparse.Namespace) -> None:
    """Chain every stage. Cloud steps run only with --allow-cloud."""
    settings = load_settings(args.settings)
    scene = load_scene_config(args.scene)
    stage_detect(args.video, args.out_dir, settings)
    stage_analyze(args.video, scene, args.out_dir, settings, args.calibration)
    stage_events(args.video, args.out_dir, settings, args.calibration, scene)
    stage_clips(args.video, args.out_dir, settings)
    stage_review(args.video, args.out_dir, settings, args.allow_cloud)
    stage_report(args.out_dir, args.narrate, args.allow_cloud)
