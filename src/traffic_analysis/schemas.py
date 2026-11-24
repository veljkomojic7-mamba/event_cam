"""Pydantic models for scene configuration, video metadata and analysis outputs."""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["approach", "exit", "both", "ring", "other"]
VehicleClass = Literal["car", "truck", "bus", "motorcycle", "bicycle", "person"]


class StrictModel(BaseModel):
    """Base model that rejects unknown fields so typos in JSON fail loudly."""

    model_config = ConfigDict(extra="forbid")


class Point2D(StrictModel):
    """A point in frame pixel coordinates."""

    x: float
    y: float


class Zone(StrictModel):
    """A named polygon region (approach, exit, or other) in frame pixels."""

    name: str = Field(min_length=1)
    role: Role = "other"
    polygon: list[Point2D] = Field(min_length=3)


class CountingLine(StrictModel):
    """A directed counting line in frame pixels."""

    name: str = Field(min_length=1)
    start: Point2D
    end: Point2D


class SceneConfig(StrictModel):
    """Confirmed per-camera scene configuration, reused on later runs."""

    camera_id: str = Field(min_length=1)
    frame_width: int = Field(gt=0)
    frame_height: int = Field(gt=0)
    zones: list[Zone] = Field(default_factory=list)
    ring_exclusion: list[Point2D] | None = Field(default=None, min_length=3)
    lines: list[CountingLine] = Field(default_factory=list)
    calibration_id: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        zone_names = [zone.name for zone in self.zones]
        if len(set(zone_names)) != len(zone_names):
            raise ValueError("zone names must be unique")
        line_names = [line.name for line in self.lines]
        if len(set(line_names)) != len(line_names):
            raise ValueError("line names must be unique")
        for zone in self.zones:
            for point in zone.polygon:
                self._check_point_in_frame(point, zone.name)
        for point in self.ring_exclusion or []:
            self._check_point_in_frame(point, "ring_exclusion")
        for line in self.lines:
            self._check_point_in_frame(line.start, line.name)
            self._check_point_in_frame(line.end, line.name)
        return self

    def _check_point_in_frame(self, point: Point2D, owner: str) -> None:
        if not (0 <= point.x <= self.frame_width and 0 <= point.y <= self.frame_height):
            raise ValueError(f"point of '{owner}' lies outside the frame")


class VideoInfo(StrictModel):
    """Metadata for the source video, read with ffprobe."""

    id: str = Field(min_length=1)
    fps: float = Field(gt=0)
    duration_s: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    calibration_id: str | None = None


class ZoneVisit(StrictModel):
    """A track entering or leaving a zone at a video time."""

    zone: str = Field(min_length=1)
    t_s: float = Field(ge=0)


class RoundaboutPassage(StrictModel):
    """How one vehicle crossed the roundabout, independent of the lane it used.

    An entry is the move from an arm zone into the ring zone, an exit the move
    from the ring back to an arm zone. An entry from an ``exit`` arm, or an exit
    into an ``approach`` arm, is flagged as wrong-way.
    """

    entry_arm: str | None = None
    entry_t_s: float | None = Field(default=None, ge=0)
    entry_wrong_way: bool = False
    exit_arm: str | None = None
    exit_t_s: float | None = Field(default=None, ge=0)
    exit_wrong_way: bool = False


class TrajectoryPoint(StrictModel):
    """A ground-plane position at a video time."""

    t_s: float = Field(ge=0)
    x_m: float
    y_m: float


class PixelTrajectoryPoint(StrictModel):
    """A bottom-centre ground point in frame pixels at a video time."""

    t_s: float = Field(ge=0)
    x_px: float
    y_px: float


class SpeedEstimate(StrictModel):
    """Speed summary for one vehicle in km/h, from the calibrated homography."""

    mean: float = Field(ge=0)
    median: float = Field(ge=0)
    max: float = Field(ge=0)
    method: Literal["homography"] = "homography"


class TrackQuality(StrictModel):
    """Tracker quality indicators shown next to every result."""

    id_switches: int = Field(ge=0)
    occluded_frac: float = Field(ge=0, le=1)


class VehicleRecord(StrictModel):
    """Per-vehicle output record. The JSON key for the class is ``class``."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    track_id: int = Field(ge=0)
    vehicle_class: VehicleClass = Field(alias="class")
    class_conf: float = Field(ge=0, le=1)
    entry: ZoneVisit | None = None
    exit: ZoneVisit | None = None
    roundabout: RoundaboutPassage | None = None
    trajectory: list[TrajectoryPoint] = Field(default_factory=list)
    trajectory_px: list[PixelTrajectoryPoint] = Field(default_factory=list)
    speed_kmh: SpeedEstimate | None = None
    quality: TrackQuality


class EventMetrics(StrictModel):
    """Conflict metrics for a candidate. Unknown values are None."""

    min_sep_m: float | None = Field(default=None, ge=0)
    ttc_min_s: float | None = Field(default=None, ge=0)
    pet_s: float | None = Field(default=None, ge=0)
    drac: float | None = Field(default=None, ge=0)
    decel_mps2: float | None = Field(default=None, ge=0)
    score: float = Field(default=0.0, ge=0)


class AIReview(StrictModel):
    """Verdict from the multimodal model on a flagged clip."""

    verdict: Literal["confirmed", "rejected", "needs_video", "unreviewed"] = (
        "unreviewed"
    )
    confidence: float | None = Field(default=None, ge=0, le=1)
    notes: str = ""


class EventRecord(StrictModel):
    """A ranked near-miss candidate with its evidence clip."""

    event_id: str = Field(min_length=1)
    type: Literal["near_miss_candidate", "hard_deceleration", "wrong_way_entry"]
    pass_no: int = Field(default=1, ge=1, le=2)
    t_start_s: float = Field(ge=0)
    t_end_s: float = Field(ge=0)
    tracks: list[int] = Field(min_length=1)
    metrics: EventMetrics = Field(default_factory=EventMetrics)
    clip: str | None = None
    ai_review: AIReview = Field(default_factory=AIReview)

    @model_validator(mode="after")
    def _check_times(self) -> Self:
        if self.t_end_s < self.t_start_s:
            raise ValueError("t_end_s must not be before t_start_s")
        return self


class AnalysisSettings(StrictModel):
    """Tunable thresholds for detection, counting, speed and conflicts."""

    model: str = "yolo11m.pt"
    tracker: str = "bytetrack.yaml"
    detect_fps: float = Field(default=10.0, gt=0)
    imgsz: int = Field(default=1280, gt=0)
    conf: float = Field(default=0.3, ge=0, le=1)
    static_diff_threshold: float = Field(default=0.0, ge=0)
    device: str | None = None
    half: bool = False
    min_track_s: float = Field(default=1.0, gt=0)
    min_observations: int = Field(default=3, ge=1)
    zone_min_obs: int = Field(default=2, ge=1)
    stitch_gap_s: float = Field(default=2.0, ge=0)
    stitch_max_px: float = Field(default=150.0, gt=0)
    time_bin_s: float = Field(default=300.0, gt=0)
    speed_window_s: float = Field(default=1.0, gt=0)
    car_length_m: float = Field(default=4.5, gt=0)
    conflict_min_speed_mps: float = Field(default=1.0, ge=0)
    sep_m: float = Field(default=2.0, gt=0)
    ttc_s: float = Field(default=1.5, gt=0)
    pet_s: float = Field(default=1.5, gt=0)
    drac_mps2: float = Field(default=3.35, gt=0)
    decel_mps2: float = Field(default=4.5, gt=0)
    max_events: int = Field(default=50, ge=1)
    pass2_pad_s: float = Field(default=1.0, ge=0)
    pass2_max_windows: int = Field(default=20, ge=1)
    clip_pad_s: float = Field(default=5.0, ge=0)
    review_max_calls: int = Field(default=30, ge=0)
    review_max_rounds: int = Field(default=2, ge=1, le=2)
    review_request_max_s: float = Field(default=30.0, gt=0)
    review_max_requests: int = Field(default=20, ge=0)


class CalibrationPoint(StrictModel):
    """One ground reference: pixel location and its real position in metres."""

    image: Point2D
    world: Point2D


class CalibrationFile(StrictModel):
    """Homography from image pixels to the road plane (metres), per camera."""

    calibration_id: str = Field(min_length=1)
    camera_id: str = Field(min_length=1)
    points: list[CalibrationPoint] = Field(min_length=4)
    homography: list[list[float]] = Field(min_length=3, max_length=3)
    rms_error_m: float = Field(ge=0)
    max_error_m: float = Field(ge=0)


class ReviewRequest(StrictModel):
    """A time range the model asks to see at full quality."""

    start_s: float = Field(ge=0)
    end_s: float = Field(ge=0)
    reason: str = ""


class ReviewVerdict(StrictModel):
    """Structured reply from the clip reviewer, validated before use."""

    event_id: str = ""
    verdict: Literal["confirmed", "rejected", "needs_video"]
    confidence: float = Field(ge=0, le=1)
    notes: str = ""
    requested_ranges: list[ReviewRequest] = Field(default_factory=list)


def load_scene_config(path: Path) -> SceneConfig:
    """Load and validate a scene configuration file."""
    if not path.is_file():
        raise FileNotFoundError(f"scene config not found: {path}")
    return SceneConfig.model_validate_json(path.read_text(encoding="utf-8"))


def save_scene_config(config: SceneConfig, path: Path) -> None:
    """Write a scene configuration as indented JSON, creating parent folders."""
    save_model(config, path)


def save_model(model: BaseModel, path: Path) -> None:
    """Write any pydantic model as indented JSON, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        model.model_dump_json(indent=2, by_alias=True) + "\n", encoding="utf-8"
    )


def load_settings(path: Path | None) -> AnalysisSettings:
    """Load analysis settings, or return the defaults when no file is given."""
    if path is None:
        return AnalysisSettings()
    if not path.is_file():
        raise FileNotFoundError(f"settings file not found: {path}")
    return AnalysisSettings.model_validate_json(path.read_text(encoding="utf-8"))


def load_calibration(path: Path) -> CalibrationFile:
    """Load and validate a calibration file."""
    if not path.is_file():
        raise FileNotFoundError(f"calibration file not found: {path}")
    return CalibrationFile.model_validate_json(path.read_text(encoding="utf-8"))
