"""Counting analytics: stitch fragments, zone entry/exit, lines, counts, O-D."""

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from pydantic import Field

from traffic_analysis.calibration import estimate_speed
from traffic_analysis.geometry import (
    FloatArray,
    apply_homography,
    point_in_polygon,
    segment_crossing_sign,
    zones_containing,
)
from traffic_analysis.schemas import (
    AnalysisSettings,
    CountingLine,
    EventRecord,
    PixelTrajectoryPoint,
    RoundaboutPassage,
    SceneConfig,
    SpeedEstimate,
    StrictModel,
    TrackQuality,
    TrajectoryPoint,
    VehicleRecord,
    ZoneVisit,
)
from traffic_analysis.store import Observation

VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle", "bicycle"})


@dataclass
class Track:
    """One vehicle or pedestrian, possibly stitched from several fragments."""

    track_id: int
    class_name: str
    class_conf: float
    observations: list[Observation]
    merges: int = 0

    @property
    def t_first(self) -> float:
        return self.observations[0].t_s

    @property
    def t_last(self) -> float:
        return self.observations[-1].t_s


def _vote_class(observations: list[Observation]) -> tuple[str, float]:
    """Pick the class with the largest summed confidence; return it and its mean."""
    weight: dict[str, float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)
    for obs in observations:
        weight[obs.class_name] += obs.conf
        count[obs.class_name] += 1
    best = max(weight, key=lambda name: weight[name])
    return best, weight[best] / count[best]


@dataclass
class _Chain:
    track_id: int
    class_name: str
    observations: list[Observation]
    merges: int = 0
    raw_ids: list[int] = field(default_factory=list)


def build_tracks(
    observations: list[Observation], settings: AnalysisSettings
) -> list[Track]:
    """Group observations by tracker ID, stitch short gaps, drop short tracks."""
    by_raw: dict[int, list[Observation]] = defaultdict(list)
    for ob in observations:
        by_raw[ob.track_id].append(ob)

    fragments = sorted(by_raw.items(), key=lambda item: item[1][0].t_s)
    chains: list[_Chain] = []
    for raw_id, obs in fragments:
        obs.sort(key=lambda o: o.t_s)
        frag_class, _ = _vote_class(obs)
        best: _Chain | None = None
        best_gap = math.inf
        for chain in chains:
            last = chain.observations[-1]
            gap = obs[0].t_s - last.t_s
            if gap < 0 or gap > settings.stitch_gap_s or gap >= best_gap:
                continue
            if chain.class_name != frag_class:
                continue
            dist = math.hypot(
                obs[0].ground_x_px - last.ground_x_px,
                obs[0].ground_y_px - last.ground_y_px,
            )
            if dist <= settings.stitch_max_px:
                best, best_gap = chain, gap
        if best is None:
            chains.append(
                _Chain(
                    track_id=raw_id,
                    class_name=frag_class,
                    observations=list(obs),
                    raw_ids=[raw_id],
                )
            )
        else:
            best.observations.extend(obs)
            best.observations.sort(key=lambda o: o.t_s)
            best.merges += 1
            best.raw_ids.append(raw_id)
            best.class_name, _ = _vote_class(best.observations)

    tracks: list[Track] = []
    for chain in chains:
        duration = chain.observations[-1].t_s - chain.observations[0].t_s
        if duration < settings.min_track_s:
            continue
        if len(chain.observations) < settings.min_observations:
            continue
        class_name, class_conf = _vote_class(chain.observations)
        tracks.append(
            Track(
                track_id=chain.track_id,
                class_name=class_name,
                class_conf=class_conf,
                observations=chain.observations,
                merges=chain.merges,
            )
        )
    return tracks


def zone_visits(
    track: Track, scene: SceneConfig
) -> tuple[ZoneVisit | None, ZoneVisit | None]:
    """Return (entry, exit): first and last counted zones the track touches."""
    zones = [z for z in scene.zones if z.role not in ("other", "ring")]
    first: dict[str, float] = {}
    last: dict[str, float] = {}
    for obs in track.observations:
        for name in zones_containing(obs.ground_x_px, obs.ground_y_px, zones):
            first.setdefault(name, obs.t_s)
            last[name] = obs.t_s
    if not first:
        return None, None
    entry_zone = min(first, key=lambda name: first[name])
    exit_zone = max(last, key=lambda name: last[name])
    return (
        ZoneVisit(zone=entry_zone, t_s=first[entry_zone]),
        ZoneVisit(zone=exit_zone, t_s=last[exit_zone]),
    )


@dataclass(frozen=True)
class LineCrossing:
    """A track crossing a counting line once."""

    line: str
    direction: str
    class_name: str
    t_s: float


def line_crossings(
    track: Track, lines: list[CountingLine], max_gap_s: float
) -> list[LineCrossing]:
    """Return the first crossing of each counting line by a track."""
    seen: set[str] = set()
    crossings: list[LineCrossing] = []
    for a, b in zip(track.observations, track.observations[1:], strict=False):
        if b.t_s - a.t_s > max_gap_s:
            continue
        for line in lines:
            if line.name in seen:
                continue
            sign = segment_crossing_sign(
                (a.ground_x_px, a.ground_y_px),
                (b.ground_x_px, b.ground_y_px),
                line,
            )
            if sign:
                direction = "forward" if sign > 0 else "reverse"
                crossings.append(
                    LineCrossing(line.name, direction, track.class_name, b.t_s)
                )
                seen.add(line.name)
    return crossings


RING = "ring"


@dataclass(frozen=True)
class _Run:
    """A stretch of consecutive observations in one ring or arm zone."""

    zone: str
    t_first: float
    t_last: float


def _zone_runs(track: Track, scene: SceneConfig, min_obs: int) -> list[_Run]:
    """Collapse a track into the ordered zones it dwelt in, ring or arm.

    Arm zones win over the ring where they overlap. Frames outside every zone are
    skipped, and runs shorter than ``min_obs`` are dropped as boundary flicker,
    except the final run, which is kept because a track ends where the vehicle
    leaves view.
    """
    arms = [z for z in scene.zones if z.role in ("approach", "exit", "both")]
    rings = [z for z in scene.zones if z.role == RING]
    runs: list[_Run] = []
    current: str | None = None
    count = 0
    t_first = 0.0
    t_last = 0.0

    def close(needed: int = min_obs) -> None:
        if current is not None and count >= needed:
            if runs and runs[-1].zone == current:
                runs[-1] = _Run(current, runs[-1].t_first, t_last)
            else:
                runs.append(_Run(current, t_first, t_last))

    for obs in track.observations:
        inside = zones_containing(obs.ground_x_px, obs.ground_y_px, arms)
        if inside:
            state: str | None = inside[0]
        elif zones_containing(obs.ground_x_px, obs.ground_y_px, rings):
            state = RING
        else:
            continue
        if state != current:
            close()
            current, count, t_first = state, 0, obs.t_s
        count += 1
        t_last = obs.t_s
    # The last run may be short because the track ends as the vehicle leaves view.
    close(needed=1)
    return runs


def roundabout_passage(
    track: Track, scene: SceneConfig, min_obs: int
) -> RoundaboutPassage | None:
    """Entry and exit of the roundabout for one track, or None without a ring zone.

    Only the first ring visit can be an entry and only the last can be an exit,
    so boundary flicker cannot create extra or reversed movements.
    """
    if not any(z.role == RING for z in scene.zones):
        return None
    role = {z.name: z.role for z in scene.zones}
    runs = _zone_runs(track, scene, min_obs)
    ring_idx = [i for i, run in enumerate(runs) if run.zone == RING]
    passage = RoundaboutPassage()
    if not ring_idx:
        return passage
    first, last = ring_idx[0], ring_idx[-1]
    if first > 0:
        arm = runs[first - 1].zone
        passage = passage.model_copy(
            update={
                "entry_arm": arm,
                "entry_t_s": runs[first].t_first,
                "entry_wrong_way": role[arm] == "exit",
            }
        )
    if last < len(runs) - 1:
        arm = runs[last + 1].zone
        passage = passage.model_copy(
            update={
                "exit_arm": arm,
                "exit_t_s": runs[last + 1].t_first,
                "exit_wrong_way": role[arm] == "approach",
            }
        )
    return passage


def wrong_way_events(
    tracks: list[Track], scene: SceneConfig, settings: AnalysisSettings
) -> list[EventRecord]:
    """One event per vehicle that entered the ring from an exit lane, in time order."""
    found: list[tuple[float, int]] = []
    for track in tracks:
        if track.class_name not in VEHICLE_CLASSES:
            continue
        passage = roundabout_passage(track, scene, settings.zone_min_obs)
        if passage is None or not passage.entry_wrong_way:
            continue
        assert passage.entry_t_s is not None
        found.append((passage.entry_t_s, track.track_id))
    found.sort()
    return [
        EventRecord(
            event_id=f"ww-{index:03d}",
            type="wrong_way_entry",
            t_start_s=max(0.0, entry - 1.0),
            t_end_s=entry + 1.0,
            tracks=[track_id],
        )
        for index, (entry, track_id) in enumerate(found, start=1)
    ]


def _vehicle_quality(track: Track, rate_fps: float) -> TrackQuality:
    span = track.t_last - track.t_first
    expected = span * rate_fps + 1.0
    occluded = 1.0 - min(1.0, len(track.observations) / expected)
    return TrackQuality(
        id_switches=track.merges, occluded_frac=round(max(0.0, occluded), 4)
    )


def build_vehicle_record(
    track: Track,
    scene: SceneConfig,
    settings: AnalysisSettings,
    rate_fps: float,
    homography: FloatArray | None,
) -> VehicleRecord:
    """Build the per-vehicle output record with entry, exit and trajectory."""
    entry, exit_ = zone_visits(track, scene)
    passage = roundabout_passage(track, scene, settings.zone_min_obs)
    pixel_path = [
        PixelTrajectoryPoint(t_s=o.t_s, x_px=o.ground_x_px, y_px=o.ground_y_px)
        for o in track.observations
    ]
    metres_path: list[TrajectoryPoint] = []
    speed: SpeedEstimate | None = None
    if homography is not None:
        pts = np.array(
            [[o.ground_x_px, o.ground_y_px] for o in track.observations],
            dtype=np.float64,
        )
        metres = apply_homography(pts, homography)
        times = np.array([o.t_s for o in track.observations], dtype=np.float64)
        metres_path = [
            TrajectoryPoint(t_s=float(t), x_m=float(x), y_m=float(y))
            for t, (x, y) in zip(times, metres, strict=True)
        ]
        speed = estimate_speed(metres, times, settings.speed_window_s)
    return VehicleRecord.model_validate(
        {
            "track_id": track.track_id,
            "class": track.class_name,
            "class_conf": round(track.class_conf, 4),
            "entry": entry.model_dump() if entry else None,
            "exit": exit_.model_dump() if exit_ else None,
            "roundabout": passage.model_dump() if passage else None,
            "trajectory": [p.model_dump() for p in metres_path],
            "trajectory_px": [p.model_dump() for p in pixel_path],
            "speed_kmh": speed.model_dump() if speed else None,
            "quality": _vehicle_quality(track, rate_fps).model_dump(),
        }
    )


class ODCell(StrictModel):
    """Count of vehicles that entered through one zone and left through another."""

    entry: str
    exit: str
    count: int = Field(ge=0)


class TimeBin(StrictModel):
    """Vehicle entries in one time bin, per class."""

    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    by_class: dict[str, int]


class SpeedSummary(StrictModel):
    """Speed distribution for one class, in km/h."""

    n: int = Field(ge=0)
    median_kmh: float = Field(ge=0)
    p85_kmh: float = Field(ge=0)


class WrongWayExample(StrictModel):
    """One vehicle that entered the roundabout from an exit lane."""

    track_id: int = Field(ge=0)
    arm: str
    t_s: float = Field(ge=0)


class RingFirstSeen(StrictModel):
    """A retained track whose first observation was on the ring roadway."""

    track_id: int = Field(ge=0)
    t_s: float = Field(ge=0)


class RoundaboutStats(StrictModel):
    """Roundabout totals regardless of lane, plus per-arm breakdowns."""

    entered: int = Field(ge=0)
    exited: int = Field(ge=0)
    wrong_way_entries: int = Field(ge=0)
    wrong_way_exits: int = Field(ge=0)
    entered_by_arm: dict[str, int]
    exited_by_arm: dict[str, int]
    wrong_way_entries_by_arm: dict[str, int]
    wrong_way_examples: list[WrongWayExample]
    first_seen_on_ring: list[RingFirstSeen] | None = None


class StatsReport(StrictModel):
    """Aggregates computed in code. The LLM only reads this; it never edits it."""

    video_id: str
    calibration_id: str | None
    total_vehicles: int = Field(ge=0)
    by_class: dict[str, int]
    entries_by_zone: dict[str, dict[str, int]]
    exits_by_zone: dict[str, dict[str, int]]
    line_crossings: dict[str, dict[str, dict[str, int]]]
    od_matrix: list[ODCell]
    time_bins: list[TimeBin]
    speed_by_class: dict[str, SpeedSummary]
    mean_id_switches: float
    mean_occluded_frac: float
    pedestrian_tracks: int = Field(ge=0)
    roundabout: RoundaboutStats | None = None
    caveats: list[str]


class RoundaboutTally:
    """Accumulates roundabout entries and exits while stats are built."""

    def __init__(self, scene: SceneConfig) -> None:
        self.seen = any(zone.role == RING for zone in scene.zones)
        self.entered: dict[str, int] = defaultdict(int)
        self.exited: dict[str, int] = defaultdict(int)
        self.wrong_way_in: dict[str, int] = defaultdict(int)
        self.wrong_way_out = 0
        self.examples: list[WrongWayExample] = []
        self.first_seen_on_ring: list[RingFirstSeen] | None = (
            [] if self.seen and scene.ring_exclusion is not None else None
        )

    def add_first_seen(self, track: Track, scene: SceneConfig) -> None:
        if self.first_seen_on_ring is None:
            return
        first = track.observations[0]
        x, y = first.ground_x_px, first.ground_y_px
        arms = [z for z in scene.zones if z.role in ("approach", "exit", "both")]
        rings = [z for z in scene.zones if z.role == RING]
        if (
            not zones_containing(x, y, arms)
            and zones_containing(x, y, rings)
            and not point_in_polygon(x, y, scene.ring_exclusion or [])
        ):
            self.first_seen_on_ring.append(
                RingFirstSeen(track_id=track.track_id, t_s=first.t_s)
            )

    def add(self, track_id: int, passage: RoundaboutPassage) -> None:
        self.seen = True
        if passage.entry_arm is not None and passage.entry_t_s is not None:
            self.entered[passage.entry_arm] += 1
            if passage.entry_wrong_way:
                self.wrong_way_in[passage.entry_arm] += 1
                self.examples.append(
                    WrongWayExample(
                        track_id=track_id,
                        arm=passage.entry_arm,
                        t_s=passage.entry_t_s,
                    )
                )
        if passage.exit_arm is not None:
            self.exited[passage.exit_arm] += 1
            self.wrong_way_out += int(passage.exit_wrong_way)

    def stats(self) -> RoundaboutStats | None:
        if not self.seen:
            return None
        return RoundaboutStats(
            entered=sum(self.entered.values()),
            exited=sum(self.exited.values()),
            wrong_way_entries=sum(self.wrong_way_in.values()),
            wrong_way_exits=self.wrong_way_out,
            entered_by_arm=dict(self.entered),
            exited_by_arm=dict(self.exited),
            wrong_way_entries_by_arm=dict(self.wrong_way_in),
            wrong_way_examples=sorted(self.examples, key=lambda e: e.t_s),
            first_seen_on_ring=(
                sorted(self.first_seen_on_ring, key=lambda e: e.t_s)
                if self.first_seen_on_ring is not None
                else None
            ),
        )


def _median_p85(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    return float(np.median(arr)), float(np.percentile(arr, 85))


def build_stats(
    video_id: str,
    scene: SceneConfig,
    settings: AnalysisSettings,
    tracks: list[Track],
    records: list[VehicleRecord],
    calibration_id: str | None,
) -> StatsReport:
    """Aggregate per-class counts, gates, O-D, time bins, speeds and quality."""
    vehicle_tracks = [t for t in tracks if t.class_name in VEHICLE_CLASSES]
    record_by_id = {r.track_id: r for r in records}
    by_class: dict[str, int] = defaultdict(int)
    entries: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    exits: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    lines: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    od: dict[tuple[str, str], int] = defaultdict(int)
    bins: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    speeds: dict[str, list[float]] = defaultdict(list)
    ring = RoundaboutTally(scene)

    for track in vehicle_tracks:
        record = record_by_id[track.track_id]
        by_class[track.class_name] += 1
        if record.roundabout is not None:
            ring.add(track.track_id, record.roundabout)
            ring.add_first_seen(track, scene)
        if record.entry is not None:
            entries[record.entry.zone][track.class_name] += 1
            bin_index = int(record.entry.t_s // settings.time_bin_s)
        else:
            bin_index = int(track.t_first // settings.time_bin_s)
        bins[bin_index][track.class_name] += 1
        if record.exit is not None:
            exits[record.exit.zone][track.class_name] += 1
        passage = record.roundabout
        if passage is not None:
            if passage.entry_arm is not None and passage.exit_arm is not None:
                od[(passage.entry_arm, passage.exit_arm)] += 1
        elif record.entry is not None and record.exit is not None:
            od[(record.entry.zone, record.exit.zone)] += 1
        if record.speed_kmh is not None:
            speeds[track.class_name].append(record.speed_kmh.median)
        for crossing in line_crossings(track, scene.lines, 3.0 / settings.detect_fps):
            lines[crossing.line][crossing.direction][crossing.class_name] += 1

    qualities = [r.quality for r in records]
    mean_switch = (
        float(np.mean([q.id_switches for q in qualities])) if qualities else 0.0
    )
    mean_occ = (
        float(np.mean([q.occluded_frac for q in qualities])) if qualities else 0.0
    )
    speed_summary: dict[str, SpeedSummary] = {}
    for class_name, values in speeds.items():
        median, p85 = _median_p85(values)
        speed_summary[class_name] = SpeedSummary(
            n=len(values), median_kmh=round(median, 2), p85_kmh=round(p85, 2)
        )
    time_bins = [
        TimeBin(
            start_s=index * settings.time_bin_s,
            end_s=(index + 1) * settings.time_bin_s,
            by_class=dict(counts),
        )
        for index, counts in sorted(bins.items())
    ]
    caveats = [
        "Counts are from tracker output; ID switches and occlusion can inflate counts.",
        "Validate against the hand count before trusting any gate.",
    ]
    if ring.seen:
        caveats.append(
            "Roundabout entries and exits need the track to be seen in an arm zone "
            "and in the ring; tracks that start or end inside the ring are not counted."
        )
    if ring.first_seen_on_ring is not None:
        caveats.append(
            "Tracks first seen on the ring are not confirmed entries; broken tracks "
            "can create duplicates; check the ring and exclusion boundaries."
        )
    if calibration_id is None:
        caveats.append(
            "No calibration: speeds and ground-plane metrics are unavailable."
        )
    return StatsReport(
        video_id=video_id,
        calibration_id=calibration_id,
        total_vehicles=len(vehicle_tracks),
        by_class=dict(by_class),
        entries_by_zone={k: dict(v) for k, v in entries.items()},
        exits_by_zone={k: dict(v) for k, v in exits.items()},
        line_crossings={
            line: {d: dict(c) for d, c in dirs.items()} for line, dirs in lines.items()
        },
        od_matrix=[
            ODCell(entry=entry, exit=exit_, count=count)
            for (entry, exit_), count in sorted(od.items())
        ],
        time_bins=time_bins,
        speed_by_class=speed_summary,
        mean_id_switches=round(mean_switch, 3),
        mean_occluded_frac=round(mean_occ, 4),
        pedestrian_tracks=sum(1 for t in tracks if t.class_name == "person"),
        roundabout=ring.stats(),
        caveats=caveats,
    )
