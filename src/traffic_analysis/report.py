"""Markdown report from computed stats, plus an optional gated LLM narrative."""

import re
from pathlib import Path
from typing import Any

from traffic_analysis.analytics import StatsReport
from traffic_analysis.schemas import EventRecord

NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def _od_lines(stats: StatsReport) -> list[str]:
    lines = ["", "## Origin-destination"]
    if stats.od_matrix:
        lines.append("| entry | exit | count |")
        lines.append("|---|---|---|")
        lines += [f"| {c.entry} | {c.exit} | {c.count} |" for c in stats.od_matrix]
    else:
        lines.append("No vehicles with both an entry and an exit zone.")
    return lines


def _zone_lines(stats: StatsReport) -> list[str]:
    lines = ["", "## Entry and exit zones"]
    for zone, classes in sorted(stats.entries_by_zone.items()):
        lines.append(f"- entries via {zone}: {sum(classes.values())}")
    for zone, classes in sorted(stats.exits_by_zone.items()):
        lines.append(f"- exits via {zone}: {sum(classes.values())}")
    return lines + _od_lines(stats)


def _roundabout_lines(stats: StatsReport) -> list[str]:
    ring = stats.roundabout
    assert ring is not None
    lines = [
        "",
        "## Roundabout",
        f"- Entered the roundabout: {ring.entered}",
        f"- Exited the roundabout: {ring.exited}",
        f"- Entered from an exit lane (wrong way): {ring.wrong_way_entries}",
        f"- Exited into an approach lane (wrong way): {ring.wrong_way_exits}",
    ]
    if ring.first_seen_on_ring is not None:
        lines.append(f"- Tracks first seen on ring: {len(ring.first_seen_on_ring)}")
        lines.append(
            "  These are not confirmed entries; broken tracks may create duplicates."
        )
        for item in ring.first_seen_on_ring:
            lines.append(f"  - Track {item.track_id}: {item.t_s:.1f} s")
    else:
        lines.append("- Tracks first seen on ring: unavailable for this run")
    lines += ["", "| arm | entered | exited | wrong-way entries |", "|---|---|---|---|"]
    arms = sorted(set(ring.entered_by_arm) | set(ring.exited_by_arm))
    for arm in arms:
        lines.append(
            f"| {arm} | {ring.entered_by_arm.get(arm, 0)} | "
            f"{ring.exited_by_arm.get(arm, 0)} | "
            f"{ring.wrong_way_entries_by_arm.get(arm, 0)} |"
        )
    lines += ["", "### Wrong-way entry examples"]
    if ring.wrong_way_examples:
        lines.append("| track | from | time s |")
        lines.append("|---|---|---|")
        lines += [
            f"| {e.track_id} | {e.arm} | {e.t_s:.1f} |" for e in ring.wrong_way_examples
        ]
    else:
        lines.append("None found.")
    return lines + _od_lines(stats)


def render_report(stats: StatsReport, events: list[EventRecord]) -> str:
    """Render the deterministic report. Every number comes from stats or events."""
    lines = [
        f"# Traffic report: {stats.video_id}",
        "",
        f"Calibration: {stats.calibration_id or 'none (speeds unavailable)'}",
        "",
        "## Counts",
        f"- Total vehicles: {stats.total_vehicles}",
        f"- Pedestrian tracks: {stats.pedestrian_tracks}",
    ]
    for class_name, count in sorted(stats.by_class.items()):
        lines.append(f"- {class_name}: {count}")
    lines += _roundabout_lines(stats) if stats.roundabout else _zone_lines(stats)
    lines += ["", "## Speeds (km/h)"]
    if stats.speed_by_class:
        for class_name, summary in sorted(stats.speed_by_class.items()):
            lines.append(
                f"- {class_name}: n={summary.n}, median {summary.median_kmh}, "
                f"p85 {summary.p85_kmh}"
            )
    else:
        lines.append("Speeds need a calibration file.")
    lines += ["", "## Tracking quality"]
    lines.append(f"- Mean ID switches per vehicle: {stats.mean_id_switches}")
    lines.append(f"- Mean occluded fraction: {stats.mean_occluded_frac}")
    lines += ["", "## Ranked events"]
    if events:
        lines.append(
            "| id | type | start s | end s | score | min sep m | TTC s | clip |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for event in events:
            m = event.metrics
            lines.append(
                f"| {event.event_id} | {event.type} | {event.t_start_s:.1f} | "
                f"{event.t_end_s:.1f} | {m.score} | {m.min_sep_m} | {m.ttc_min_s} | "
                f"{event.clip or '-'} |"
            )
        lines.append("")
        lines.append(
            "Events are candidates from geometry and tracker output. Check each clip "
            "before acting on it."
        )
    else:
        lines.append("No events passed the thresholds.")
    lines += ["", "## Caveats"]
    lines += [f"- {caveat}" for caveat in stats.caveats]
    return "\n".join(lines) + "\n"


def numbers_in(text: str) -> set[str]:
    """Numeric tokens in a text, used to check an LLM narrative against the stats."""
    return set(NUMBER_PATTERN.findall(text))


def allowed_numbers(stats: StatsReport, events: list[EventRecord]) -> set[str]:
    """Every number that appears in the deterministic report is allowed."""
    return numbers_in(render_report(stats, events))


def narrate(
    client: Any,
    model: str,
    stats: StatsReport,
    events: list[EventRecord],
    out_path: Path,
) -> bool:
    """Ask the model for a short narrative and keep it only if its numbers are grounded.

    Returns True when the narrative was written. The model sees only the report text,
    never the video.
    """
    base = render_report(stats, events)
    prompt = (
        "Write a short plain-language summary of this traffic report. Use only the "
        "numbers that appear in it. Do not invent figures.\n\n" + base
    )
    response = client.models.generate_content(model=model, contents=prompt)
    text = (response.text or "").strip()
    if not text:
        return False
    unknown = numbers_in(text) - allowed_numbers(stats, events)
    if unknown:
        return False
    out_path.write_text(text + "\n", encoding="utf-8")
    return True
