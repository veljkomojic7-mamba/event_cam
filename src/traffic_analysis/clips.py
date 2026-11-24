"""Evidence clips for ranked events: frame-accurate re-encode plus a JSON sidecar."""

from pathlib import Path

from pydantic import Field

from traffic_analysis.ffmpeg_tools import (
    SampleRequest,
    build_sample_command,
    run_ffmpeg,
)
from traffic_analysis.schemas import (
    AnalysisSettings,
    EventRecord,
    StrictModel,
    save_model,
)


class ClipSidecar(StrictModel):
    """Maps a clip back to the source video, event and tracks. No burned-in overlays."""

    event_id: str
    source: str
    clip: str
    start_s: float = Field(ge=0)
    end_s: float = Field(gt=0)
    event_start_s: float = Field(ge=0)
    event_end_s: float = Field(ge=0)
    tracks: list[int]


def clip_window(
    event: EventRecord, duration_s: float, pad_s: float
) -> tuple[float, float]:
    """Padded window for an event, clamped to the video."""
    start = max(0.0, event.t_start_s - pad_s)
    end = min(duration_s, event.t_end_s + pad_s)
    if end <= start:
        raise ValueError(f"event {event.event_id} has an empty clip window")
    return start, end


def cut_event_clips(
    events: list[EventRecord],
    source: Path,
    duration_s: float,
    out_dir: Path,
    settings: AnalysisSettings,
) -> list[EventRecord]:
    """Write one clip and one sidecar per event; return events with clip paths set."""
    out_dir.mkdir(parents=True, exist_ok=True)
    updated: list[EventRecord] = []
    for event in events:
        start, end = clip_window(event, duration_s, settings.clip_pad_s)
        clip_path = out_dir / f"{event.event_id}.mp4"
        run_ffmpeg(
            build_sample_command(
                SampleRequest(
                    source=source,
                    start_s=start,
                    duration_s=end - start,
                    output=clip_path,
                )
            )
        )
        sidecar = ClipSidecar(
            event_id=event.event_id,
            source=source.name,
            clip=clip_path.name,
            start_s=start,
            end_s=end,
            event_start_s=event.t_start_s,
            event_end_s=event.t_end_s,
            tracks=event.tracks,
        )
        save_model(sidecar, out_dir / f"{event.event_id}.json")
        updated.append(event.model_copy(update={"clip": clip_path.name}))
    return updated
