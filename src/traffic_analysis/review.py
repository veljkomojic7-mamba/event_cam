"""Clip review: a null default, plus a gated Gemini reviewer with logging and budgets.

Nothing leaves the machine unless the caller passes an explicit cloud opt-in and
GEMINI_API_KEY is set. Every model call is appended to model_calls.jsonl with the
clip hash and prompt hash, never the video bytes.
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from traffic_analysis.ffmpeg_tools import (
    SampleRequest,
    build_sample_command,
    run_ffmpeg,
)
from traffic_analysis.schemas import (
    AIReview,
    AnalysisSettings,
    EventRecord,
    ReviewRequest,
    ReviewVerdict,
)

DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"
UPLOAD_POLL_S = 2.0
UPLOAD_TIMEOUT_S = 300.0
PROMPT = (
    "You are reviewing a short clip from a fixed traffic camera. The clip shows a "
    "candidate near-miss between road users. Decide whether the clip shows a real "
    "conflict (confirmed), no real conflict (rejected), or that you cannot judge "
    "without more detail (needs_video). If you need more detail, request up to two "
    "time ranges in seconds relative to the start of this clip, each at most 30 s. "
    "Reply only with JSON matching the schema."
)


class ClipReviewer(Protocol):
    """Returns a verdict for one clip, or None when no review was performed."""

    def review(self, event: EventRecord, clip: Path) -> ReviewVerdict | None: ...


class NullReviewer:
    """Default reviewer: makes no calls and leaves every event unreviewed."""

    def review(self, event: EventRecord, clip: Path) -> ReviewVerdict | None:
        return None


@dataclass
class ReviewBudget:
    """Hard caps on model calls and clip-range requests for one run."""

    max_calls: int
    max_requests: int
    calls_made: int = 0
    requests_made: int = 0

    def take_call(self) -> None:
        if self.calls_made >= self.max_calls:
            raise RuntimeError(f"model call budget of {self.max_calls} exhausted")
        self.calls_made += 1

    def take_requests(self, count: int) -> None:
        if self.requests_made + count > self.max_requests:
            raise RuntimeError(f"clip request budget of {self.max_requests} exhausted")
        self.requests_made += count


def validate_requests(
    requests: list[ReviewRequest], clip_duration_s: float, settings: AnalysisSettings
) -> list[ReviewRequest]:
    """Keep only requests that are ordered, within the clip, and at most 30 s long."""
    valid: list[ReviewRequest] = []
    for request in requests:
        if request.end_s <= request.start_s:
            continue
        if request.end_s - request.start_s > settings.review_request_max_s:
            continue
        if request.end_s > clip_duration_s:
            continue
        valid.append(request)
    return valid[: settings.review_max_requests]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class GeminiReviewer:
    """Sends a clip to Gemini and parses a validated ReviewVerdict."""

    client: Any
    model: str
    log_path: Path
    budget: ReviewBudget
    cache: dict[str, ReviewVerdict] = field(default_factory=dict)

    @classmethod
    def from_env(
        cls, log_path: Path, budget: ReviewBudget, api_key: str | None = None
    ) -> "GeminiReviewer":
        from google import genai

        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set; cloud review is unavailable")
        model = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        return cls(
            client=genai.Client(api_key=key),
            model=model,
            log_path=log_path,
            budget=budget,
        )

    def review(self, event: EventRecord, clip: Path) -> ReviewVerdict | None:
        from google.genai import types

        clip_hash = sha256_file(clip)
        prompt_hash = hashlib.sha256(PROMPT.encode()).hexdigest()
        cache_key = f"{clip_hash}:{self.model}:{prompt_hash}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        self.budget.take_call()
        uploaded = self.client.files.upload(file=clip)
        self._wait_active(uploaded)
        response = self.client.models.generate_content(
            model=self.model,
            contents=[uploaded, PROMPT],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ReviewVerdict,
            ),
        )
        verdict = ReviewVerdict.model_validate_json(response.text or "")
        verdict = verdict.model_copy(update={"event_id": event.event_id})
        self._log(event, clip, clip_hash, prompt_hash, verdict)
        self.cache[cache_key] = verdict
        return verdict

    def _wait_active(self, uploaded: Any) -> None:
        """Poll the uploaded file until Gemini has finished processing it."""
        deadline = time.monotonic() + UPLOAD_TIMEOUT_S
        current = uploaded
        while "ACTIVE" not in str(current.state):
            if "FAILED" in str(current.state):
                raise RuntimeError("Gemini file processing failed")
            if time.monotonic() > deadline:
                raise RuntimeError("timed out waiting for Gemini file processing")
            time.sleep(UPLOAD_POLL_S)
            current = self.client.files.get(name=current.name)

    def _log(
        self,
        event: EventRecord,
        clip: Path,
        clip_hash: str,
        prompt_hash: str,
        verdict: ReviewVerdict,
    ) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "at": datetime.now(UTC).isoformat(),
            "event_id": event.event_id,
            "model": self.model,
            "clip": clip.name,
            "clip_sha256": clip_hash,
            "prompt_sha256": prompt_hash,
            "verdict": verdict.verdict,
            "confidence": verdict.confidence,
        }
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def review_event(
    reviewer: ClipReviewer,
    event: EventRecord,
    clip: Path,
    source: Path,
    clip_start_s: float,
    clip_duration_s: float,
    settings: AnalysisSettings,
    budget: ReviewBudget,
    work_dir: Path,
) -> AIReview:
    """Review one clip, following at most ``review_max_rounds`` requested ranges."""
    verdict = reviewer.review(event, clip)
    if verdict is None:
        return AIReview()
    rounds = 0
    while (
        verdict.requested_ranges
        and rounds < settings.review_max_rounds
        and verdict.verdict == "needs_video"
    ):
        valid = validate_requests(verdict.requested_ranges, clip_duration_s, settings)
        if not valid:
            break
        budget.take_requests(len(valid))
        rounds += 1
        for index, request in enumerate(valid):
            detail = work_dir / f"{event.event_id}_r{rounds}_{index}.mp4"
            run_ffmpeg(
                build_sample_command(
                    SampleRequest(
                        source=source,
                        start_s=clip_start_s + request.start_s,
                        duration_s=request.end_s - request.start_s,
                        output=detail,
                    )
                )
            )
            followup = reviewer.review(event, detail)
            if followup is not None:
                verdict = followup
    return AIReview(
        verdict=verdict.verdict, confidence=verdict.confidence, notes=verdict.notes
    )
