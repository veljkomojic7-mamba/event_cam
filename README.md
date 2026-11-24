# Traffic analysis (local-first)

Local tools for analysing fixed-camera traffic video (roundabout, high overhead
view). It covers detection and tracking, counts and O-D, calibrated speeds,
near-miss candidates with evidence clips, and a Markdown report.

Video stays on your machine. Cloud review (Gemini) runs only with
`--allow-cloud`, and it is off by default.

## Pipeline

| Stage | Command | Output (in `--out-dir`) |
|-------|---------|-------------------------|
| 1. Detect and track | `detect` | `observations.sqlite` (YOLO + ByteTrack, GPU if available) |
| 2. Calibrate | `calibrate --camera-id C --points P.json --out cal.json` | `cal.json` (pixel to metre homography) |
| 3. Analyse | `analyze` | `vehicles.json`, `stats.json` (counts, O-D, speeds) |
| 4. Events | `events` | `events.json` (ranked candidates) |
| 5. Clips | `clips` | `clips/ev-NNN.mp4` and `.json` |
| 6. Review | `review` | writes verdicts into `events.json` (needs `--allow-cloud`) |
| 7. Report | `report` | `report.md` |

`run` chains every stage: `run --video V --out-dir O --scene S [--calibration cal.json] [--allow-cloud] [--narrate]`.

Without a calibration file, speeds and distance-based events are not computed.
`events` then writes an empty file and logs a warning.

## Calibration

Speed needs metres, so the homography must be fitted from known ground points.

1. Pick at least 4 ground points you can measure, such as lane markings or
   kerb corners, and read their pixel positions from a frame (`frame`).
2. Write a JSON list of pairs:

   ```json
   [
     {"image": {"x": 412, "y": 610}, "world": {"x": 0.0, "y": 0.0}},
     {"image": {"x": 1510, "y": 605}, "world": {"x": 12.0, "y": 0.0}},
     {"image": {"x": 1430, "y": 1020}, "world": {"x": 12.0, "y": 9.0}},
     {"image": {"x": 380, "y": 1015}, "world": {"x": 0.0, "y": 9.0}}
   ]
   ```

3. Run `calibrate`. Check the reported reprojection error before you trust speeds.

## Roundabout counts

When the scene has a zone with role `ring` (draw it over the circulating
carriageway; press `r` in the zone editor to cycle to `ring`), `analyze` counts
by movement, not by lane:

- Entered: a track seen in an arm zone and then in the ring. Exited: seen in the
  ring and then in an arm zone. Which lane or arm is used does not matter for
  the totals; per-arm counts are shown beside them.
- Wrong-way entry: the arm the vehicle came from has role `exit`. Wrong-way
  exit: it left into an `approach` arm.
- Arm zones win where they overlap the ring. Only the first ring visit can be an
  entry and only the last can be an exit, and `zone_min_obs` (default 2) drops
  one-frame boundary flicker.
- `events --scene S.json` (also done by `run`) writes one `wrong_way_entry`
  event per wrong-way entry (`ww-001`, ...), so `clips` cuts an example clip for
  each. This needs no calibration. These events are not sent to cloud review.

Tracks that start or end inside the ring are not counted as entries or exits, so
fragmented tracks undercount. Check the totals against a hand count.

For scenes with `ring_exclusion`, reports also show retained vehicle tracks whose
first observation is in the ring polygon, outside that exclusion and outside
the approach/exit zones. The DJI exclusion approximates the inner boundary from
the provided image; verify it using the GUI zone preview. This is a separate
diagnostic, not an entry count: tracks can begin mid-journey or fragment into
duplicates. The exclusion is used only for this diagnostic; existing passage
counting still uses the original filled ring polygon. The preview outlines the
exclusion in cyan. Older runs without this diagnostic show it as unavailable.

## GUI

`python -m uv sync --extra gui`, then `python -m uv run traffic-analysis gui --data-dir data/gui`
opens a local page. Choose a video and a scene, press Go, and read the roundabout
counts, wrong-way clips and stats. Runs happen in separate worker processes, so
closing the browser does not stop them (`src/traffic_analysis/jobs.py`).
Zones are still drawn with `traffic-analysis zones`; the GUI takes the scene JSON.

The GUI also offers an experimental aerial YOLO11m-OBB model trained on DOTA.
Its small- and large-vehicle labels are both reported as `car` for compatibility;
they do not establish vehicle type. This model does not detect pedestrians or
bicycles. Rotated boxes are converted to axis-aligned boxes before tracking
observations are counted. Compare results with a hand count, including missed
entries and exits. Model weights may be downloaded on first use.
Standard YOLO11l and YOLO11x are also selectable; they can be slower than
YOLO11m and are not guaranteed to improve accuracy. Each result shows the model
used so runs can be compared.

To run it on a Google Cloud GPU VM behind a password, see
[deploy/gcp/README.md](deploy/gcp/README.md).

The experimental Gemini frame review is a separate model option. After explicit
opt-in, it sends sampled JPEG frames, timestamps and the scene geometry to the
Gemini API, not the video file or YOLO results. It requires `GEMINI_API_KEY` in
the app environment and may incur API charges. Runs are limited to 60 seconds
and 10 fps, in independent five-second requests. The downloaded JSON contains
candidate events, not verified totals; movements at request boundaries may be
missed or duplicated. API failures fail the job rather than returning counts.

## Known limitations

- Accuracy is not validated. Compare counts with a hand count of a sample
  window before you rely on any number.
- The ground point is the bottom-centre of each box. In a near-overhead view,
  this is less accurate than a side view.
- Conflict metrics (TTC, DRAC, PET) are geometric approximations. Events are
  candidates for human review, not findings.
- Pass 2 windows are written to `events.json`, but the pipeline does not re-run
  detection on them automatically.
- Clips are cut without burned-in annotations.
- Static-frame skipping does not update the tracker, so long stationary tracks
  can split on real footage.

## Setup (Windows, PowerShell)

1. Install FFmpeg: `winget install Gyan.FFmpeg`. Open a new terminal afterwards
   so `ffmpeg` and `ffprobe` are on `PATH`.
2. Install uv: `pip install --user uv`, then use `python -m uv` if `uv` is not
   on `PATH`.
3. Install the project environment: `python -m uv sync`.

## CLI

Run commands with `python -m uv run traffic-analysis <command>`:

| Command | Purpose |
|---------|---------|
| `probe --video V --output J` | Write video metadata (fps, duration, size) as JSON. |
| `sample --video V --start S --duration D --output O.mp4` | Cut a clip. Writes `O.json` with the source and start time. |
| `frame --video V --time T --output F.png` | Save one frame as PNG. |
| `zones --frame F.png --scene S.json` | Open the zone editor on a frame. |

Add `--log-level DEBUG` for more output. The exit code is 1 on error, for
example when the video does not exist.

## Zone editor

`zones` opens a window on the frame. Controls:

- Left-click: add a vertex. Drag a vertex to move it.
- Enter: close the current polygon into a zone. `c` cancels the draft.
- `u`: undo the last vertex. `r`: cycle the role (other, approach, exit, both, ring).
- `x`: delete the last zone.
- `s`: save to the scene JSON. `q` or Esc: quit.

Scene files follow the schema in `src/traffic_analysis/schemas.py`. See
`configs/example.json` for a template with placeholder coordinates.

## Development

`make` is not installed by default on Windows. The Makefile targets are
equivalent to these commands:

- Lint: `python -m uv run ruff check .` and `python -m uv run ruff format --check .`
- Types: `python -m uv run mypy`
- Tests: `python -m uv run pytest`

On Linux or macOS, `make check` runs all of these.

## Privacy

Keep raw video and outputs under `data/` and `outputs/`. Both are git-ignored.
Do not commit `.env` files.

Cloud review sends clip files to Google. It runs only when you pass
`--allow-cloud` (or `CLOUD=1` with make) and set `GEMINI_API_KEY` in `.env`.
Run a data-classification check on your footage before enabling it. Review is
capped by call and clip-request budgets from the settings file.
