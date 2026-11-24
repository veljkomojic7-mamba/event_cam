"""SQLite store for per-frame tracked observations (the index into the video)."""

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    pass_no INTEGER NOT NULL,
    track_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    conf REAL NOT NULL,
    frame_idx INTEGER NOT NULL,
    t_s REAL NOT NULL,
    x1 REAL NOT NULL,
    y1 REAL NOT NULL,
    x2 REAL NOT NULL,
    y2 REAL NOT NULL,
    ground_x_px REAL NOT NULL,
    ground_y_px REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_obs_pass_track ON observations (pass_no, track_id);
CREATE INDEX IF NOT EXISTS idx_obs_pass_time ON observations (pass_no, t_s);
"""


@dataclass(frozen=True)
class Observation:
    """One tracked box in one processed frame.

    ``ground_x_px``/``ground_y_px`` is the bottom-centre of the box, the point
    used for zones, lines and ground-plane projection.
    """

    pass_no: int
    track_id: int
    class_name: str
    conf: float
    frame_idx: int
    t_s: float
    x1: float
    y1: float
    x2: float
    y2: float
    ground_x_px: float
    ground_y_px: float


def open_store(path: Path) -> sqlite3.Connection:
    """Open (and create if needed) the observation database."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def clear_pass(conn: sqlite3.Connection, pass_no: int) -> None:
    """Delete all observations of one pass so the pass can be re-run."""
    conn.execute("DELETE FROM observations WHERE pass_no = ?", (pass_no,))
    conn.commit()


def insert_observations(conn: sqlite3.Connection, rows: Iterable[Observation]) -> int:
    """Insert observations and return how many were written."""
    data = [
        (
            r.pass_no,
            r.track_id,
            r.class_name,
            r.conf,
            r.frame_idx,
            r.t_s,
            r.x1,
            r.y1,
            r.x2,
            r.y2,
            r.ground_x_px,
            r.ground_y_px,
        )
        for r in rows
    ]
    conn.executemany("INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", data)
    conn.commit()
    return len(data)


def load_observations(
    conn: sqlite3.Connection,
    pass_no: int,
    start_s: float | None = None,
    end_s: float | None = None,
) -> list[Observation]:
    """Load observations of one pass, optionally limited to a time window."""
    query = (
        "SELECT pass_no, track_id, class_name, conf, frame_idx, t_s, x1, y1, x2, y2,"
        " ground_x_px, ground_y_px FROM observations WHERE pass_no = ?"
    )
    params: list[float | int] = [pass_no]
    if start_s is not None:
        query += " AND t_s >= ?"
        params.append(start_s)
    if end_s is not None:
        query += " AND t_s <= ?"
        params.append(end_s)
    query += " ORDER BY track_id, t_s"
    rows = conn.execute(query, params).fetchall()
    return [
        Observation(
            pass_no=int(r[0]),
            track_id=int(r[1]),
            class_name=str(r[2]),
            conf=float(r[3]),
            frame_idx=int(r[4]),
            t_s=float(r[5]),
            x1=float(r[6]),
            y1=float(r[7]),
            x2=float(r[8]),
            y2=float(r[9]),
            ground_x_px=float(r[10]),
            ground_y_px=float(r[11]),
        )
        for r in rows
    ]
