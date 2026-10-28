"""M6: clean, resample, q(t), fit/held-out split. (Day 2.)"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .tracking import Observation, validate_observations

MIN_POINTS = 30
SMOOTH_WINDOW = 7
FIT_FRAC = 0.6


def clean_observations(obs: list[Observation]) -> list[Observation]:
    """Drop mask-area >3σ outliers. Requires MIN_POINTS survivors."""
    validate_observations(obs)
    if len(obs) < MIN_POINTS:
        raise ValueError(f"need >= {MIN_POINTS} observations, got {len(obs)}")
    areas = np.array([o.mask_area for o in obs])
    keep = np.abs(areas - areas.mean()) <= 3.0 * areas.std()
    cleaned = [o for o, k in zip(obs, keep) if k]
    if len(cleaned) < MIN_POINTS:
        raise ValueError(f"cleaning left {len(cleaned)} < {MIN_POINTS} observations")
    return cleaned


def split_by_gaps(obs: list[Observation],
                  max_gap_frames: int = 5) -> list[list[Observation]]:
    """Split into consecutive runs; single gap-tolerant window per run.

    Tracking dropouts (hand release, motion blur) end a run instead of being
    interpolated across. Callers analyze the longest run.
    """
    runs: list[list[Observation]] = []
    current: list[Observation] = []
    for o in obs:
        if current and o.frame_idx - current[-1].frame_idx > max_gap_frames:
            runs.append(current)
            current = []
        current.append(o)
    if current:
        runs.append(current)
    return runs


def resample_uniform(obs: list[Observation], fps: float,
                     max_gap_frames: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear-interpolate centroids onto a uniform time grid.

    Gaps longer than max_gap_frames raise instead of fabricating physics.
    """
    frames = np.array([o.frame_idx for o in obs])
    gaps = np.diff(frames)
    if len(gaps) and gaps.max() > max_gap_frames:
        at = int(frames[int(np.argmax(gaps))])
        raise ValueError(f"tracking gap of {int(gaps.max())} frames at {at}; "
                         "re-track or shorten the window")
    t = np.array([o.t for o in obs])
    grid = np.arange(t[0], t[-1] + 0.5 / fps, 1.0 / fps)
    x = np.interp(grid, t, np.array([o.cx for o in obs]))
    y = np.interp(grid, t, np.array([o.cy for o in obs]))
    return grid, x, y


def smooth_position(y: np.ndarray) -> np.ndarray:
    """Savitzky-Golay smoothing; window 7 or largest odd <= n."""
    from scipy.signal import savgol_filter

    n = len(y)
    window = min(SMOOTH_WINDOW, n if n % 2 else n - 1)
    if window < 3:
        raise ValueError(f"need >= 3 samples to smooth, got {n}")
    return savgol_filter(y, window, min(3, window - 1))


def vertical_q(cy: np.ndarray) -> np.ndarray:
    """Up-positive 1-DOF coordinate in pixels (image y grows downward)."""
    return -(cy - cy[0])


def rest_relative_q(x: np.ndarray) -> np.ndarray:
    """Zero-mean 1-DOF coordinate for oscillation about a rest position.

    Used for the pendulum, whose pivot is off-frame: for small angles the
    horizontal displacement is proportional to the swing angle.
    """
    return x - float(np.median(x))


def pendulum_angle(cx: np.ndarray, cy: np.ndarray,
                   pivot: tuple[float, float]) -> np.ndarray:
    """Swing angle from straight-down given a visible pivot (radians)."""
    return np.arctan2(cx - pivot[0], cy - pivot[1])


def split_fit_held(t: np.ndarray, q: np.ndarray, frac: float = FIT_FRAC) -> tuple[np.ndarray, ...]:
    """60/40 split. Index == time here because the grid is uniform."""
    cut = int(len(t) * frac)
    return t[:cut], q[:cut], t[cut:], q[cut:]


TRIM_MIN_POINTS = 15
TRIM_PERSIST_FRAMES = 3


def trim_free_flight(t: np.ndarray, q: np.ndarray) -> tuple[int, int]:
    """Indices [start, end) of the free-flight window: launch to first impact.

    Start is the first rise above 5% of apex height (cuts hand-hold);
    end is the first downward-velocity reversal that persists for
    TRIM_PERSIST_FRAMES (impact). Raises when no usable window exists.
    """
    if len(t) < 3:
        raise ValueError(f"need >= 3 samples to trim, got {len(t)}")
    apex = int(np.argmax(q))
    if apex < 1:
        raise ValueError("no rise before apex: not a free-flight arc")
    height = float(q[apex] - q[0])
    if height <= 0:
        raise ValueError("apex is not above launch: not a free-flight arc")
    start = next(i for i in range(apex) if q[i] - q[0] >= 0.05 * height)
    velocity = np.gradient(q, t)
    falling = velocity < 0.0
    end = len(t)
    for i in range(apex + 1, len(t) - TRIM_PERSIST_FRAMES + 1):
        if falling[i] and bool(np.all(~falling[i + 1:i + 1 + TRIM_PERSIST_FRAMES])):
            end = i + 1
            break
    if end - start < TRIM_MIN_POINTS:
        raise ValueError(f"free-flight window has {end - start} points, "
                         f"need >= {TRIM_MIN_POINTS}")
    return start, end


def save_observations(obs: list[Observation], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(o) for o in obs]))
    return path


def cache_path(clip: Path, prompt: str) -> Path:
    """Cache key includes prompt + clip size/mtime: re-films and re-prompts re-track."""
    from .config import RUNS_DIR

    stat = clip.stat()
    slug = "".join(c if c.isalnum() else "_" for c in prompt)[:32]
    return RUNS_DIR / f"obs_{clip.stem}_{slug}_{stat.st_size}_{int(stat.st_mtime)}.json"


def load_observations(path: Path) -> list[Observation]:
    loaded = []
    for row in json.loads(path.read_text()):
        row["bbox"] = tuple(row["bbox"])
        loaded.append(Observation(**row))
    return loaded


def trajectory_summary(t: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    return {
        "n": len(t),
        "duration_s": round(float(t[-1] - t[0]), 3),
        "apex_px": round(float(q.max()), 1),
    }
