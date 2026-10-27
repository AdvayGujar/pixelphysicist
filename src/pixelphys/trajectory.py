"""M6: clean, resample, q(t), fit/held-out split. (Day 2.)"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .tracking import Observation

MIN_POINTS = 30
SMOOTH_WINDOW = 7
FIT_FRAC = 0.6


def clean_observations(obs: list[Observation]) -> list[Observation]:
    """Drop mask-area >3σ outliers. Requires MIN_POINTS survivors."""
    if len(obs) < MIN_POINTS:
        raise ValueError(f"need >= {MIN_POINTS} observations, got {len(obs)}")
    areas = np.array([o.mask_area for o in obs])
    keep = np.abs(areas - areas.mean()) <= 3.0 * areas.std()
    cleaned = [o for o, k in zip(obs, keep) if k]
    if len(cleaned) < MIN_POINTS:
        raise ValueError(f"cleaning left {len(cleaned)} < {MIN_POINTS} observations")
    return cleaned


def resample_uniform(obs: list[Observation], fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear-interpolate centroids onto a uniform time grid."""
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


def split_fit_held(t: np.ndarray, q: np.ndarray, frac: float = FIT_FRAC) -> tuple[np.ndarray, ...]:
    cut = int(len(t) * frac)
    return t[:cut], q[:cut], t[cut:], q[cut:]


def save_observations(obs: list[Observation], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(o) for o in obs]))
    return path


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
