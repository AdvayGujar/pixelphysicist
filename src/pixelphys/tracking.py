"""M3: Observation, SamVideoTracker, ColorTracker. (Smoke test Day 1, finish Day 2.)"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    frame_idx: int
    t: float
    cx: float
    cy: float
    bbox: tuple[float, float, float, float]
    mask_area: float
    confidence: float
