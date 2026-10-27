"""M5: manual scale/pivot. (Day 2.)

No-scale clips are first-class: the pipeline discovers structure and scores
predictions in pixels; physical units are a bonus applied only when a scale
is supplied via --scale (pixels per meter) from a known in-plane length
(e.g. the 30 cm ruler) or a prior calibration.
"""
from __future__ import annotations


def to_meters(values: list[float], px_per_m: float | None) -> list[float]:
    """Pixels pass through untouched when no scale is known."""
    if px_per_m is None:
        return list(values)
    if px_per_m <= 0:
        raise ValueError(f"px_per_m must be positive, got {px_per_m}")
    return [v / px_per_m for v in values]


def describe_units(px_per_m: float | None) -> str:
    return "m" if px_per_m is not None else "px"
