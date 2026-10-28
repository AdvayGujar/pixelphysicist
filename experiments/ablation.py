"""M12: A/B ablation runner -> results/ablation.csv. (Day 5.)

Condition preparation is FROZEN here (audit note, Day 4): the H/N/S rows
must be reproducible exactly, so every choice mirrors
``scripts/run_baseline.py``' loaders rather than living in ad-hoc scripts.
Day-5 work wires the Spark loop around ``load_condition``; nothing below
may change without a plan amendment.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixelphys import trajectory as T  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"

# Pre-registered Day-5 protocol (plan §7): reasoning_effort="low" for all
# 12 repeats; every row carries terms, coeffs, held_nrmse, bic, rounds, cost.
EFFORT = "low"
N_REPEATS = 3
HORIZON = "min(2 s, clean tail)"

# Condition table: (fixture, axis, trim_free_flight). Axis q(t) follows the
# Day-3 convention -- hero is vertical displacement, pendulum is
# rest-relative horizontal displacement (pivot off-frame). Synthetic runs
# from TRACKED observations (px -> coordinate via the same path), never
# from truth arrays (Day-4 audit finding 2).
CONDITIONS: dict[str, dict[str, object]] = {
    # Hero: tossed ball, free-flight trim (launch -> first impact).
    "H": {"fixture": "obs_hero_toss_take5.json", "axis": "y", "trim": True},
    # Negative control: pendulum, full cleaned window (no trim).
    "N": {"fixture": "obs_pendulum_take1.json", "axis": "x", "trim": False},
    # Synthetic: tracked observations of the known-drag toss.
    "S": {"fixture": "obs_synthetic_toss_drag.json", "axis": "y",
          "trim": False},
}


def load_condition(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a frozen condition exactly as scripts/run_baseline.py does.

    The synthetic condition goes through the identical tracked-observation
    pipeline (load -> clean -> longest run -> resample -> smooth ->
    coordinate); only the fixture differs. That is what makes the S row
    end-to-end (Day-4 audit finding 2).
    """
    spec = CONDITIONS[name]
    t, q = _real_clip(str(spec["fixture"]), str(spec["axis"]),
                      bool(spec["trim"]))
    return np.asarray(t, dtype=float), np.asarray(q, dtype=float)


def _real_clip(fixture: str, axis: str, trim: bool) -> tuple[np.ndarray,
                                                             np.ndarray]:
    """Hero/pendulum path: longest gap-free run, optional free-flight trim."""
    obs = T.load_observations(FIXTURES / fixture)
    gaps = np.diff([o.t for o in obs])
    fps = 1.0 / float(np.median(gaps))
    obs = T.clean_observations(obs)
    obs = max(T.split_by_gaps(obs), key=len)
    t, cx, cy = T.resample_uniform(obs, fps)
    raw = cx if axis == "x" else cy
    q = (T.rest_relative_q(T.smooth_position(raw)) if axis == "x"
         else T.vertical_q(T.smooth_position(raw)))
    if trim:
        start, end = T.trim_free_flight(t, q)
        t, q = t[start:end], q[start:end]
    return t, q
