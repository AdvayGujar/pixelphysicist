"""Tests for the synthetic spring-mass and rolling-on-slope renderers."""
from __future__ import annotations

import json
import math
from pathlib import Path

from pixelphys.synthetic import (
    FPS,
    PX_PER_M,
    RollingParams,
    SpringParams,
    integrate_rolling,
    integrate_spring,
    render_rolling,
    render_spring,
)


def _upward_zero_crossing_period(times: list[float], q: list[float]) -> float:
    crossings: list[float] = []
    for i in range(1, len(q)):
        if q[i - 1] < 0.0 <= q[i]:
            crossings.append(times[i])
    assert len(crossings) >= 2, "need at least two periods of oscillation"
    gaps = [b - a for a, b in zip(crossings, crossings[1:])]
    return sum(gaps) / len(gaps)


def test_spring_decays_and_period_matches_sqrt_k():
    params = SpringParams()
    times, q, _ = integrate_spring(params)
    mid = len(q) // 2
    assert max(abs(v) for v in q[:mid]) > 0.0
    assert max(abs(v) for v in q[mid:]) < max(abs(v) for v in q[:mid])
    measured = _upward_zero_crossing_period(times, q)
    expected = 2.0 * math.pi / math.sqrt(params.stiffness)
    assert abs(measured - expected) / expected < 0.15


def test_spring_conservative_case_conserves_amplitude():
    params = SpringParams(q0=0.55, stiffness=22.0, damping=0.0, cubic=0.0, duration_s=4.0)
    _, q, _ = integrate_spring(params)
    expected_period = 2.0 * math.pi / math.sqrt(params.stiffness)
    tail = q[-int(expected_period * 240):]
    assert abs(max(abs(v) for v in tail) - params.q0) / params.q0 < 0.02


def test_rolling_sign_flips_once_and_returns():
    params = RollingParams()
    _, q, v = integrate_rolling(params)
    assert v[0] > 0.0
    signs = [1 if x > 0.0 else (-1 if x < 0.0 else 0) for x in v]
    flips = sum(1 for a, b in zip(signs, signs[1:]) if a != b and a != 0 and b != 0)
    # The stop instant may land exactly on v == 0; ignore zero samples.
    nonzero = [s for s in signs if s != 0]
    strict_flips = sum(1 for a, b in zip(nonzero, nonzero[1:]) if a != b)
    assert strict_flips == 1
    assert nonzero[-1] < 0.0
    assert max(q) > 1.0  # it really rolls up-slope first
    assert q[-1] < 0.5  # ... and comes back near/below the start


def test_rolling_without_friction_is_parabola():
    params = RollingParams(v0=3.4, slope_accel=1.6, coulomb=0.0, viscous=0.0, duration_s=6.0)
    times, q, _ = integrate_rolling(params, dt=1.0 / 2000.0)
    for t, qi in zip(times, q):
        assert abs(qi - (params.v0 * t - 0.5 * params.slope_accel * t * t)) < 1e-6


def _check_clip(mp4: Path, js: Path) -> None:
    import cv2

    assert mp4.exists() and js.exists()
    cap = cv2.VideoCapture(str(mp4))
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert n > 10
    finally:
        cap.release()
    truth = json.loads(js.read_text())
    assert len(truth["t"]) == len(truth["q"]) == len(truth["v"])


def test_render_spring(tmp_path: Path):
    mp4, js = tmp_path / "spring.mp4", tmp_path / "spring.json"
    render_spring(mp4, js)
    _check_clip(mp4, js)
    truth = json.loads(js.read_text())
    assert truth["px_per_m"] == PX_PER_M
    assert truth["fps"] == FPS
    assert "centre_y_px" in truth


def test_render_rolling(tmp_path: Path):
    mp4, js = tmp_path / "rolling.mp4", tmp_path / "rolling.json"
    render_rolling(mp4, js)
    _check_clip(mp4, js)
    truth = json.loads(js.read_text())
    assert truth["px_per_m"] == PX_PER_M
    assert truth["fps"] == FPS


# --- Scientist-mode recovery: the fitter must find each mode's hidden term.

import numpy as np
import pytest

from pixelphys import physics as P
from pixelphys.synthetic import sample_at_fps


def _sampled_spring():
    times, q, v = integrate_spring(SpringParams())
    return (np.array(sample_at_fps(times, times)),
            np.array(sample_at_fps(times, q)))


def _sampled_rolling():
    times, q, v = integrate_rolling(RollingParams())
    return (np.array(sample_at_fps(times, times)),
            np.array(sample_at_fps(times, q)))


@pytest.mark.slow
def test_spring_fit_recovers_cubic_stiffness() -> None:
    t, q = _sampled_spring()
    cut = int(len(t) * 0.6)
    true_model = P.score_candidate(["1", "q", "v", "q**3"],
                                   t[:cut], q[:cut], t[cut:], q[cut:])
    assert true_model.ok
    coeffs = dict(zip(true_model.terms, true_model.coeffs))
    assert coeffs["q"] == pytest.approx(-22.0, rel=0.1)
    assert coeffs["v"] == pytest.approx(-0.25, rel=0.25)
    assert coeffs["q**3"] == pytest.approx(-6.0, rel=0.15)
    baseline = P.score_candidate(list(P.BASELINE_TERMS),
                                 t[:cut], q[:cut], t[cut:], q[cut:])
    assert baseline.ok and baseline.held_nrmse > true_model.held_nrmse


@pytest.mark.slow
def test_rolling_fit_recovers_coulomb_friction() -> None:
    t, q = _sampled_rolling()
    cut = int(len(t) * 0.6)
    true_model = P.score_candidate(["1", "sign(v)", "v"],
                                   t[:cut], q[:cut], t[cut:], q[cut:])
    assert true_model.ok
    coeffs = dict(zip(true_model.terms, true_model.coeffs))
    assert coeffs["1"] == pytest.approx(-1.6, rel=0.1)
    assert coeffs["sign(v)"] == pytest.approx(-0.55, rel=0.15)
    assert coeffs["v"] == pytest.approx(-0.05, rel=0.3)
    baseline = P.score_candidate(list(P.BASELINE_TERMS),
                                 t[:cut], q[:cut], t[cut:], q[cut:])
    assert baseline.ok and baseline.held_nrmse > true_model.held_nrmse
