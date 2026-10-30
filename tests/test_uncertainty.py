"""Coefficient uncertainty from the shooting Jacobian (Task 4).

House pattern: exact-truth identity first, then noisy recovery. The
asymptotic covariance collapses on the degenerate noiseless fixture
(RSS ~ 1e-19 at the integrator floor), so the coverage test uses the
same toss with fixed-seed noise -- coverage is only defined with noise.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pixelphys import physics as P

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_coefficient_stderr_covers_truth_on_noisy_synthetic() -> None:
    truth = json.loads((FIXTURES / "toss_drag.json").read_text())
    t = np.array(truth["t"])
    q = np.array(truth["q"])
    rng = np.random.default_rng(0)
    q_noisy = q + rng.normal(0.0, 2e-4, size=q.shape)
    cut = int(len(t) * 0.6)
    fit = P.fit_terms(["1", "abs(v)*v"], t[:cut], q_noisy[:cut])
    err = P.coefficient_stderr(fit)
    assert err.shape == (2,)
    assert bool(np.all(np.isfinite(err))) and bool(np.all(err > 0))
    assert abs(fit.coeffs[0] + 9.81) <= 2 * err[0]
    assert abs(fit.coeffs[1] + 0.35) <= 2 * err[1]


@pytest.mark.slow
def test_stderr_sane_on_clean_synthetic() -> None:
    truth = json.loads((FIXTURES / "toss_drag.json").read_text())
    t = np.array(truth["t"])
    q = np.array(truth["q"])
    cut = int(len(t) * 0.6)
    fit = P.fit_terms(["1", "abs(v)*v"], t[:cut], q[:cut])
    err = P.coefficient_stderr(fit)
    assert err.shape == (2,)
    assert bool(np.all(np.isfinite(err)))
    assert bool(np.all(err > 0))
    # Integrator-floor tight: coefficients match truth to ~1e-7 relative.
    assert abs(fit.coeffs[0] + 9.81) / 9.81 < 1e-5
    assert abs(fit.coeffs[1] + 0.35) / 0.35 < 1e-4


def test_coefficient_stderr_bad_shape_gives_nan() -> None:
    fit = P.FitResult(terms=["1", "abs(v)*v"], coeffs=np.array([1.0, 2.0]),
                      q0=0.0, v0=0.0)
    assert bool(np.all(np.isnan(P.coefficient_stderr(fit))))
