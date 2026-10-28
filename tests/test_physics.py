"""M13: physics tests -- simulation, weak init, shooting fit, metrics."""

import json
from pathlib import Path

import numpy as np
import pytest

from pixelphys import expr, physics

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load_synthetic_truth() -> dict:
    return json.loads((FIXTURES / "toss_drag.json").read_text())


def test_simulate_matches_analytic_parabola() -> None:
    t = np.linspace(0.0, 1.0, 31)
    q, v = physics.simulate(["1"], np.array([-9.81]), t, (0.0, 6.0))
    assert np.allclose(q, 6.0 * t - 0.5 * 9.81 * t**2, atol=1e-6)
    assert np.allclose(v, 6.0 - 9.81 * t, atol=1e-6)


def test_simulate_rejects_bad_coeff_count() -> None:
    with pytest.raises(ValueError, match="coefficients"):
        physics.simulate(["1", "q"], np.array([1.0]), np.array([0.0, 0.1]), (0.0, 0.0))


def test_sign_term_integrates_without_blowup() -> None:
    t = np.linspace(0.0, 0.5, 16)
    q, v = physics.simulate(["1", "sign(v)"], np.array([0.0, -0.5]), t, (0.0, 2.0))
    assert np.all(np.isfinite(q)) and np.all(np.isfinite(v))
    assert v[-1] < v[0]  # Coulomb-style damping removes speed


def test_fit_recovers_synthetic_drag_truth() -> None:
    truth = load_synthetic_truth()
    t = np.array(truth["t"])
    q = np.array(truth["q"])
    cut = int(len(t) * 0.6)
    fit = physics.fit_terms(["1", "abs(v)*v"], t[:cut], q[:cut])
    assert fit.coeffs[0] == pytest.approx(-truth["params"]["gravity"], rel=0.1)
    assert fit.coeffs[1] == pytest.approx(-truth["params"]["drag"], rel=0.1)


def test_baseline_fits_worse_than_true_model_on_synthetic() -> None:
    # Rollouts continue each model's own fitted trajectory from its fitted
    # (q0, v0): restarting from a data-gradient anchor injects edge-smoothing
    # error (~0.12 NRMSE even for a perfect model), which would swamp the
    # structure comparison the ablation needs.
    truth = load_synthetic_truth()
    t = np.array(truth["t"])
    q = np.array(truth["q"])
    cut = int(len(t) * 0.6)
    t_fit, q_fit, q_held = t[:cut], q[:cut], q[cut:]
    full = physics.fit_terms(["1", "abs(v)*v"], t_fit, q_fit)
    base = physics.fit_terms(list(physics.BASELINE_TERMS), t_fit, q_fit)
    nrmse_full = physics.nrmse(physics.simulate(full.terms, full.coeffs, t, (full.q0, full.v0))[0][cut:], q_held)
    nrmse_base = physics.nrmse(physics.simulate(base.terms, base.coeffs, t, (base.q0, base.v0))[0][cut:], q_held)
    assert nrmse_full < 1e-6
    assert nrmse_base > 0.01


def test_nrmse_perfect_zero_and_degenerate_guards() -> None:
    a = np.array([1.0, 2.0, 3.0])
    assert physics.nrmse(a, a) == 0.0
    assert physics.nrmse(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.0
    assert physics.nrmse(np.array([1.0, 2.0]), np.array([1.0, 1.0])) == float("inf")


def test_bic_prefers_fewer_params_at_equal_fit() -> None:
    assert physics.bic(100, 1.0, 2) < physics.bic(100, 1.0, 7)
    with pytest.raises(ValueError):
        physics.bic(0, 1.0, 2)


def test_weak_init_needs_enough_points() -> None:
    with pytest.raises(ValueError, match="points"):
        physics.weak_init(["1", "q", "v"], np.array([0.0, 0.1]), np.array([0.0, 0.1]))


def test_fit_terms_rejects_empty_library() -> None:
    with pytest.raises(ValueError, match="1..8"):
        physics.fit_terms([], np.array([0.0, 0.1, 0.2]), np.array([0.0, 0.1, 0.2]))


def test_fit_recovers_truth_in_pixel_units() -> None:
    """Scale-aware bounds: pixel-scale gravity (~1e4) must not clip.

    Regression: a fixed ±1e3 bound pinned gravity and the drag fit went
    wrong-sign on real footage. Synthetic truth scaled by 1000 px/m.
    """
    truth = load_synthetic_truth()
    px_per_m = 1000.0
    t = np.array(truth["t"])
    q = np.array(truth["q"]) * px_per_m
    cut = int(len(t) * 0.6)
    fit = physics.fit_terms(["1", "abs(v)*v"], t[:cut], q[:cut])
    assert fit.coeffs[0] == pytest.approx(-truth["params"]["gravity"] * px_per_m, rel=0.1)
    assert fit.coeffs[1] == pytest.approx(-truth["params"]["drag"] / px_per_m, rel=0.2)


def test_expr_basis_available_to_physics() -> None:
    assert expr.validate_term("abs(v)*v") == "abs(v)*v"
