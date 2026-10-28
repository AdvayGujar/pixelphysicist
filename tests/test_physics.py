"""M13: physics tests -- simulation, weak init, shooting fit, metrics."""

import json
from pathlib import Path

import numpy as np
import pytest

from pixelphys import physics

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


@pytest.mark.slow
def test_fit_recovers_synthetic_drag_truth() -> None:
    truth = load_synthetic_truth()
    t = np.array(truth["t"])
    q = np.array(truth["q"])
    cut = int(len(t) * 0.6)
    fit = physics.fit_terms(["1", "abs(v)*v"], t[:cut], q[:cut])
    assert fit.coeffs[0] == pytest.approx(-truth["params"]["gravity"], rel=0.1)
    assert fit.coeffs[1] == pytest.approx(-truth["params"]["drag"], rel=0.1)


@pytest.mark.slow
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


def test_simulate_validates_time_grid() -> None:
    with pytest.raises(ValueError, match=">= 2 points"):
        physics.simulate(["1"], np.array([-9.81]), np.array([0.0]), (0.0, 0.0))
    with pytest.raises(ValueError, match=">= 2 points"):
        physics.simulate(["1"], np.array([-9.81]), np.array([]), (0.0, 0.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        physics.simulate(["1"], np.array([-9.81]), np.array([0.2, 0.1, 0.0]), (0.0, 0.0))
    with pytest.raises(ValueError, match="finite"):
        physics.simulate(["1"], np.array([np.inf]), np.array([0.0, 0.1]), (0.0, 0.0))


@pytest.mark.slow
def test_score_candidate_reports_success() -> None:
    import math

    truth = load_synthetic_truth()
    t = np.array(truth["t"])
    q = np.array(truth["q"])
    cut = int(len(t) * 0.6)
    scored = physics.score_candidate(["1", "abs(v)*v"], t[:cut], q[:cut], t[cut:], q[cut:])
    assert scored.ok and scored.error is None
    assert scored.held_nrmse < 1e-6
    assert scored.n_scored == len(t) - cut  # horizon exceeds the tail here
    n, k = cut, 2 + 2  # coefficients + 2 initial conditions
    rss = scored.train_rmse**2 * n
    assert scored.bic == pytest.approx(n * math.log(rss / n) + k * math.log(n))


def test_score_candidate_converts_failure_to_data() -> None:
    truth = load_synthetic_truth()
    t = np.array(truth["t"])
    q = np.array(truth["q"])
    cut = int(len(t) * 0.6)
    timed_out = physics.score_candidate(["1", "abs(v)*v"], t[:cut], q[:cut],
                                        t[cut:], q[cut:], budget_s=0.0)
    assert not timed_out.ok and "budget" in timed_out.error
    assert timed_out.held_nrmse == float("inf")
    empty = physics.score_candidate(["1"], t[:cut], q[:cut], t[:0], q[:0])
    assert not empty.ok and "held-out" in empty.error


def test_rollout_score_exact_truth_near_zero() -> None:
    """Exact coefficients on the synthetic clip must score ~0.

    Regression: the scorer once started the tail integration with the t[0]
    state, shifting the trajectory 3 frames and scoring 0.23 for truth.
    """
    truth = load_synthetic_truth()
    t = np.array(truth["t"])
    q = np.array(truth["q"])
    held, n = physics.rollout_score(
        ["1", "abs(v)*v"],
        np.array([-truth["params"]["gravity"], -truth["params"]["drag"]]), t, q)
    assert held < 1e-6
    assert n == len(t) - 3


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


@pytest.mark.slow
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


def test_weak_init_estimates_synthetic_truth() -> None:
    """Boundary-corrected weak form lands near truth on clean data."""
    truth = load_synthetic_truth()
    t = np.array(truth["t"])
    q = np.array(truth["q"])
    cut = int(len(t) * 0.6)
    weak = physics.weak_init(["1", "abs(v)*v"], t[:cut], q[:cut])
    assert weak[0] == pytest.approx(-truth["params"]["gravity"], rel=0.1)
    assert weak[1] == pytest.approx(-truth["params"]["drag"], rel=0.2)


def test_fit_terms_rejects_underdetermined_fit() -> None:
    t = np.linspace(0.0, 0.1, 5)
    with pytest.raises(ValueError, match="need >="):
        physics.fit_terms(["1", "q", "v", "v**2"], t, t**2)


def test_weak_init_rejects_nonfinite_basis() -> None:
    t = np.linspace(0.0, 1.0, 21)
    q = np.linspace(-1.0, 1.0, 21)  # hits q == 0 exactly: 1/q blows up
    with pytest.raises(ValueError, match="non-finite"):
        physics.weak_init(["1/q"], t, q)


def test_sim_time_limit_aborts_hanging_call() -> None:
    """The SIGALRM guard (Day-5 N pilot) must fire inside a slow call."""
    import time

    with pytest.raises(TimeoutError, match="exceeded"):
        with physics._sim_time_limit(0.05):
            time.sleep(0.5)
