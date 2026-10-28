"""M13: 2-DOF projectile mode -- grammar, fitter, synthetic truth, tools."""

import numpy as np
import pytest

from pixelphys import expr, physics as P, synthetic as syn


def projectile_data(stride: int = 8):
    times, xs, ys, vxs, vys = syn.integrate_projectile(syn.ProjectileParams())
    sl = slice(None, None, stride)
    return (np.array(times[sl]), np.array(xs[sl]), np.array(ys[sl]))


def test_grammar_2d_drag_term_matches_numpy() -> None:
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 0.0])
    vx = np.array([3.0, -1.0])
    vy = np.array([4.0, 2.0])
    got = expr.compile_term_2d("sqrt(vx**2+vy**2)*vx")(x, y, vx, vy)
    assert np.allclose(got, np.hypot(vx, vy) * vx)
    assert expr.validate_terms([P.DRAG_TERM_X], expr.ALLOWED_NAMES_2D)
    with pytest.raises(ValueError, match="unknown name"):
        expr.validate_term("q", expr.ALLOWED_NAMES_2D)
    with pytest.raises(ValueError, match="unknown name"):
        expr.validate_term("vx", expr.ALLOWED_NAMES)  # not in the 1-DOF set


def test_simulate_projectile_matches_parabola() -> None:
    t = np.linspace(0.0, 1.0, 31)
    x, y, vx, vy = P.simulate_projectile(
        ["1"], ["1"], np.array([0.0]), np.array([-9.81]), t, (0.0, 0.0, 5.0, 8.0))
    assert np.allclose(x, 5.0 * t, atol=1e-8)
    assert np.allclose(y, 8.0 * t - 4.905 * t**2, atol=1e-8)
    assert np.allclose(vy, 8.0 - 9.81 * t, atol=1e-8)


def test_synthetic_projectile_flies_and_lands() -> None:
    times, xs, ys, vxs, vys = syn.integrate_projectile(syn.ProjectileParams())
    assert np.all(np.diff(xs) > 0)          # always moving forward
    assert max(ys) > 1.0                    # it actually flies
    assert ys[-1] <= 0.0 < ys[-2]           # ends at launch height


@pytest.mark.slow
def test_fit_projectile_recovers_truth_and_beats_baseline() -> None:
    t, x, y = projectile_data()
    cut = int(len(t) * 0.6)
    true_model = P.score_projectile([P.DRAG_TERM_X], ["1", P.DRAG_TERM_Y],
                                    t[:cut], x[:cut], y[:cut],
                                    t[cut:], x[cut:], y[cut:])
    assert true_model.ok
    assert true_model.coeffs_x[0] == pytest.approx(-0.06, rel=0.1)
    assert true_model.coeffs_y[0] == pytest.approx(-9.81, rel=0.05)
    assert true_model.coeffs_y[1] == pytest.approx(-0.06, rel=0.1)
    assert true_model.held_nrmse < 1e-4
    baseline = P.score_projectile(P.PROJECTILE_BASELINE_X,
                                  P.PROJECTILE_BASELINE_Y,
                                  t[:cut], x[:cut], y[:cut],
                                  t[cut:], x[cut:], y[cut:])
    assert baseline.ok and baseline.held_nrmse > true_model.held_nrmse


def test_score_projectile_failure_is_data() -> None:
    t, x, y = projectile_data()
    cut = int(len(t) * 0.6)
    bad = P.score_projectile(["q"], ["1"], t[:cut], x[:cut], y[:cut],
                             t[cut:], x[cut:], y[cut:])
    assert not bad.ok and bad.error
    empty = P.score_projectile(["1"], ["1"], t[:cut], x[:cut], y[:cut],
                               t[:0], x[:0], y[:0])
    assert not empty.ok and "held-out" in (empty.error or "")


@pytest.mark.slow
def test_projectile_tools_and_scripted_loop_find_drag(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json
    from types import SimpleNamespace

    from pixelphys import scientist as S

    t, x, y = projectile_data()
    cut = int(len(t) * 0.6)
    tools = S.ProjectileTools(t[:cut], x[:cut], y[:cut],
                              t[cut:], x[cut:], y[cut:], tmp_path)
    assert tools.baseline.ok

    def call(name: str, arguments: dict) -> SimpleNamespace:
        return SimpleNamespace(
            id="c1", function=SimpleNamespace(
                name=name, arguments=json.dumps(arguments)))

    def response(*calls: SimpleNamespace) -> SimpleNamespace:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=None, tool_calls=list(calls)))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=100))

    rounds = [
        response(call("fit_model", {
            "terms_x": [P.DRAG_TERM_X],
            "terms_y": ["1", P.DRAG_TERM_Y],
            "rationale": "quadratic drag coupled through the speed"})),
        response(call("submit_model", {
            "model_id": "model-1",
            "explanation": "coupled quadratic drag on both axes"})),
    ]
    result = S.run_loop(tools, lambda messages, schemas: rounds.pop(0))
    assert result.submitted
    assert result.axis_terms == {"x": [P.DRAG_TERM_X],
                                 "y": ["1", P.DRAG_TERM_Y]}
    assert result.axis_coeffs is not None
    assert result.axis_coeffs["y"][0] == pytest.approx(-9.81, rel=0.05)


@pytest.mark.slow
def test_projectile_identity_truth_scores_near_zero() -> None:
    """Extension gate B1: scoring the true family must return ~zero held error."""
    t, x, y = projectile_data()
    cut = int(len(t) * 0.6)
    scored = P.score_projectile([P.DRAG_TERM_X], ["1", P.DRAG_TERM_Y],
                                t[:cut], x[:cut], y[:cut],
                                t[cut:], x[cut:], y[cut:])
    assert scored.ok and scored.held_nrmse < 1e-6


@pytest.mark.slow
def test_projectile_noisy_recovery_within_tolerance() -> None:
    """Extension gate B2: at 0.5 px noise, g within 5% and k within 25%."""
    rng = np.random.default_rng(7)
    t, x, y = projectile_data()
    sigma_m = 0.5 / syn.PX_PER_M_2D
    xn = x + rng.normal(0.0, sigma_m, size=x.shape)
    yn = y + rng.normal(0.0, sigma_m, size=y.shape)
    cut = int(len(t) * 0.6)
    scored = P.score_projectile([P.DRAG_TERM_X], ["1", P.DRAG_TERM_Y],
                                t[:cut], xn[:cut], yn[:cut],
                                t[cut:], xn[cut:], yn[cut:])
    assert scored.ok
    assert abs(-scored.coeffs_y[0] - 9.81) / 9.81 < 0.05
    assert abs(-scored.coeffs_x[0] - 0.06) / 0.06 < 0.25
    assert abs(-scored.coeffs_y[1] - 0.06) / 0.06 < 0.25
