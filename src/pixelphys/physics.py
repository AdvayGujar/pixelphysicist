"""M8: simulate, weak-form init, shooting fit, NRMSE/BIC. (Day 3.)

Model: ``q'' = sum c_i phi_i(q, v)`` over a whitelisted term library.
Fitting is two-stage: a weak (integral) form gives a linear least-squares
starting point without differentiating noisy data, then a shooting
refinement integrates the ODE and fits coefficients + initial conditions
to the observed positions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .expr import basis_matrix, compile_term, validate_terms
from .trajectory import smooth_position

try:
    from numpy import trapezoid as _trapz
except ImportError:  # numpy < 2.0
    from numpy import trapz as _trapz  # type: ignore[no-redef]

# NOTE (Day 3): sin/cos and q² were cut from the baseline. With q in pixels
# (no scale object in frame), q spans hundreds of radians, so sin(q)/cos(q)
# oscillate ~100x across the window: they fit noise, not physics, and the
# stiffness hangs the shooter (>10 min on the 267-pt pendulum). q² added a
# flat degenerate valley on oscillatory data (same hang). The plan blesses
# shrinking the library. Kept v² deliberately: the UNSIGNED square is the
# distractor that cannot express bidirectional drag -- the story needs the
# baseline to have it and still fail. Hero terms (|v|v, sign(v)) stay excluded.
BASELINE_TERMS = ["1", "q", "v", "v**2"]
COEFF_MARGIN = 50.0
IC_FRACTION = 0.10
WEAK_FUNCTIONS = 6
MIN_WEAK_POINTS = 8


@dataclass
class FitResult:
    terms: list[str]
    coeffs: np.ndarray
    q0: float
    v0: float


def estimate_velocity(t: np.ndarray, q: np.ndarray) -> np.ndarray:
    """First derivative of the smoothed position; no second differences."""
    t_arr = np.asarray(t, dtype=float)
    if len(t_arr) < 2:
        raise ValueError(f"need >= 2 samples for velocity, got {len(t_arr)}")
    return np.gradient(smooth_position(np.asarray(q, dtype=float)), t_arr)


def accel_scale(t: np.ndarray, q: np.ndarray) -> float:
    """Characteristic acceleration from the smoothed second derivative.

    Window geometry (span/duration²) underestimates oscillatory motion by
    ~(2π·cycles)² -- on the pendulum it clipped the true stiffness
    (bound ±5.2 vs truth -11.6) and pinned the fit at the bound. The
    data-driven peak is generous when noisy, which is the safe direction
    for a ceiling.
    """
    t_arr = np.asarray(t, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    second = np.gradient(np.gradient(smooth_position(q_arr), t_arr), t_arr)
    return max(float(np.max(np.abs(second))), 1e-9)


def coeff_bounds(terms: list[str], t: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-term coefficient bounds scaled to the data's units.

    A fixed bound (e.g. ±1e3) assumes SI units, but the pipeline runs in
    pixels when no scale is known, where gravity is ~1e4 px/s². Each term's
    ceiling is ``COEFF_MARGIN`` times the characteristic acceleration divided
    by that term's peak magnitude, so one term alone at its bound explains
    50× the observed motion -- generous, but still finite.
    """
    clean = validate_terms(list(terms))
    t_arr = np.asarray(t, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    scale = accel_scale(t_arr, q_arr)
    velocity = estimate_velocity(t_arr, q_arr)
    basis = basis_matrix(clean, q_arr, velocity)
    peaks = np.max(np.abs(basis), axis=0)
    peaks = np.maximum(peaks, 1e-300)
    ceiling = COEFF_MARGIN * scale / peaks
    return -ceiling, ceiling


def weak_init(terms: list[str], t: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Linear least-squares coefficients via Gaussian test functions.

    Integrates by parts twice: ``int w'' q = sum c_i int w phi_i``,
    so only a (smoothed) velocity estimate is needed, never ``q''``.
    """
    clean = validate_terms(list(terms))
    t_arr = np.asarray(t, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    n = len(t_arr)
    if n < max(len(clean), MIN_WEAK_POINTS):
        raise ValueError(f"need >= {max(len(clean), MIN_WEAK_POINTS)} points, got {n}")
    velocity = estimate_velocity(t_arr, q_arr)
    basis = np.column_stack([compile_term(s)(q_arr, velocity) for s in clean])
    span = float(t_arr[-1] - t_arr[0])
    if span <= 0:
        raise ValueError("time grid must be strictly increasing")
    centers = np.linspace(t_arr[0] + 0.15 * span, t_arr[-1] - 0.15 * span, WEAK_FUNCTIONS)
    sigma = span / WEAK_FUNCTIONS / 2.0
    rows = []
    targets = []
    for center in centers:
        shifted = (t_arr - center) / sigma
        weights = np.exp(-0.5 * shifted**2)
        curvatures = weights * (shifted**2 - 1.0) / sigma**2
        rows.append(_trapz(weights[:, None] * basis, t_arr, axis=0))
        targets.append(float(_trapz(curvatures * q_arr, t_arr)))
    coeffs, _, _, _ = np.linalg.lstsq(np.stack(rows), np.array(targets), rcond=None)
    lower, upper = coeff_bounds(clean, t_arr, q_arr)
    return np.clip(coeffs, lower, upper)


def simulate(terms: list[str], coeffs: np.ndarray,
             t: np.ndarray, state0: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Integrate ``q'' = sum c_i phi_i`` from ``(q0, v0)``; ``(q, v)`` at ``t``.

    DOP853 by default; terms with ``sign`` switch to BDF first because the
    discontinuity is stiff. The other method is the fallback either way.
    """
    clean = validate_terms(list(terms))
    gains = np.asarray(coeffs, dtype=float)
    if gains.shape != (len(clean),):
        raise ValueError(f"need {len(clean)} coefficients, got {gains.shape}")
    basis = [compile_term(s) for s in clean]
    t_arr = np.asarray(t, dtype=float)

    def rhs(_time: float, state: list[float]) -> list[float]:
        pos, vel = float(state[0]), float(state[1])
        total = sum(g * f(np.asarray(pos), np.asarray(vel)) for g, f in zip(gains, basis))
        return [vel, float(np.asarray(total))]

    stiff_first = any("sign" in s for s in clean)
    methods = ["BDF", "DOP853"] if stiff_first else ["DOP853", "BDF"]
    for method in methods:
        sol = solve_ivp(rhs, (t_arr[0], t_arr[-1]), list(state0), method=method,
                        t_eval=t_arr, rtol=1e-9, atol=1e-12)
        if sol.success and np.all(np.isfinite(sol.y)):
            return np.asarray(sol.y[0]), np.asarray(sol.y[1])
    raise RuntimeError(f"integration failed for terms {clean}")


def _ic_bounds(t: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    q_arr = np.asarray(q, dtype=float)
    t_arr = np.asarray(t, dtype=float)
    span_q = max(float(q_arr.max() - q_arr.min()), 1e-9)
    step = float(np.median(np.diff(t_arr)))
    if step <= 0:
        raise ValueError("time grid must be strictly increasing")
    vel_scale = span_q / step
    q0_guess = float(q_arr[0])
    v0_guess = float((q_arr[1] - q_arr[0]) / step)
    q_spread = IC_FRACTION * span_q
    v_spread = IC_FRACTION * vel_scale
    return (np.array([q0_guess - q_spread, v0_guess - v_spread]),
            np.array([q0_guess + q_spread, v0_guess + v_spread]),
            q0_guess, v0_guess)


def fit_terms(terms: list[str], t: np.ndarray, q: np.ndarray) -> FitResult:
    """Weak-form start + shooting refinement over coefficients and ``(q0, v0)``.

    Three deterministic starts (weak init, zeros, shrunk weak init); the
    lowest-cost converged run wins.
    """
    clean = validate_terms(list(terms))
    t_arr = np.asarray(t, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    n_terms = len(clean)
    try:
        weak = weak_init(clean, t_arr, q_arr)
    except ValueError:
        weak = np.zeros(n_terms)
    ic_lower, ic_upper, q0_guess, v0_guess = _ic_bounds(t_arr, q_arr)
    coef_lower, coef_upper = coeff_bounds(clean, t_arr, q_arr)
    lower = np.concatenate([coef_lower, ic_lower])
    upper = np.concatenate([coef_upper, ic_upper])
    starts = [
        np.concatenate([weak, [q0_guess, v0_guess]]),
        np.concatenate([np.zeros(n_terms), [q0_guess, v0_guess]]),
        np.concatenate([0.5 * weak, [q0_guess, v0_guess]]),
    ]

    def residual(params: np.ndarray) -> np.ndarray:
        try:
            pred, _ = simulate(clean, params[:n_terms], t_arr,
                               (float(params[n_terms]), float(params[n_terms + 1])))
        except RuntimeError:
            return np.full_like(q_arr, 1e6)
        return pred - q_arr

    best: FitResult | None = None
    best_cost = float("inf")
    for start in starts:
        result = least_squares(residual, start, bounds=(lower, upper))
        if result.cost < best_cost and np.all(np.isfinite(result.x)):
            best_cost = result.cost
            best = FitResult(terms=clean, coeffs=result.x[:n_terms],
                             q0=float(result.x[n_terms]), v0=float(result.x[n_terms + 1]))
    if best is None:
        raise RuntimeError(f"shooting fit failed for terms {clean}")
    return best


def nrmse(pred: np.ndarray, true: np.ndarray) -> float:
    """RMSE normalized by the held-out amplitude (max - min)."""
    pred_arr = np.asarray(pred, dtype=float)
    true_arr = np.asarray(true, dtype=float)
    rmse = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
    amplitude = float(true_arr.max() - true_arr.min())
    if amplitude <= 1e-12:
        return 0.0 if rmse <= 1e-12 else float("inf")
    return rmse / amplitude


def bic(n: int, rss: float, k: int) -> float:
    """Bayesian information criterion for model comparison."""
    if n <= 0:
        raise ValueError(f"need n > 0, got {n}")
    return n * np.log(max(float(rss), 1e-300) / n) + k * np.log(n)
