"""M8: simulate, weak-form init, shooting fit, NRMSE/BIC. (Day 3.)

Model: ``q'' = sum c_i phi_i(q, v)`` over a whitelisted term library.
Fitting is two-stage: a weak (integral) form gives a linear least-squares
starting point without differentiating noisy data, then a shooting
refinement integrates the ODE and fits coefficients + initial conditions
to the observed positions.
"""
from __future__ import annotations

import ast
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

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
WEAK_FUNCTIONS = 8  # keep >= MAX_TERMS: every valid library gets a weak start
MIN_WEAK_POINTS = 8
MAX_NFEV_PER_VAR = 30
SHOOT_FTOL = 1e-6
SHOOT_XTOL = 1e-6
SHOOT_GTOL = 1e-6
DEFAULT_FIT_BUDGET_S = 60.0
BLOWUP_RESIDUAL = 1e6
# One ODE integration must never take this long on our windows (normal
# sims are ~ms); anything slower is a stiff-solver stall inside solve_ivp,
# which the per-fit deadline cannot interrupt. Abort the sim, not the fit.
SIM_TIMEOUT_S = 5.0


@contextmanager
def _sim_time_limit(seconds: float):
    """Abort a hanging solve_ivp via SIGALRM (main thread only)."""
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        yield
        return
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(_signum, _frame):
        raise TimeoutError(f"ODE integration exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


@dataclass
class FitResult:
    terms: list[str]
    coeffs: np.ndarray
    q0: float
    v0: float
    cost: float = field(default=float("inf"))
    start_index: int = field(default=-1)


@dataclass
class ScoredFit:
    """Outcome of fitting on a fit window and scoring on held-out data.

    ``ok`` is False (with ``held_nrmse``/``bic`` set to inf and ``error``
    naming the cause) whenever fitting fails, the rollout explodes, or the
    budget runs out -- never an exception past this boundary, so the Day-4
    scientist loop can treat "too slow / exploded" as data, not a crash.
    """

    ok: bool
    terms: list[str]
    coeffs: np.ndarray
    q0: float
    v0: float
    train_rmse: float
    held_nrmse: float
    bic: float
    n_scored: int
    cost: float
    start_index: int
    error: str | None = None


class _BudgetExceeded(RuntimeError):
    """Internal: per-fit wall-clock budget consumed mid-optimization."""


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

    Integrates by parts twice: ``int w'' q = sum c_i int w phi_i``, so only
    a (smoothed) velocity estimate is needed, never ``q''``. Gaussians are
    not compactly supported, so the identity's boundary terms
    ``[w'q - wv]`` at the window ends are subtracted explicitly -- without
    them the drag coefficient comes out ~50% off even on noiseless data.
    """
    clean = validate_terms(list(terms))
    t_arr = np.asarray(t, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    n = len(t_arr)
    if n < max(len(clean), MIN_WEAK_POINTS):
        raise ValueError(f"need >= {max(len(clean), MIN_WEAK_POINTS)} points, got {n}")
    span = float(t_arr[-1] - t_arr[0])
    if span <= 0:
        raise ValueError("time grid must be strictly increasing")
    velocity = estimate_velocity(t_arr, q_arr)
    basis = basis_matrix(clean, q_arr, velocity)
    if not np.all(np.isfinite(basis)):
        raise ValueError("non-finite basis values (e.g. division by zero in a term)")
    step = float(np.median(np.diff(t_arr)))
    if step <= 0:
        raise ValueError("time grid must be strictly increasing")
    # Wide enough to quadrature: sigma below ~3 grid steps makes the
    # trapezoidal integrals (and hence the start) garbage even on clean data.
    sigma = max(span / 4.0, 4.0 * step)
    centers = np.linspace(t_arr[0] + 0.15 * span, t_arr[-1] - 0.15 * span, WEAK_FUNCTIONS)
    rows = []
    targets = []
    for center in centers:
        shifted = (t_arr - center) / sigma
        weights = np.exp(-0.5 * shifted**2)
        slopes = -shifted / sigma * weights
        curvatures = weights * (shifted**2 - 1.0) / sigma**2
        rows.append(_trapz(weights[:, None] * basis, t_arr, axis=0))
        edge = ((slopes[-1] * q_arr[-1] - weights[-1] * velocity[-1])
                - (slopes[0] * q_arr[0] - weights[0] * velocity[0]))
        targets.append(float(_trapz(curvatures * q_arr, t_arr)) - edge)
    coeffs, _, _, _ = np.linalg.lstsq(np.stack(rows), np.array(targets), rcond=None)
    lower, upper = coeff_bounds(clean, t_arr, q_arr)
    return np.clip(coeffs, lower, upper)


def _uses_sign(terms: list[str]) -> bool:
    """True when a term calls ``sign(...)`` (stiff discontinuity)."""
    for term in terms:
        for node in ast.walk(ast.parse(term, mode="eval")):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "sign"):
                return True
    return False


def simulate(terms: list[str], coeffs: np.ndarray,
             t: np.ndarray, state0: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Integrate ``q'' = sum c_i phi_i`` from ``(q0, v0)``; ``(q, v)`` at ``t``.

    DOP853 by default; terms calling ``sign`` use BDF only because the
    discontinuity is stiff -- falling back to explicit DOP853 after a BDF
    failure hangs it in microscopic rk_steps on the discontinuity
    (Day-5 N pilot: one solve_ivp stuck >90 s, fit never returned).
    A BDF failure therefore raises at once; the caller treats it as data.
    """
    clean = validate_terms(list(terms))
    gains = np.asarray(coeffs, dtype=float)
    if gains.shape != (len(clean),):
        raise ValueError(f"need {len(clean)} coefficients, got {gains.shape}")
    if not np.all(np.isfinite(gains)):
        raise ValueError(f"coefficients must be finite, got {gains}")
    t_arr = np.asarray(t, dtype=float)
    if t_arr.ndim != 1 or len(t_arr) < 2:
        raise ValueError(f"need a 1-D grid of >= 2 points, got shape {t_arr.shape}")
    if not bool(np.all(np.diff(t_arr) > 0)):
        raise ValueError("time grid must be strictly increasing")
    basis = [compile_term(s) for s in clean]

    def rhs(_time: float, state: list[float]) -> list[float]:
        pos, vel = float(state[0]), float(state[1])
        total = sum(g * f(np.asarray(pos), np.asarray(vel)) for g, f in zip(gains, basis))
        return [vel, float(np.asarray(total))]

    stiff_first = _uses_sign(clean)
    # Stiff (sign) terms: BDF only, looser tolerances (discontinuity caps
    # achievable accuracy; 1e-6/1e-9 still far below NRMSE needs).
    # Smooth terms: DOP853 first, BDF fallback (unchanged).
    methods = ["BDF"] if stiff_first else ["DOP853", "BDF"]
    for method in methods:
        rtol, atol = (1e-6, 1e-9) if (stiff_first and method == "BDF") else (1e-9, 1e-12)
        try:
            with _sim_time_limit(SIM_TIMEOUT_S):
                sol = solve_ivp(rhs, (t_arr[0], t_arr[-1]), list(state0), method=method,
                                t_eval=t_arr, rtol=rtol, atol=atol)
        except TimeoutError as exc:
            raise RuntimeError(f"integration timed out for terms {clean}: {exc}") from None
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
    v0_guess = float(estimate_velocity(t_arr, q_arr)[0])
    q_spread = IC_FRACTION * span_q
    v_spread = IC_FRACTION * vel_scale
    return (np.array([q0_guess - q_spread, v0_guess - v_spread]),
            np.array([q0_guess + q_spread, v0_guess + v_spread]),
            q0_guess, v0_guess)


def fit_terms(terms: list[str], t: np.ndarray, q: np.ndarray,
              budget_s: float | None = None) -> FitResult:
    """Weak-form start + shooting refinement over coefficients and ``(q0, v0)``.

    Three deterministic starts (weak init, zeros, shrunk weak init); the
    lowest-cost converged run wins. Refuses underdetermined fits outright
    instead of returning silent garbage. ``budget_s`` caps wall-clock time;
    expiry raises ``RuntimeError`` so slowness surfaces instead of hanging.
    """
    clean = validate_terms(list(terms))
    t_arr = np.asarray(t, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    n_terms = len(clean)
    if len(t_arr) < n_terms + 2:
        raise ValueError(f"need >= {n_terms + 2} points for {n_terms} terms + ICs, "
                         f"got {len(t_arr)}")
    if not np.all(np.isfinite(basis_matrix(clean, q_arr, estimate_velocity(t_arr, q_arr)))):
        raise ValueError(f"non-finite basis values for terms {clean}")
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
    deadline = None if budget_s is None else time.monotonic() + budget_s

    def residual(params: np.ndarray) -> np.ndarray:
        if deadline is not None and time.monotonic() > deadline:
            raise _BudgetExceeded(f"fit budget of {budget_s}s exceeded")
        try:
            pred, _ = simulate(clean, params[:n_terms], t_arr,
                               (float(params[n_terms]), float(params[n_terms + 1])))
        except RuntimeError:
            return np.full_like(q_arr, BLOWUP_RESIDUAL)
        return pred - q_arr

    best: FitResult | None = None
    best_cost = float("inf")
    max_nfev = MAX_NFEV_PER_VAR * (n_terms + 2)
    try:
        for index, start in enumerate(starts):
            result = least_squares(residual, start, bounds=(lower, upper),
                                   max_nfev=max_nfev, ftol=SHOOT_FTOL,
                                   xtol=SHOOT_XTOL, gtol=SHOOT_GTOL)
            if (result.success and np.all(np.isfinite(result.x))
                    and np.isfinite(result.cost) and result.cost < best_cost):
                best_cost = result.cost
                best = FitResult(terms=clean, coeffs=result.x[:n_terms],
                                 q0=float(result.x[n_terms]),
                                 v0=float(result.x[n_terms + 1]),
                                 cost=float(result.cost), start_index=index)
    except _BudgetExceeded as exc:
        raise RuntimeError(str(exc)) from None
    if best is None:
        raise RuntimeError(f"shooting fit failed for terms {clean}")
    return best


def score_candidate(terms: list[str], t_fit: np.ndarray, q_fit: np.ndarray,
                    t_held: np.ndarray, q_held: np.ndarray,
                    horizon_s: float = 2.0,
                    budget_s: float = DEFAULT_FIT_BUDGET_S) -> ScoredFit:
    """Fit on the fit window, score the continuation on held-out data.

    The scored horizon is min(``horizon_s``, held data). BIC counts
    coefficients plus the 2 fitted initial conditions. Never raises for
    model/data problems: failure (bad fit, exploding rollout, exhausted
    budget) returns ``ok=False`` with ``held_nrmse``/``bic`` set to inf and
    ``error`` naming the cause, so the scientist loop treats it as data.
    """
    t_fit_arr = np.asarray(t_fit, dtype=float)
    q_fit_arr = np.asarray(q_fit, dtype=float)
    t_held_arr = np.asarray(t_held, dtype=float)
    q_held_arr = np.asarray(q_held, dtype=float)

    def failed(clean: list[str], error: str) -> ScoredFit:
        return ScoredFit(ok=False, terms=clean,
                         coeffs=np.full(len(clean), np.nan),
                         q0=float("nan"), v0=float("nan"),
                         train_rmse=float("inf"), held_nrmse=float("inf"),
                         bic=float("inf"), n_scored=0,
                         cost=float("inf"), start_index=-1, error=error)

    try:
        clean = validate_terms(list(terms))
        if len(t_held_arr) < 1:
            return failed(clean, "no held-out points to score")
        fit = fit_terms(clean, t_fit_arr, q_fit_arr, budget_s=budget_s)
        horizon_end = float(t_held_arr[0]) + horizon_s
        scored = min(max(int(np.searchsorted(t_held_arr, horizon_end)), 1),
                     len(t_held_arr))
        grid = np.concatenate([t_fit_arr, t_held_arr])
        full, _ = simulate(fit.terms, fit.coeffs, grid, (fit.q0, fit.v0))
        held_pred = full[len(t_fit_arr):len(t_fit_arr) + scored]
        if not np.all(np.isfinite(held_pred)):
            return failed(clean, "rollout went non-finite on held-out data")
        train_pred, _ = simulate(fit.terms, fit.coeffs, t_fit_arr, (fit.q0, fit.v0))
        residuals = train_pred - q_fit_arr
        rss = float(np.sum(residuals**2))
        return ScoredFit(ok=True, terms=clean, coeffs=fit.coeffs,
                         q0=fit.q0, v0=fit.v0,
                         train_rmse=float(np.sqrt(np.mean(residuals**2))),
                         held_nrmse=nrmse(held_pred, q_held_arr[:scored]),
                         bic=bic(len(t_fit_arr), rss, len(clean) + 2),
                         n_scored=scored, cost=fit.cost,
                         start_index=fit.start_index)
    except (ValueError, RuntimeError, TypeError) as exc:
        clean = locals().get("clean", list(terms) if isinstance(terms, list) else [])
        return failed(clean, f"{type(exc).__name__}: {exc}")


def rollout_score(terms: list[str], coeffs: np.ndarray,
                  t: np.ndarray, q: np.ndarray,
                  horizon_s: float = 2.0, ic_points: int = 3) -> tuple[float, int]:
    """Score fixed coefficients on a new clip from refit initial conditions.

    Cross-take generalization: coefficients transfer, initial conditions do
    not, so the ICs are refit on the clip's first ``ic_points`` samples and
    only the remaining points are scored (up to ``horizon_s``). Refitting
    beats anchoring on an edge-smoothed gradient, whose error otherwise
    dominates the ranking. Returns ``(nrmse, n_scored)``; ``inf`` on failure.
    """
    t_arr = np.asarray(t, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    try:
        clean = validate_terms(list(terms))
        gains = np.asarray(coeffs, dtype=float)
        if len(t_arr) < 2 or gains.shape != (len(clean),):
            return float("inf"), 0
        n_ic = min(ic_points, len(t_arr) - 1)

        def ic_residual(state: np.ndarray) -> np.ndarray:
            pred, _ = simulate(clean, gains, t_arr[:n_ic],
                               (float(state[0]), float(state[1])))
            return pred - q_arr[:n_ic]

        anchor = (float(q_arr[0]), float(estimate_velocity(t_arr, q_arr)[0]))
        ic = least_squares(ic_residual, np.array(anchor)).x
        # The refit state lives at t_arr[0]: integrate the FULL grid from it
        # and slice the scored tail. Starting the integration at the tail
        # with the t[0] state silently shifts the trajectory by ic_points
        # frames and corrupts every cross-take number.
        full, _ = simulate(clean, gains, t_arr, (float(ic[0]), float(ic[1])))
        tail_q = q_arr[n_ic:]
        scored = min(max(int(np.searchsorted(t_arr[n_ic:], t_arr[n_ic] + horizon_s)), 1),
                     len(tail_q))
        pred = full[n_ic:n_ic + scored]
        if not np.all(np.isfinite(pred)):
            return float("inf"), scored
        return nrmse(pred, tail_q[:scored]), scored
    except (ValueError, RuntimeError):
        return float("inf"), 0


# --- 2-DOF projectile mode (Section 4.5b): state [x, y, vx, vy] -----------
# Terms are per-axis expressions in (x, y, vx, vy); the drag coupling enters
# through terms like sqrt(vx**2 + vy**2) * vx, which mix both velocities.

PROJECTILE_BASELINE_X = ["1", "x", "vx", "vx**2"]
PROJECTILE_BASELINE_Y = ["1", "y", "vy", "vy**2"]
DRAG_TERM_X = "sqrt(vx**2+vy**2)*vx"
DRAG_TERM_Y = "sqrt(vx**2+vy**2)*vy"


@dataclass
class FitResult2D:
    terms_x: list[str]
    terms_y: list[str]
    coeffs_x: np.ndarray
    coeffs_y: np.ndarray
    state0: tuple[float, float, float, float]
    cost: float = field(default=float("inf"))
    start_index: int = field(default=-1)


@dataclass
class ScoredFit2D:
    """score_projectile outcome; failure is data (ok=False), never a raise."""

    ok: bool
    terms_x: list[str]
    terms_y: list[str]
    coeffs_x: np.ndarray
    coeffs_y: np.ndarray
    state0: tuple[float, float, float, float]
    train_rmse: float
    held_nrmse: float
    bic: float
    n_scored: int
    cost: float
    start_index: int
    error: str | None = None


def simulate_projectile(terms_x: list[str], terms_y: list[str],
                        coeffs_x: np.ndarray, coeffs_y: np.ndarray,
                        t: np.ndarray,
                        state0: tuple[float, float, float, float],
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the coupled system; returns ``(x, y, vx, vy)`` at ``t``."""
    from .expr import compile_term_2d, validate_terms as _vt
    from .expr import ALLOWED_NAMES_2D

    clean_x = _vt(list(terms_x), ALLOWED_NAMES_2D)
    clean_y = _vt(list(terms_y), ALLOWED_NAMES_2D)
    gains_x = np.asarray(coeffs_x, dtype=float)
    gains_y = np.asarray(coeffs_y, dtype=float)
    if gains_x.shape != (len(clean_x),) or gains_y.shape != (len(clean_y),):
        raise ValueError("coefficient counts must match term counts")
    if not (np.all(np.isfinite(gains_x)) and np.all(np.isfinite(gains_y))):
        raise ValueError("coefficients must be finite")
    t_arr = np.asarray(t, dtype=float)
    if t_arr.ndim != 1 or len(t_arr) < 2:
        raise ValueError(f"need a 1-D grid of >= 2 points, got shape {t_arr.shape}")
    if not bool(np.all(np.diff(t_arr) > 0)):
        raise ValueError("time grid must be strictly increasing")
    basis_x = [compile_term_2d(s) for s in clean_x]
    basis_y = [compile_term_2d(s) for s in clean_y]

    def rhs(_time: float, state: list[float]) -> list[float]:
        x, y, vx, vy = (float(s) for s in state)
        point = (np.asarray(x), np.asarray(y), np.asarray(vx), np.asarray(vy))
        ax = sum(g * f(*point) for g, f in zip(gains_x, basis_x))
        ay = sum(g * f(*point) for g, f in zip(gains_y, basis_y))
        return [vx, vy, float(np.asarray(ax)), float(np.asarray(ay))]

    stiff_first = _uses_sign(clean_x + clean_y)
    # Same stiff-only-BDF rule as the 1-DOF simulate: no DOP853 fallback
    # after a BDF failure on discontinuous terms (hangs in rk_step).
    methods = ["BDF"] if stiff_first else ["DOP853", "BDF"]
    for method in methods:
        rtol, atol = (1e-6, 1e-9) if (stiff_first and method == "BDF") else (1e-9, 1e-12)
        try:
            with _sim_time_limit(SIM_TIMEOUT_S):
                sol = solve_ivp(rhs, (t_arr[0], t_arr[-1]), list(state0), method=method,
                                t_eval=t_arr, rtol=rtol, atol=atol)
        except TimeoutError as exc:
            raise RuntimeError(
                f"integration timed out for terms {clean_x} / {clean_y}: {exc}") from None
        if sol.success and np.all(np.isfinite(sol.y)):
            return (np.asarray(sol.y[0]), np.asarray(sol.y[1]),
                    np.asarray(sol.y[2]), np.asarray(sol.y[3]))
    raise RuntimeError(f"integration failed for terms {clean_x} / {clean_y}")


def _weak_axis_2d(terms: list[str], t: np.ndarray, pos: np.ndarray,
                  pos_vel: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """One axis of the weak form over a precomputed 2-D basis matrix."""
    span = float(t[-1] - t[0])
    step = float(np.median(np.diff(t)))
    sigma = max(span / 4.0, 4.0 * step)
    centers = np.linspace(t[0] + 0.15 * span, t[-1] - 0.15 * span, WEAK_FUNCTIONS)
    rows, targets = [], []
    for center in centers:
        shifted = (t - center) / sigma
        weights = np.exp(-0.5 * shifted**2)
        slopes = -shifted / sigma * weights
        curvatures = weights * (shifted**2 - 1.0) / sigma**2
        rows.append(_trapz(weights[:, None] * basis, t, axis=0))
        edge = ((slopes[-1] * pos[-1] - weights[-1] * pos_vel[-1])
                - (slopes[0] * pos[0] - weights[0] * pos_vel[0]))
        targets.append(float(_trapz(curvatures * pos, t)) - edge)
    coeffs, _, _, _ = np.linalg.lstsq(np.stack(rows), np.array(targets),
                                      rcond=None)
    return coeffs


def _projectile_setup(terms_x: list[str], terms_y: list[str],
                      t: np.ndarray, x: np.ndarray, y: np.ndarray,
                      ) -> tuple[list[str], list[str], np.ndarray, np.ndarray,
                                 np.ndarray, np.ndarray]:
    from .expr import ALLOWED_NAMES_2D, basis_matrix_2d, validate_terms as _vt

    clean_x = _vt(list(terms_x), ALLOWED_NAMES_2D)
    clean_y = _vt(list(terms_y), ALLOWED_NAMES_2D)
    vx = estimate_velocity(t, x)
    vy = estimate_velocity(t, y)
    basis_x = basis_matrix_2d(clean_x, x, y, vx, vy)
    basis_y = basis_matrix_2d(clean_y, x, y, vx, vy)
    if not (np.all(np.isfinite(basis_x)) and np.all(np.isfinite(basis_y))):
        raise ValueError("non-finite basis values for projectile terms")
    return clean_x, clean_y, vx, vy, basis_x, basis_y


def _axis_bounds_2d(scale: float, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    peaks = np.maximum(np.max(np.abs(basis), axis=0), 1e-300)
    ceiling = COEFF_MARGIN * scale / peaks
    return -ceiling, ceiling


def fit_projectile(terms_x: list[str], terms_y: list[str],
                   t: np.ndarray, x: np.ndarray, y: np.ndarray,
                   budget_s: float | None = None) -> FitResult2D:
    """Weak-form start + shooting over per-axis coefficients and the state."""
    t_arr = np.asarray(t, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    clean_x, clean_y, vx, vy, basis_x, basis_y = _projectile_setup(
        terms_x, terms_y, t_arr, x_arr, y_arr)
    nx, ny = len(clean_x), len(clean_y)
    if len(t_arr) < nx + ny + 4:
        raise ValueError(f"need >= {nx + ny + 4} points for {nx}+{ny} terms "
                         f"+ 4 state values, got {len(t_arr)}")
    try:
        weak_x = np.clip(_weak_axis_2d(clean_x, t_arr, x_arr, vx, basis_x),
                         *_axis_bounds_2d(accel_scale(t_arr, x_arr), basis_x))
        weak_y = np.clip(_weak_axis_2d(clean_y, t_arr, y_arr, vy, basis_y),
                         *_axis_bounds_2d(accel_scale(t_arr, y_arr), basis_y))
        weak = np.concatenate([weak_x, weak_y])
    except ValueError:
        weak = np.zeros(nx + ny)
    ix_low, ix_up, x0g, vx0g = _ic_bounds(t_arr, x_arr)
    iy_low, iy_up, y0g, vy0g = _ic_bounds(t_arr, y_arr)
    bx_low, bx_up = _axis_bounds_2d(accel_scale(t_arr, x_arr), basis_x)
    by_low, by_up = _axis_bounds_2d(accel_scale(t_arr, y_arr), basis_y)
    state_guess = np.array([x0g, y0g, vx0g, vy0g])
    lower = np.concatenate([bx_low, by_low,
                            [ix_low[0], iy_low[0], ix_low[1], iy_low[1]]])
    upper = np.concatenate([bx_up, by_up,
                            [ix_up[0], iy_up[0], ix_up[1], iy_up[1]]])
    starts = [np.concatenate([weak, state_guess]),
              np.concatenate([np.zeros(nx + ny), state_guess]),
              np.concatenate([0.5 * weak, state_guess])]
    deadline = None if budget_s is None else time.monotonic() + budget_s

    def residual(params: np.ndarray) -> np.ndarray:
        if deadline is not None and time.monotonic() > deadline:
            raise _BudgetExceeded(f"fit budget of {budget_s}s exceeded")
        state = (float(params[nx + ny]), float(params[nx + ny + 1]),
                 float(params[nx + ny + 2]), float(params[nx + ny + 3]))
        try:
            px, py, _, _ = simulate_projectile(
                clean_x, clean_y, params[:nx], params[nx:nx + ny],
                t_arr, state)
        except RuntimeError:
            return np.full(2 * len(t_arr), BLOWUP_RESIDUAL)
        return np.concatenate([px - x_arr, py - y_arr])

    best: FitResult2D | None = None
    best_cost = float("inf")
    max_nfev = MAX_NFEV_PER_VAR * (nx + ny + 4)
    try:
        for index, start in enumerate(starts):
            result = least_squares(residual, start, bounds=(lower, upper),
                                   max_nfev=max_nfev, ftol=SHOOT_FTOL,
                                   xtol=SHOOT_XTOL, gtol=SHOOT_GTOL)
            if (result.success and np.all(np.isfinite(result.x))
                    and np.isfinite(result.cost) and result.cost < best_cost):
                best_cost = result.cost
                state = tuple(float(v) for v in result.x[nx + ny:nx + ny + 4])
                best = FitResult2D(clean_x, clean_y,
                                   result.x[:nx], result.x[nx:nx + ny],
                                   state, float(result.cost), index)
    except _BudgetExceeded as exc:
        raise RuntimeError(str(exc)) from None
    if best is None:
        raise RuntimeError(f"projectile shooting fit failed for {clean_x} / {clean_y}")
    return best


def score_projectile(terms_x: list[str], terms_y: list[str],
                     t_fit: np.ndarray, x_fit: np.ndarray, y_fit: np.ndarray,
                     t_held: np.ndarray, x_held: np.ndarray, y_held: np.ndarray,
                     horizon_s: float = 2.0,
                     budget_s: float = DEFAULT_FIT_BUDGET_S) -> ScoredFit2D:
    """Fit on the fit window, score the continuation on held-out (x, y).

    Held NRMSE is the 2-D RMSE normalized by the held segment's amplitude
    (hypotenuse of the per-axis ranges). BIC counts coefficients plus the
    4 fitted initial-state values. Failures return ok=False with inf
    metrics -- the same contract as score_candidate.
    """
    def failed(cx: list[str], cy: list[str], error: str) -> ScoredFit2D:
        return ScoredFit2D(False, cx, cy, np.full(len(cx), np.nan),
                           np.full(len(cy), np.nan),
                           (float("nan"),) * 4, float("inf"), float("inf"),
                           float("inf"), 0, float("inf"), -1, error)

    clean_x: list[str] = []
    clean_y: list[str] = []
    try:
        from .expr import ALLOWED_NAMES_2D, validate_terms as _vt

        clean_x = _vt(list(terms_x), ALLOWED_NAMES_2D)
        clean_y = _vt(list(terms_y), ALLOWED_NAMES_2D)
        t_fit_a = np.asarray(t_fit, dtype=float)
        t_held_a = np.asarray(t_held, dtype=float)
        if len(t_held_a) < 1:
            return failed(clean_x, clean_y, "no held-out points to score")
        fit = fit_projectile(clean_x, clean_y, t_fit_a,
                             np.asarray(x_fit, dtype=float),
                             np.asarray(y_fit, dtype=float),
                             budget_s=budget_s)
        horizon_end = float(t_held_a[0]) + horizon_s
        scored = min(max(int(np.searchsorted(t_held_a, horizon_end)), 1),
                     len(t_held_a))
        grid = np.concatenate([t_fit_a, t_held_a])
        px, py, _, _ = simulate_projectile(fit.terms_x, fit.terms_y,
                                           fit.coeffs_x, fit.coeffs_y,
                                           grid, fit.state0)
        n_fit = len(t_fit_a)
        held_x, held_y = px[n_fit:n_fit + scored], py[n_fit:n_fit + scored]
        if not (np.all(np.isfinite(held_x)) and np.all(np.isfinite(held_y))):
            return failed(clean_x, clean_y, "rollout went non-finite on held-out data")
        train_x, train_y = px[:n_fit], py[:n_fit]
        rss = float(np.sum((train_x - np.asarray(x_fit, dtype=float))**2)
                    + np.sum((train_y - np.asarray(y_fit, dtype=float))**2))
        rmse_2d = float(np.sqrt(np.mean(
            (held_x - np.asarray(x_held, dtype=float)[:scored])**2
            + (held_y - np.asarray(y_held, dtype=float)[:scored])**2)))
        amplitude = float(np.hypot(
            np.ptp(np.asarray(x_held, dtype=float)[:scored]),
            np.ptp(np.asarray(y_held, dtype=float)[:scored])))
        held_nrmse = (rmse_2d / amplitude if amplitude > 1e-12
                      else (0.0 if rmse_2d <= 1e-12 else float("inf")))
        return ScoredFit2D(True, fit.terms_x, fit.terms_y, fit.coeffs_x,
                           fit.coeffs_y, fit.state0,
                           float(np.sqrt(rss / (2 * n_fit))), held_nrmse,
                           bic(n_fit, rss, len(fit.coeffs_x)
                               + len(fit.coeffs_y) + 4),
                           scored, fit.cost, fit.start_index)
    except (ValueError, RuntimeError, TypeError) as exc:
        return failed(clean_x, clean_y, f"{type(exc).__name__}: {exc}")


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
