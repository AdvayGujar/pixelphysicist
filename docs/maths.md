# Mathematics

How the fitter works, precisely but without assumed background. All
references are to `src/pixelphys/physics.py` unless noted.

## 1. The model class

1-DOF motion is modelled as acceleration = weighted sum of candidate terms:

```
q'' = c_1*phi_1(q,v) + c_2*phi_2(q,v) + ...
```

Each `phi` is a short string in the grammar of `src/pixelphys/expr.py`
(names `q, v`; numbers; `+ - * / **`; `sin cos tanh sign abs sqrt exp`;
max 8 terms, validated by AST whitelist, compiled once and cached).
Example: `["1", "abs(v)*v"]` means `q'' = c0 + c1*|v|*v`.

Projectile mode is the coupled 2-DOF version with state `[x, y, vx, vy]`
and per-axis term sets in `(x, y, vx, vy)`:

```
x'' = sum a_i*phi_i(x,y,vx,vy),   y'' = sum b_j*psi_j(x,y,vx,vy)
```

The drag coupling enters through terms like
`sqrt(vx**2+vy**2)*vx`, which mix both velocities and no 1-DOF grammar
can express.

## 2. Why not differentiate the data

Getting `q''` from tracked positions needs two numerical derivatives, and
each derivative amplifies high-frequency tracking noise (differencing
divides noise by `dt`; twice means roughly `1/dt**2`). On 30 fps video
that buries the physics. So the pipeline never forms `q''`: the weak form
needs only a smoothed velocity, and shooting compares positions directly.

## 3. The weak (integral) form: initialization only

Multiply the ODE by a smooth test function `w(t)` and integrate over the
fit window, then integrate by parts twice to move derivatives off the noisy
data onto the known `w`:

```
int w*q'' dt = [w*q' - w'*q] + int w''*q dt  =  sum c_i * int w*phi_i(q,v) dt
```

The bracket is the boundary term, evaluated with the smoothed velocity
estimate and subtracted explicitly -- without it the drag coefficient is
~50% off even on noiseless data. The integrals use the trapezoid rule over
Gaussian test functions (`WEAK_FUNCTIONS = 8` centers across the window).
This gives a linear least-squares problem `A*c = b` for the coefficients.

Test-function width lesson (from `docs/FAILURES.md`): sigma must satisfy

```
sigma = max(span/4, 4*dt)
```

Narrower than ~3 grid steps makes the quadrature garbage (weak drag
estimate 48% off on clean data, first misread as a boundary-term bug).
Solves to ~2%/11% on clean synthetic once wide enough.

Weak-form output is only a starting point for the shooter, never a
reported score.

## 4. Shooting refinement

`fit_terms` fits coefficients AND initial conditions `(q0, v0)` by
integrating the candidate ODE (`solve_ivp`) and minimizing position
residuals with `scipy.optimize.least_squares`:

- Three deterministic starts: weak-init coefficients, all zeros, halved
  weak-init (each with the smoothed-trajectory IC guess); lowest-cost
  converged run wins. Refuses underdetermined fits
  (`n_points >= n_terms + 2`) outright.
- Per-parameter `x_scale = max(|start|, 1)`: pixel-scale coefficients
  (~1e4) next to O(1) drag coefficients otherwise starve the
  finite-difference Jacobian.
- Coefficient bounds are data-driven per term (`coeff_bounds`):
  `ceiling = 50 * peak|q''| / peak|phi|`, where `peak|q''|` comes from the
  smoothed second derivative. Fixed bounds assume SI units and pin
  pixel-scale gravity (~1.3e4 px/s²). Initial conditions are bounded to
  ±10% of the window span around the smoothed guess (`IC_FRACTION`).
- Stiff terms (any term calling `sign`) integrate with BDF only at
  1e-6/1e-9; smooth terms use DOP853 first with BDF fallback. A BDF
  failure on `sign` raises at once (falling back to explicit DOP853 hung
  >90 s in microscopic steps on the pendulum).
- Fit budgets and the integration guard: the committed baseline gets 60 s
  (`DEFAULT_FIT_BUDGET_S`); in-loop candidates get 15 s with 5 s screens
  (single weak-init start, loose 1e-4 tolerances, advisory only). Every
  `solve_ivp` call is wrapped in a 5 s SIGALRM guard (`SIM_TIMEOUT_S`):
  a stall raises as integration failure, which becomes error-string data
  for Spark, never a hang.

## 5. Scoring: held-out NRMSE, BIC, and the submit guard

`score_candidate` fits on the first 60% and scores the continuation over
`min(2 s, clean tail)` (`split_by_gaps`/`trim_free_flight` pick the window):

```
NRMSE = RMSE(held_pred, held_true) / (max(q_held) - min(q_held))
BIC   = n*ln(RSS/n) + k*ln(n),   k = coefficients + 2 fitted ICs
```

(Projectile: 2-D RMSE normalized by the hypotenuse of the per-axis ranges;
`k` counts both axes' coefficients + 4 state values.)

Crucially, `held_pred` continues each model's OWN fitted trajectory from
its fitted `(q0, v0)` -- it never restarts from the true split state.
Restarting from an edge-smoothed velocity anchor injects enough error that
a perfect model scored 0.12 held-NRMSE; continuing its own trajectory
scores ~2e-9 (`docs/FAILURES.md`, 2026-10-28). Scoring the full tail is
also avoided: cycles of phase drift dominate even a 1%-accurate frequency,
hence the 2 s horizon cap.

Submit guard (`scientist.py`, `ScientistTools.submit_model`):
the baseline always passes; a candidate must beat it.

```
v1 (shipped):  held_NRMSE <= 0.98 * baseline  OR  BIC < baseline_BIC
v2 (audited):  held_NRMSE <= 0.98 * baseline  AND BIC < baseline_BIC
               AND train_RMSE <= 1.05 * baseline
```

v1 accepted held-luck overfits (pendulum `v**3`/`sign(v)` additions with
worse train RMSE and worse BIC passed on ~10% held gains). v2 cleaned the
control (no spurious submissions) but its train gate rejects the TRUE
sparse model: the 4-term baseline overfits the fit window, so sparse truth
loses on train by construction (S drag 0.117 vs 0.102). The documented
follow-up is held-NRMSE AND BIC, dropping train -- BIC already prices
complexity.

## 6. Cross-take transfer via `rollout_score`

Coefficients transfer across takes; initial conditions do not. So
`rollout_score` refits the 2 ICs on the new clip's first 3 points
(`ic_points=3`) with least-squares, integrates the FULL grid from that
refit state, and scores the remaining tail up to 2 s. (An earlier version
started the integration at the tail with the t[0] state -- a 3-frame time
shift that scored exact truth at 0.23; the exact-coefficients identity test
now guards this.) Transfer is a reported column (`results/transfer.json`),
never a gate: true drag transfers (0.008 → 0.003 vs baseline 0.443 on a
second toss) while the `q*v` distractor does not (0.004 → 0.349).

## 7. Identifiability limits

- **Single-trajectory aliasing:** on one noisy orbit, `q*v` can out-predict
  the true `abs(v)*v` on held data (S ablation: `q*v` held 0.004 vs truth
  0.008) while BIC still prefers truth (-65.6 vs -61.0) and transfer
  exposes the distractor (0.349 vs 0.003). Held-NRMSE alone does not
  identify structure; BIC + transfer agree.
- **Thin arcs:** 20-pt real windows (12 fit / 8 scored on the hero) barely
  resolve drag however pooled -- the joint two-take fit gets the sign right
  but ties gravity-only on BIC. Synthetic carries the discovery claim; the
  real hero carries tracking + baseline-failure evidence.
- **Direction reversal is load-bearing:** without `v` changing sign,
  `v**2` mimics `abs(v)*v` exactly and `sign(v)` mimics the constant term
  (pure drops were re-filmed as up-down arcs for this reason).
- **Gravity-only (`["1"]`, projectile `(["0"], ["1"])`) is always reported
  as a control** (`cli.py` renders baseline / control / final side by side).
  On the real hero it beats the drag model (0.065 vs 0.187) -- that is the
  reported result, not hidden. A submission must beat the baseline, and the
  transcript must not claim more than the evidence supports.

## 8. Uncertainty on coefficients

Winning fits report `±2σ` intervals from the shooting Jacobian
(`physics.coefficient_stderr`, stored on `FitResult.stderr`).

Method choice: Jacobian-based asymptotic covariance, not residual
bootstrap. A bootstrap would refit the shooter dozens of times per
clip -- minutes each on long windows against the 60 s fit budget --
while the least-squares Jacobian at the optimum is already in hand:
`cov = s2 * (J^T J)^-1` with `s2 = RSS/dof` (`RSS = 2*cost`), standard
errors from `sqrt(diag(cov))`, pseudo-inverse so rank-deficient
(aliasing) cases come back as inf/nan instead of raising.

Caveats, read before quoting an interval: the formula assumes iid
Gaussian residuals and a correctly specified model, so under tracking
noise or model misspecification it underestimates -- treat the band as
a lower bound on real uncertainty. On noiseless synthetic data the
residuals sit at the integrator floor (`RSS ~ 1e-19`), so the band
collapses to the 8th decimal and cannot see integrator bias a few
σ away; the coverage test therefore uses the same toss with
fixed-seed noise (`tests/test_uncertainty.py`), where truth sits
comfortably inside `±2σ`. Wide bands on real clips are reported as
wide, not dropped.
