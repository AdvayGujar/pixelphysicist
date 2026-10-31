# Scientist modes

Four Spark scientist modes. Each is a separate prompt + tool wiring in
`src/pixelphys/scientist.py`, selected with `pixelphys run --mode`.
The fitting machinery underneath is shared (see `docs/maths.md`).

## 1-DOF (default, `--mode 1d`)

- **Setup:** one generalized coordinate `q(t)` from video. Usually vertical
  pixel position for a tossed ball, or horizontal displacement for the
  pendulum. Sign convention is whatever the tracker emits (pixels,
  down-positive rows unless converted); the constant term absorbs gravity.
- **Baseline:** `["1", "q", "v", "v**2"]` (`src/pixelphys/physics.py`,
  `BASELINE_TERMS`). The unsigned `v**2` is a deliberate distractor: it
  cannot oppose motion in both directions, so any bidirectional drag
  breaks it visibly.
- **Hidden physics Spark looks for:** `abs(v)*v` (signed quadratic drag).
  Menu terms (`MENU_TERMS`): `abs(v)*v`, `sign(v)`, `q*v`, `v**3`,
  `sin(q)*v`, `cos(q)*v`.
- **Why it is identifiable:** a tossed ball reverses direction, so signed
  drag (`abs(v)*v`) and unsigned `v**2` make opposite predictions on half
  the trajectory. A pure drop (no reversal) cannot separate them
  (`docs/FAILURES.md`, 2026-10-27: first filmed takes were drops, re-filmed).
- **Synthetic ground truth:** `TossParams(v0=6.0, gravity=9.81, drag=0.35)`,
  model `q'' = -g - c*v*|v|` (`src/pixelphys/synthetic.py`).
- **Measured result:** 5/5 gate GREEN (`results/day4.json`, prompt/guard
  v1 -- predates the prompt-v2 parsimony rule, so historical; see
  `docs/results.md` §7): Spark submits
  `[1, abs(v)*v]` with `[-9.81, -0.35]`, held-NRMSE ~1e-9 vs baseline 0.135,
  BIC -788 vs -344.

## Projectile (`--mode projectile`)

- **Setup:** coupled coordinates `x(t)` (right), `y(t)` (UP, converted from
  image rows), origin at window start (`cli.py:projectile_coordinates`).
  State is `[x, y, vx, vy]`; each axis gets its own acceleration model.
- **Baseline:** x: `["1", "x", "vx", "vx**2"]`, y: `["1", "y", "vy",
  "vy**2"]` (`PROJECTILE_BASELINE_X/Y`). Same unsigned-square flaw per axis.
- **Hidden physics:** speed-coupled quadratic drag, exact strings
  `sqrt(vx**2+vy**2)*vx` (x) and `sqrt(vx**2+vy**2)*vy` (y)
  (`DRAG_TERM_X/Y`). The same (negative) coefficient is expected on both
  axes; wildly disagreeing signs mean the structure is wrong. Linear drag
  (`vx` / `vy`) is the weaker alternative.
- **Why it is identifiable:** the coupling through total speed
  `|v| = sqrt(vx**2+vy**2)` cannot be expressed by any per-axis term, so
  the baseline fails on curved flight while the coupled term fits both axes.
- **Synthetic ground truth:** `ProjectileParams(v0=11.0, angle=52°,
  gravity=9.81, drag_k=0.06)` (`src/pixelphys/synthetic.py`).
- **Measured result:** 5/5 gate GREEN (`results/day4.json`, prompt/guard
  v1 -- predates the prompt-v2 parsimony rule, so historical): Spark submits
  x: `[sqrt(vx**2+vy**2)*vx]`, y: `[1, sqrt(vx**2+vy**2)*vy]` with k=-0.06
  on both axes, g=-9.81, held-NRMSE ~2e-7 vs baseline 0.015, BIC -939 vs -484.

## Spring (`--mode spring`)

- **Setup:** `q(t)` = vertical displacement from equilibrium (median of the
  window), up positive (`cli.py:to_coordinate`). Without `--scale` the units
  are pixels, so stiffness/damping are per-pixel quantities.
- **Baseline:** same 4-term `[1, q, v, v**2]`: constant absorbs the
  gravity/equilibrium offset, `q` is linear stiffness, `v` viscous damping.
  A linear spring with viscous damping is fully inside the baseline, so
  Spark should submit the baseline unchanged on linear data.
- **Hidden physics:** `q**3` (nonlinear Duffing stiffness: the spring
  stiffens or softens at large stretch). Menu (`SPRING_MENU_TERMS`):
  `q**3`, `sign(v)`, `abs(v)*v`, `q*v`, `v**3`.
- **Why it is identifiable:** a cubic residual-vs-`q` trend grows with
  amplitude and cannot be absorbed by linear stiffness; damping terms
  depend on `v`, not `q`.
- **Synthetic ground truth:** `SpringParams(q0=0.55, stiffness=22.0,
  damping=0.25, cubic=6.0, duration 4 s)`, model `q''=-k*q-c*v-beta*q**3`.
- **Measured result:** design target is Spark submitting `q**3` with
  coefficients matching truth. Per `plan.md` §4.5c the synthetic Duffing
  end-to-end run submitted `q**3` (held 0.0008 vs baseline 0.0139); no
  committed results file pins these numbers, so treat them as reported,
  not re-verified here.

## Rolling (`--mode rolling`)

- **Setup:** `q(t)` = distance along the slope from window start,
  up-slope positive (`cli.py:to_coordinate`). There is no restoring
  coordinate: the prompt tells Spark to try sparse `[1, sign(v)]` forms
  before anything containing `q`.
- **Baseline:** `[1, q, v, v**2]`; the constant term is the downslope
  gravity component.
- **Hidden physics:** `sign(v)` (Coulomb rolling friction: a roughly
  constant deceleration that flips with velocity). Menu
  (`ROLLING_MENU_TERMS`): `sign(v)`, `abs(v)*v`, `v**3`, `q*v`.
- **Why it is identifiable only on reversal:** `sign(v)` separates from the
  constant term only because the ball rolls up, stops, and comes back --
  a one-direction roll cannot distinguish Coulomb friction from slightly
  stronger/weaker gravity. The prompt states this explicitly.
- **Synthetic ground truth:** `RollingParams(v0=3.4, slope_accel=1.6,
  coulomb=0.55, viscous=0.05, duration 6 s)`, model
  `q''=-a-mu*sign(v)-c*v`.
- **Measured result:** design target is Spark finding `sign(v)`. Per
  `plan.md` §4.5c the synthetic run recovered friction -139 vs truth
  -137.5 px/s² and gravity -393 vs -400 (held 0.045 vs baseline ~0.45);
  again reported in the plan, not pinned by a committed results file.

## Which mode do I use?

| My clip shows... | Mode |
|---|---|
| Ball tossed/dropped vertically, pendulum swing | `1d` |
| Ball flying in 2-D (arc across the frame) | `projectile` |
| Mass bouncing on a spring, oscillating vertically about equilibrium | `spring` |
| Ball rolling up/down a slope or ramp | `rolling` |
| Anything with a moving camera, bounce/impact in the window, or two objects | none: out of envelope |

## Known limits per mode

- **1-DOF:** gravity-only (`["1"]`) is the published control on real clips:
  on the real hero it beats the drag model, so a winning submission must
  beat the baseline and the transcript must not claim more
  (`scientist.py` header). Thin 20-pt real arcs barely resolve drag however
  pooled (joint two-take fit: right sign, BIC tied with gravity-only).
- **Projectile:** needs a clean 2-D arc; the color/motion tracker failing on
  static backgrounds silently fits furniture (Morpheus probe,
  `docs/FAILURES.md` 2026-10-29). Gravity-only control is `(["0"], ["1"])`.
- **Spring:** validated on short clips only. The 31 s Morpheus spring (930
  pts) defeats the long-window shooter: rollouts fail to integrate, an 8 s
  truncation scores held-NRMSE 5.2. Long/complex spring footage is out of
  envelope.
- **Rolling:** a one-direction roll is the trivial case: the constant-only
  control already scores ~0.007 and there is nothing to discover. Needs the
  up-and-back reversal.
