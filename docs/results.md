# Experimental record

Sources: `results/baseline.json`, `results/day4.json`, `results/day4_diag.json`,
`results/ablation.json` (+ `.csv`), `results/ablation_v2.json` (+ `.csv`),
`results/transfer.json`. Split protocol everywhere: fit first 60%, score
continuation on held-out tail, horizon min(2 s, clean tail). Held metric is
NRMSE; BIC counts coefficients plus fitted initial conditions.
Quoted numbers are exact JSON values.

## 1. Day-3 baselines

Baseline library everywhere: `[1, q, v, v**2]`. The unsigned `v**2` is a
deliberate distractor: it cannot express bidirectional drag.

**Synthetic toss with known quadratic drag** (`baseline.json`, `synthetic`):

| model | held NRMSE | BIC | train RMSE |
|---|---|---|---|
| baseline `[1, q, v, v**2]` | 0.135203579713803 | -344.0479340805834 | 4.3680198413252064e-05 |
| gravity-only `[1]` | 0.221027084167157 | -153.31296088273095 | 0.011113911456367882 |
| true terms `[1, abs(v)*v]` | 1.414968067633179e-09 | -788.4288647219829 | 2.2342301449434832e-10 |

True-term coeffs: `[-9.809999950279593, -0.35000000396031355]`
(g = 9.81, drag 0.35). n_fit 18, n_scored 12.

**Hero split, take 5** (fit take 5, held take 5; n_fit 12, n_scored 8):

| model | held NRMSE | BIC | train RMSE |
|---|---|---|---|
| baseline | 0.7324369618049869 | 18.58344287324811 | 1.1654222378445156 |
| gravity-only | 0.06488475679031361 | 40.11130021888802 | 3.8988858957309582 |
| drag `[1, abs(v)*v]` | 0.18723623067529227 | 39.3441804853021 | 3.4048120684409477 |

Gravity-only beats both baseline and drag on the real arc. The drag term
beats the baseline ~4x but still loses to plain gravity.

**Cross-take** (fit take 5, score take 4; n_fit 20, n_scored 17):

| model | held NRMSE | BIC |
|---|---|---|
| baseline | 0.23364621606449393 | 61.11492798043567 |
| gravity-only | 0.28088642663987046 | 127.56966476136922 |
| drag `[1, abs(v)*v]` | 0.36630019435368294 | 117.83403574853014 |

Cross-take the drag model is worst of the three. Take-5-fitted numbers do
not transfer to take 4 under any model.

**Pendulum** (n_fit 129, n_scored 60):

| model | held NRMSE | BIC | train RMSE |
|---|---|---|---|
| baseline | 0.04128793329098349 | 645.7797087862409 | 10.913529236833847 |
| gravity-only | 0.9507850102456791 | 1422.723662661359 | 234.6097169182901 |

Fitted stiffness (coeff of `q`): -11.510255720211484, vs truth -11.6 cited
in the `physics.py` `accel_scale` comment (the old window-geometry bound
clipped at ±5.2 and pinned the fit; the data-driven `accel_scale` ceiling
fixed it). Pendulum take-2 period 1.850 s (`data/clips/notes.md`).

## 2. Day-4 Spark gates

**G4 (1-DOF) 5/5, gate GREEN; G4b (2-DOF projectile) 5/5, gate GREEN.**
Total cost $0.125123 (`day4.json`).

G4: four runs submitted exactly `[1, abs(v)*v]` with coeffs
`[-9.809999950279593, -0.35000000396031355]`, held NRMSE
1.414968067633179e-09, BIC -788.4288647219829, train RMSE
2.2342301449434832e-10 (rounds 4,4,4,4; costs 0.015065, 0.015867,
0.015811, 0.01456). Run g4-3 submitted `[1, q, v, abs(v)*v]` with coeffs
`[-9.810000343130996, 4.6661908008840025e-07, -1.528634275175645e-07,
-0.34999995557774405]`, held 1.2071149747832434e-08, BIC
-757.0380903754517 (rounds 3, cost 0.01074); q/v coeffs ~1e-7, i.e. zero.
Baseline beaten: 0.135203579713803 -> ~1e-9, BIC -344.0479340805834 -> -788.

G4b: all five runs submitted x `[sqrt(vx**2+vy**2)*vx]` coeff
`[-0.06000023581449877]`, y `[1, sqrt(vx**2+vy**2)*vy]` coeffs
`[-9.81000581367412, -0.06000047405600888]` (rounds 3 each; costs 0.010623,
0.01027, 0.010703, 0.010687, 0.010797). Held 2.211741500397925e-07, BIC
-938.7347840804572, train 2.4460156748538625e-08 vs baseline held
0.015067293598666827, BIC -484.3017156673648. Shared drag ~0.06 both axes.

**Baseline-first verification** (`day4_diag.json`, not a gate): g4 1/1,
g4b 1/1, cost $0.032127. Same coefficients recovered with baseline
residual plots attached up front (Day-4 audit finding 1).

## 3. Day-5 ablation

A = baseline once; B = baseline + Spark, effort `low`, all from tracked
observations. G5 RED in both experiments.

**v1** (`ablation.json`, repeats 3, cost $0.217561):

| clip | A held | B mean held | gate |
|---|---|---|---|
| H hero | 0.7324369618049869 | 0.3985231810388316 (improvement 0.4558942245941168, 3/3 beat) | GREEN |
| N pendulum | 0.04128793329098349 | 0.03691439675057082 (rel gap 0.10592771766001105, 0/3 clean) | RED |
| S synthetic tracked | 0.06220275838026542 | 0.005447356944987952 (1/3 correct) | RED |

- H: all 3 beat baseline on held NRMSE (0.4898774689793389,
  0.18723623067529227, 0.5184558434618637) but all lose on BIC
  (34.35 / 39.34 / 22.24 vs 18.58). Terms found: r1 `[1, q, abs(v)*v]`,
  r2 `[1, abs(v)*v]`, r3 `[1, q, v, sign(v)]`.
- N RED: Spark adds spurious terms in 3/3: r1 `[1, q, v, v**3]` (held
  0.03716271124899527, BIC 689.2577905080836, train 12.916724319288248),
  r2/r3 `[1, q, sign(v), v**3]` (held 0.0367902395013586, BIC
  713.6549877796972, train 14.19778136746781) vs baseline BIC
  645.7797087862409, train 10.913529236833847. Extra terms shave ~11%
  off held error while fitting train and BIC worse. r1/r2 never called
  `submit_model` (unsubmitted fallbacks).
- S RED on exact recovery (1/3), GREEN on beating baseline (3/3, -91%).
  True `[1, abs(v)*v]` once (r2: held 0.008446490488920417, BIC
  -65.56312834150981, train 0.11737862863652405); `q*v` distractor twice
  (r1/r3 `[1, q, v, q*v]`: held 0.003947790173021718, BIC
  -61.029017499515135, train 0.11338353271192166). The distractor
  out-predicts truth on held NRMSE while BIC prefers truth (-65.56 < -61.03).

**v2** (`ablation_v2.json`, guard v2 = held AND BIC AND train-within-5%;
repeats H:3, N:5, S:5; cost $0.437849):

| clip | A held | B mean held | gate |
|---|---|---|---|
| H | 0.7324369618049869 | 0.5507033847617553 (improvement 0.2481217995817346, 1/3 beat) | RED |
| N | 0.04128793329098349 | 0.03952315892204935 (rel gap 0.04274310260328621, 3/5 clean) | GREEN |
| S | 0.06220275838026542 | 0.04070025122372742 (0/5 correct) | RED |

- N GREEN: baseline retained 3/5 (submitted); `v**3` and `q**2` fallbacks
  once each. The one accepted extra term (N r5 `[1, q, q**2]`, held
  0.036589283488301, BIC 642.4943022667348, train 10.98033101096029)
  passes all three narrowly (held -11.4%, BIC better, train +0.6%).
- S 0/5 submitted-correct: true drag recovered 2/5 but only as
  unsubmitted fallbacks (held 0.008446490488920417 each); baseline
  submitted 3/5; no `q*v` submissions.
- Train-rule miscalibration: the overfit baseline owns the fit window, so
  true sparse models lose on train (S drag 0.11737862863652405 vs
  0.10206605820617158, +15%; H drag 3.4048120684409477 vs
  1.1654222378445156) and can never pass v2. Follow-up: held AND BIC,
  drop train. Strictness price: fallbacks don't count under the
  pre-registered rule. G5 RED stands.

## 4. Transfer checks

`transfer.json` (`rollout_score`, ICs refit; never rewrites a gate).
S take-2 is `obs_synthetic_toss_drag_v0b.json` (v0=7.5); N take-2 is
`obs_pendulum_take2.json`.

| model | held | transfer | transfer n |
|---|---|---|---|
| S `[1, abs(v)*v]` (v1 r2) | 0.008446490488920417 | 0.003226792438957984 | 32 |
| S `[1, q, v, q*v]` (v1 r1, r3) | 0.003947790173021718 | 0.3490912788510475 | 32 |
| S baseline (v2 r2/r3/r5) | 0.06220275838026542 | 0.4434297002659848 | 32 |
| N `[1, q, sign(v), v**3]` (v1 r3) | 0.0367902395013586 | 0.07458649023722023 | 60 |
| N baseline (v2 r1/r3) | 0.04128793329098349 | 0.08843910835744867 | 60 |
| N `[1, q, q**2]` (v2 r5) | 0.036589283488301 | 0.09020815026712904 | 60 |

True drag transfers (0.003 vs baseline 0.443); the `q*v` distractor does
not (0.349 vs 0.443). N transfer is inconclusive (0.075 vs 0.088, both
~2x worse on the harder take); the v2 `q**2` find does not beat baseline
in transfer (0.090 vs 0.088). BIC + transfer agree on S; held-NRMSE alone
does not.

## 5. Robustness

- **Spring mode**: validated end-to-end on the synthetic Duffing clip
  (README: submitted `q**3`, coefficients matching truth).
  `tests/test_spring_rolling.py` covers period-vs-`sqrt(k)` identity,
  amplitude conservation, renderer round-trips, and exact cubic-stiffness
  recovery (`spring_duffing.json` fixture: k=22.0, damping 0.25, cubic 6.0).
- **Rolling mode**: validated on the synthetic slope (README: recovered
  friction -139 vs -137.5 px/s^2 and gravity terms). Tests cover the
  sign-flip-and-return trajectory, the frictionless parabola, and exact
  Coulomb-friction recovery. One-direction rolls are reported as the
  constant-deceleration case they are.
- No third-party-clip robustness numbers exist in any `results/*.json`;
  see the flag at the end of this file.

## 6. What the numbers do and do not show

Supported: Spark recovers exact synthetic drag laws 5/5 in both modes
(G4/G4b GREEN); the baseline-first verification reproduces this; the hero
drag term beats the *baseline* ~4x on 20-point arcs; v2 guard cleans the
negative control (N GREEN); transfer + BIC jointly identify the true S
term where held-NRMSE alone is fooled by `q*v`.

Explicitly NOT claimed: **no drag-on-real-arcs claim**. On real hero data
gravity-only (0.06488475679031361) beats the drag model
(0.18723623067529227) on the split protocol, and cross-take the drag
model (0.36630019435368294) is worst of the three. Discovery is proven on
synthetic truth, not on phone footage. The negative control is not clean
under v1 (spurious `v**3`/`sign(v)` in 3/3); extra terms need BIC and
transfer agreement.

**Flags vs the summary this file was asked to match.** (a) The
Morpheus/third-party numbers (projectile: gravity-only 0.0198 vs drag
0.030 vs baseline 0.216; rolling orange: constant-only 0.0069) were
initially only in run artifacts under `runs/`; they are now pinned in
`results/modes.json` (extracted from the runs' `model.json` files on
2026-10-30), alongside the spring/rolling synthetic mode results
(spring submitted `q**3` at held 0.0008 vs baseline 0.0139; rolling
found `sign(v)` at held 0.045). The runs themselves stay local. (b) N v2 `n_clean` is "3/5", not a
per-repeat breakdown issue: repeats 1-3 submitted/kept baseline, r4 was an
unsubmitted `v**3` fallback, r5 submitted `q**2`. (c) S v2 B-mean
0.04070025122372742 mixes 2 fallbacks at 0.0084 with 3 baselines at
0.0622 — the mean is reported verbatim, not as a success.
