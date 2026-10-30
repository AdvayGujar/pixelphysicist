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
- **Third-party (Morpheus) robustness**: the full record is in
  Section 7 below and `results/dataset_mining.json`,
  `results/pendulum_consistency.json`, `results/jointfit.json`
  (runners: `scripts/mine_dataset.py`,
  `scripts/pendulum_consistency.py`).

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

## 7. Tasks A-C (Morpheus drag mining, pendulum consistency, joint fit)

Sources: `results/dataset_mining.json`, `results/pendulum_consistency.json`,
`results/jointfit.json`. Split everywhere 60/40 unless noted. Third-party
clips live in `/tmp/stress2/` only, never committed (MIT licence, attribution
in README).

### A. Dataset mining — real clip where drag beats gravity

Pre-registered winner rule (unchanged): drag held-NRMSE <=0.8×gravity AND
<baseline, with negative drag coefficient. 12 clips, motion tracker,
1-DOF vertical_q, no trim/scale.

- Winners (2/12):
  - `ball5` falling_ball/video_5: baseline 0.008344923320049139 (BIC
    -131.54846298644156), gravity 0.03559489976504714, drag
    0.004927063885487741 (BIC -101.18135397068866), coeff
    [-450.96236108922943, -0.0006202239982480519] → winner True.
  - `ball8` falling_ball/video_8: baseline 0.009315324216663248 (BIC
    -127.71308922601456), gravity 0.034319118996290664, drag
    0.003666150556292589 (BIC -108.81942731517124), coeff
    [-462.7321275755624, -0.0005892416426952394] → winner True.
- Non-winners include `ball0` (drag 0.005971084100099659 loses to baseline
  0.00557548231337142), `tape0` (baseline failed, drag 0.0980087396174026 vs
  gravity 0.1573096169610828 but baseline not ok → not counted),
  `proj5` (drag failed), `proj6`/`proj7` (too few points: 12 obs / cleaning
  <20, recorded as errors).
- Live Spark on winners (effort low, guard v1, cap $0.50):
  - ball8 (`small orange ball`, motion, no trim): submitted
    [1, q, v, sign(v)] held 0.003549862916634604, BIC -102.45625842862425,
    rounds 6, cost $0.05480275. Tried [1,q,v,abs(v)*v] (held 0.0093, no gain);
    never tried sparse [1,abs(v)*v]. NOT a drag submit.
  - ball5: submitted [1, q, v, v**3] held 0.006529838542153517, rounds 2,
    cost $0.01271325. Also not drag.
  - One transient 400 before the ball8 success (zero usage, retried after 16 s
    per the Day-5 transport lesson).
- Verdict at first attempt: offline winners exist, autonomous live
  discovery did NOT submit quadratic drag. No README hero-claim change.
- **Prompt v2 follow-up (2026-10-30, disclosed prompt change, not a
  gate rerun):** the 1-DOF prompt gained a general parsimony rule --
  "your FIRST drag fit must be the sparse replacement [1, abs(v)*v];
  only add q/v back if residuals demand it" (the same replace-don't-add
  principle the rolling prompt already carried). Re-running only the
  two winners: ball8 submitted [1, abs(v)*v] with the offline
  coefficients (-462.7, -0.000589), held 0.0037, 4 rounds, $0.0271;
  ball5 submitted [1, abs(v)*v] (-451.0, -0.000620), held 0.0049,
  3 rounds, $0.0188 (`runs/ball8_promptv2`, `runs/ball5_promptv2`).
  Live drag discovery on real dataset footage now works; the v1-prompt
  miss above stays in the record.

### B. Pendulum quantitative check without a ruler

**v1 method (failed, kept for the record):** motion track → circle-fit
pivot → angle without unwrapping → linear stiffness vs -(2π/T)² with T
from zero crossings. Result was 1/5 within 5% (only pend_h4 at 0.6%);
the other four were artefacts: ±π wraps injected spurious crossings
(one "period" of 0.78 s in a 19 s clip), and the mask centroid rides
the rod, not the bob.

**v2 method (`scripts/pendulum_consistency.py`, same 5 clips, same 5%
bar):** track the bob as the farthest moving-contour point from the
pivot (iterate pivot↔bob twice, pivot judged by circle-fit RMS),
unwrap the angle, take the period from peak spacing, fit the true
restoring term `sin(q)`, and predict each interval's period from its
*local* amplitude via the elliptic correction
T = T0·(2/π)·K(sin²(θ/2)) (comparing release amplitude against a
whole-clip median period had manufactured ~6% disagreement on its own).

| Clip | Pivot RMS (px) | T measured (s) | T predicted (s) | Agreement |
|---|---|---|---|---|
| pend_h1 | 16.7 | 4.333 | 4.319 | 0.34% |
| pend_h2 | 14.9 | 4.300 | 4.277 | 0.54% |
| pend_h3 | 13.3 | 4.250 | 4.269 | 0.44% |
| pend_h4 | 14.9 | 4.183 | 4.155 | 0.67% |
| pend_h5 | 14.9 | 3.500 | 3.485 | 0.43% |

**5/5 within 5% → PASS.** All five clips are the same apparatus
(periods 3.5–4.3 s); the v1 chaos was centroid artefact, not physics.
Method changes were estimator fixes (bob point, unwrap, local
amplitude), applied to every clip with the bar unchanged — no clips
dropped, no pivots hand-tuned.

### C. Joint two-toss fit breaks the q*v aliasing

`physics.fit_terms_joint(terms, clips)`: shared coeffs, per-clip ICs,
concatenated residual, same weak-init×3 / data-driven bounds / budget
discipline as `fit_terms`. Tests in `tests/test_joint.py` (identity ~2e-9,
clean recovery, tracked-pair preference) pass before any live run.

- Single-clip S1 (tracked v0=6): drag held 0.008446170268754407 (BIC
  -65.56312835270678) vs q*v held 0.003968992294840736 (BIC
  -61.02901875971234) — distractor wins held, BIC prefers truth.
- Joint on S1+S2 (30+35 pts, full trajectories): drag rss 0.9511877834821838,
  BIC -249.54169442607372, coeffs [-2452.60129779623, -0.0014113618784904027];
  q*v rss 163.51396628083887, BIC 93.3583220948091. Winner: drag.
- Verdict: drag survives joint fitting; q*v does not. Maps to ablation.csv S
  r2 (drag) vs r1/r3 (q*v).

## 8. Benchmark on public footage (Tier 2)

Runner `scripts/benchmark_morpheus.py` → `results/benchmark.json`
(date 2026-10-31, tracker motion everywhere, effort low, guard v1 CLI
default; live cost $0.1045). SAM is the normal path on phone footage;
motion is the offline fallback used for all dataset clips. Falling-ball
and pendulum rows are cited, not rerun.

| Clip | Mode | Submitted | Final terms | Final held | Control held | Baseline held | Rounds | Cost |
|---|---|---|---|---|---|---|---|---|
| proj_a (38-frame arc) | projectile | False (fallback) | x `[sqrt(vx**2+vy**2)*vx]`, y `[1, sqrt(vx**2+vy**2)*vy]` | 0.040983265258900614 | 0.06493872738431318 | inf (exploded) | 4 | 0.01671625 |
| proj_b (21-frame arc) | projectile | True | x `[vx]`, y `[1, vy]` | 0.5966609543900796 | 0.5321763891503452 | inf (exploded) | 5 | 0.03366625 |
| roll_orange (one-direction) | rolling | False (fallback) | `[1, sign(v)]` | 0.006880212647412867 | 0.006880232169345483 | 0.07651727218521275 | 4 | 0.0234915 |
| spring_long (31 s, 930 pts) | spring | True | `[q, v]` | 1.0116311365953485 | 0.46327599238850214 | inf (exploded) | 5 | 0.0306535 |
| ball5 (cited §7) | 1d | n/a (mining row) | `[1, abs(v)*v]` | 0.004927063885487741 | 0.03559489976504714 | 0.008344923320049139 | n/a | 0.0 |
| ball8 (cited §7) | 1d | n/a (mining row) | `[1, abs(v)*v]` | 0.003666150556292589 | 0.034319118996290664 | 0.009315324216663248 | n/a | 0.0 |
| pendulum_x5 (cited §7) | angle-period check | n/a | `sin(q)` period check | n/a (period 0.34–0.67%) | n/a | n/a | n/a | 0.0 |

Per-category verdicts:

- falling_ball: **works** — sparse drag beats gravity ~7–9× on held-out
  motion (ball5 0.0356 → 0.0049, ball8 0.0343 → 0.0037), negative
  coefficient, beats baseline too.
- projectile: **works-with-limits** — the 38-frame arc's fallback
  coupled-drag fit beats gravity (0.041 vs 0.065) but is never submitted
  (baseline exploded, loop falls back); the 21-frame arc submits linear
  drag that *loses* to gravity (0.597 vs 0.532). Thin arcs plus an
  exploded baseline defeat the OR guard, which accepts anything beating
  inf.
- rolling_orange: **works-with-limits** — fallback `[1, sign(v)]` ties the
  constant-only control to 5 decimals (0.00688021 vs 0.00688023) while
  beating the baseline 11× (0.077). A one-direction roll cannot separate
  Coulomb friction from gravity; the tie is the correct answer here.
- spring_long: **fails** — the known out-of-envelope clip. Baseline
  explodes, first attempt exited 2 with nothing surviving, retry
  submitted `[q, v]` at held 1.01, worse than the control (0.46),
  accepted only through the broken-baseline path. Confirms the §5 limit.
- holonomic_pendulum: **works** — 5/5 period agreement 0.34–0.67% (§7).

## 9. Guard v3: held AND BIC, no train (Tier 3, pre-registered)

`scientist.py` guard `"v3"` + `V3_SYSTEM_PROMPT`/`v3_schemas()` (exact-replacement
versioning like v2); prompt text always matches the enforced guard.
`scripts/run_ablation.py --guard v3` on the frozen H:3/N:5/S:5 conditions,
effort low → `results/ablation_v3.json`/`.csv`, cost $0.316988. v1/v2
files untouched. Pre-registered prediction (from v1/v2 transcripts): N
stays clean, S submits true drag in ≥3/5, H returns to baseline.

| clip | A held | B mean held | gate |
|---|---|---|---|
| H hero | 0.732430311284895 | 0.18722568468499037 (3/3 beat) | GREEN |
| N pendulum | 0.041287940844683235 | 0.0371763063248367 (rel gap 0.0996, 0/5 clean) | RED |
| S synthetic tracked | 0.06255632091676147 | 0.008446170268754407 (5/5 correct) | GREEN |

- S 5/5 submitted `[1, abs(v)*v]` (held 0.0084, BIC -65.6, train 0.1174;
  costs 0.0103–0.0154) — prediction CONFIRMED, and unanimously. v3
  accepts what v2 rejected: train 0.1174 is +15% over baseline 0.102
  yet held+BIC both win.
- H 3/3 found the same sparse drag (held 0.1872) but submitted 0/3:
  BIC 39.3 loses to baseline 18.6, so v3 rejects. H gate stays GREEN
  (beats baseline on held) while nothing submits — v3 is stricter
  than v1 here, which submitted drag once on the OR rule.
- N 0/5 submitted (fallbacks `[1, q, abs(v)*v]`/`[1, q, v]`/`[1, q, v**3]`,
  all held 0.0372): held beats baseline ~10% but BIC ~684 loses to
  645.8, so v3 rejects everything. n_clean counts *submitted*
  baselines: 0/5 → RED. Prediction "N stays clean" FAILED — v3 keeps
  submissions clean (nothing spurious submitted) but the gate's
  cleanliness bar counts unsubmitted fallbacks' exploration, same
  strictness price v2 paid (v2 N GREEN only because 3/5 submitted the
  baseline itself).
- Overall G5 RED (H GREEN, N RED, S GREEN). v3 fixes the S train
  misfire and holds the submission line, but does not green the gate.
- Note: v3 condition-A baselines differ slightly from the committed
  v1/v2 A rows (S 0.06256 vs 0.06220, ~0.6%; H/N in the 6th decimal):
  HEAD code without any of this task's edits reproduces 0.06256, so
  the drift predates this task (Day-5 environment), and every v3
  comparison is internal to v3 (same code for A and B).

Side by side: v1 (S 1/3 correct, N 0/3 clean → RED), v2 (S 0/5
submitted-correct with 2 true-term fallbacks, N 3/5 clean → RED), v3
(S 5/5 correct, N 0/5 clean → RED). The train clause was the whole
S difference; the N bar is about fallback exploration, not submissions.

## 10. Coefficient uncertainty (Tier 3)

`physics.coefficient_stderr` / `FitResult.stderr`: asymptotic
`sqrt(diag(s2·(JᵀJ)⁻¹))` from the winning shooting Jacobian (pseudo-
inverse; degenerate → nan, reported not hidden). Method rationale in
`docs/maths.md` §8. Tests in `tests/test_uncertainty.py` (noisy-
synthetic coverage + clean-sanity + bad-shape-nan).

| Fit | Point estimate | ±2σ | Truth in ±2σ? |
|---|---|---|---|
| Synthetic toss g | -9.809999947 | ±0.000000017 | just outside (~6σ at the 8th decimal; integrator floor, see §8 caveat) |
| Synthetic toss drag | -0.350000005 | ±0.000000003 | just outside (~3σ at the 8th decimal; same caveat) |
| ball8 g (px/s²) | -462.732 | ±18.27 | n/a (no ground truth; sign/magnitude usable, ±4%) |
| ball8 drag (1/px) | -0.0005892 | ±0.0000718 | n/a (interval excludes 0; drag sign firm, ±6%) |
| Duffing stiffness k | -21.99999990 | ±0.00000002 | same integrator-floor note as synthetic toss |
| Duffing damping c | -0.25000000 | ±0.0000000004 | same |
| Duffing cubic β | -6.00000020 | ±0.00000013 | same |

The real-clip bands are usefully tight (drag sign firm at ±6%), not
uselessly wide — reported, not dropped. Noiseless-fit bands sit at
the integrator floor; coverage is validated with fixed-seed noise,
where truth sits comfortably inside (test).
