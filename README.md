# Pixel Physicist

Film a moving object with a phone. SAM 3.1 tracks it, Muse Spark acts as the
scientist: it reads the residual evidence, diagnoses why the first model fails,
proposes the missing physics, and the system validates the result by predicting
motion it never saw.

## How it works

```
phone clip
  → tracking (SAM 3.1 video, HSV color fallback)
  → calibration to coordinates: 1-DOF q(t), or projectile x(t), y(t)
  → classical fit of a default physical model (weak-form init + shooting)
  → Spark loop: read residual plots, propose the missing term, refit
  → held-out prediction, scored by NRMSE + BIC over a fixed horizon
```

The numeric fitting is classical. Spark's contribution is proposing and
revising model *structure* from residual evidence; an ablation with a negative
control measures exactly that contribution. The pipeline handles one object
and one static camera, in two scientist modes:

- **1-DOF mode** — a single coordinate `q(t)` (vertical toss, pendulum
  displacement) with terms in `q, v`; Spark looks for terms like quadratic
  drag `abs(v)*v` or Coulomb friction `sign(v)` outside the baseline
  `[1, q, v, v**2]`.
- **Projectile mode** — two coupled coordinates `x(t), y(t)` with terms in
  `x, y, vx, vy`; Spark looks for drag coupled through the speed, e.g.
  `sqrt(vx**2+vy**2)*vx`, with the same coefficient expected on both axes.

## Current state (Day 5 done: ablation collected, G5 RED)

Built and verified:

- Tracking: SAM 3.1 video and an HSV color fallback; synthetic clips with
  known ground truth in both modes (`tests/fixtures/`).
- Fitting: weak-form (integral) initialization plus shooting refinement;
  every candidate is scored on held-out motion it never saw
  (`results/baseline.json`, reproduced by `scripts/run_baseline.py`).
  Synthetic truth is recovered exactly in both modes; the baseline fails
  visibly on the real hero clip (held-out NRMSE 0.73 vs gravity-only 0.065
  — the simple control is reported, not hidden).
- Scientist: the 3-tool Spark loop (`fit_model`, `plot_residuals`,
  `submit_model` with a beat-the-baseline guard) is built in both modes
  and both 5-run synthetic gates are green (G4/G4b 5/5, `results/day4.json`
  via `scripts/run_day4.py`): the hidden drag coefficients are recovered
  exactly. After an audit noted the gate runs fit the menu answer before
  inspecting any residual, the loop now attaches the baseline residual
  plots up front so the first fit diagnoses seen evidence
  (`results/day4_diag.json`, 1 verification run per mode).
- Ablation: 12 runs (3 baselines + 9 Spark loops at `low` effort, all from
  tracked observations) via `scripts/run_ablation.py` — see the table
  below. G5 is RED: the hero claim holds, the negative control and the
  exact-term recovery do not.

Not built yet: the one-command `pixelphys run …` wiring, the report and
ghost-overlay renderer, and the demo video.

## Ablation (Day 5, `results/ablation.csv`)

Split protocol throughout: fit the first 60%, score the next 2 s
(min(2 s, clean tail)) by continuing each model's own fitted trajectory.
B = baseline + Spark (3 repeats, `reasoning_effort="low"`); A = baseline.
Mean B vs A; "beat" = lower held-out NRMSE.

| Clip | A held-NRMSE | B held-NRMSE (mean of 3) | Beats A | Terms found | Cost |
|---|---|---|---|---|---|
| H hero (tossed ball, 20 pts) | 0.732 | 0.399 (0.490 / 0.187 / 0.519) | 3/3 | drag family (`abs(v)*v` ± `q`, `sign(v)`) | ~$0.045 |
| N negative control (pendulum, 215 pts) | 0.041 | 0.037 (0.037 / 0.037 / 0.037) | 3/3 (worse: see below) | spurious `v**3` + `sign(v)` every repeat | ~$0.098 |
| S synthetic tracked (30 pts) | 0.062 | 0.005 (0.004 / 0.008 / 0.004) | 3/3 | true `abs(v)*v` 1/3; `q*v` distractor 2/3 | ~$0.074 |

Total: $0.22 for 9 Spark runs. Gate: H GREEN (3/3 beat, −46% mean),
N RED (Spark adds terms in 3/3; 10.6% mean gap, outside the 5% band),
S RED on exact recovery (1/3), GREEN on beating the baseline (3/3,
−91% mean).

**Honest limits (read before citing any number above).**
The numeric fitting is classical; Spark proposes structure only. The
hero result is real but narrow: on 20-point arcs the drag term beats the
*baseline* 4× yet still loses to gravity-only (0.065) on real data, so no
drag-on-real-arcs claim is made — the discovery mechanism is proven on
synthetic truth (G4 5/5 exact), not on phone footage. The negative
control failed: Spark added cubic/Coulomb terms that shave ~10% off the
pendulum's held NRMSE while *worsening* train RMSE (12.9–14.2 vs 10.9)
and BIC (689–714 vs 646) — held-luck overfitting the 2%-NRMSE-or-BIC
submit guard accepts. Requiring NRMSE *and* BIC agreement is explicit
future work, not implemented. On tracked (vs truth-array) synthetic,
exact recovery drops to 1/3: the full-library-plus-drag fit scores worse
than the baseline (0.070 vs 0.062), so Spark usually never sparsifies to
the true 2-term form, and a correlated `q*v` distractor out-predicts the
truth on this noisy realization (BIC still prefers the truth, −65.6 vs
−61.0). One object, one static camera, short 1-DOF clips only.

## Seeing the equation (report visual, lands with the renderer)

The intended main output of a run is a single graph, not a table of numbers.
On the same axes it will show the measured trajectory as points, the baseline
model's curve, and the final model's curve continued (dashed) into motion
the fit never saw — with the discovered equation, its fitted coefficients,
and both models' held-out NRMSE written on the graph itself. If the final
curve peels away from the held-out points, the graph says so; that is the
validation, not a decoration. (The residual plots Spark reads during a run
already exist; the combined equation graph is part of the Day-6 report.)

## Still planned (post-freeze; gated on G5 — currently RED, so cut)

A one-go projectile flow — finished throw clip in, fitted equations,
equation graph, report, and a 3-D replay ball re-flying the *fitted*
equations out — plus the replay renderer itself. Two honest limits are part
of the design: SAM 3.1 is a batch API, so this is a one-go post-capture flow
(tens of seconds), **not** realtime; and a single phone camera only
determines motion in the throw plane, which is embedded in the 3-D scene —
no depth is claimed. The replay is labelled "fitted model", never exact
motion.

## Quickstart

```sh
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env   # fill in MODEL_API_KEY (same key serves LLMs + SAM 3.1)
pixelphys run --help
pytest -q
```

On a slow connection, install packages smallest-first (`meta-sam-parser`,
`numpy`, …) rather than one bulk install. Scripts that need no API key run
with `PYTHONPATH=src python3`.
