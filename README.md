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

## Current state (Day 3 done, Day 4 in progress)

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
  (`results/day4_diag.json`, 1 verification run per mode). The real-clip
  ablation is Day-5 work.

Not built yet: the one-command `pixelphys run …` wiring, the report and
ghost-overlay renderer, the ablation table, and the demo video.

## Seeing the equation (report visual, lands with the renderer)

The intended main output of a run is a single graph, not a table of numbers.
On the same axes it will show the measured trajectory as points, the baseline
model's curve, and the final model's curve continued (dashed) into motion
the fit never saw — with the discovered equation, its fitted coefficients,
and both models' held-out NRMSE written on the graph itself. If the final
curve peels away from the held-out points, the graph says so; that is the
validation, not a decoration. (The residual plots Spark reads during a run
already exist; the combined equation graph is part of the Day-6 report.)

## Still planned (post-freeze)

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
