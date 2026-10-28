# Pixel Physicist

Film a moving object with a phone. SAM 3.1 tracks it, Muse Spark acts as the
scientist: it reads the residual evidence, diagnoses why the first model fails,
proposes the missing physics, and the system validates the result by predicting
motion it never saw.

## How it works

```
phone clip
  → tracking (SAM 3.1 video, HSV color fallback)
  → calibration to a 1-DOF coordinate q(t)
  → classical fit of a default physical model
  → Spark loop: read residual plots, propose the missing term, refit
  → held-out prediction → equation graph + overlay video + report
```

The numeric fitting is classical. Spark's contribution is proposing and
revising model *structure* from residual evidence; an ablation with a negative
control measures exactly that contribution. The pipeline handles one 1-DOF
coordinate, one object, one static camera.

## Seeing the equation

The main output of a run is a single graph, not a table of numbers. On the
same axes it shows the measured trajectory as points, the baseline model's
curve, and the final model's curve continued (dashed) into motion the fit
never saw — with the discovered equation, its fitted coefficients, and both
models' held-out NRMSE written on the graph itself. If the final curve peels
away from the held-out points, the graph says so; that is the validation, not
a decoration. The same graph is embedded in each run's `report.html`.

## Planned extension (post-freeze, not yet built)

After the core pipeline is frozen, a one-go projectile mode is planned: a
finished throw clip goes in, and a single run produces the fitted projectile
equations for both axes, the equation graph, the report, and a 3-D replay in
which a rendered ball re-flies the *fitted* equations (the replay is driven
by the exported model, not by the tracked points, and is labelled "fitted
model"). Two honest limits are part of the design: SAM 3.1 is a batch API,
so this is a one-go post-capture flow (tens of seconds), **not** realtime;
and a single phone camera only determines motion in the throw plane, which
is embedded in the 3-D scene — no depth is claimed.

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
