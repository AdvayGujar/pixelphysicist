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
  → held-out prediction → overlay video + report
```

The numeric fitting is classical. Spark's contribution is proposing and
revising model *structure* from residual evidence; an ablation with a negative
control measures exactly that contribution. The pipeline handles one 1-DOF
coordinate, one object, one static camera.

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
