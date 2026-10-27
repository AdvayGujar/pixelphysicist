# Pixel Physicist

Film a moving object with a phone. SAM 3.1 tracks it, Muse Spark acts as the
scientist: it reads the residual evidence, diagnoses why the first model fails,
proposes the missing physics, and the system validates by predicting motion it
never saw.

See `plan.md` (v2, current) for the 8-day build plan. `plan-v1.md` is archived.

## Status

Module 0 (repo foundation) only. `pixelphys run` skeleton exists; tracking,
fitting, and the Spark loop land in later modules.

## Quickstart

```sh
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env   # fill in MODEL_API_KEY
pixelphys run --help
pytest -q
```

## Layout

See plan.md Section 8 for the 11-module structure under `src/pixelphys/`.
