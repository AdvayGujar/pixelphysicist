# System architecture

Pipeline (one command: `pixelphys run <clip> --prompt ...`):

```
clip -> tracking -> calibration/coordinates -> trajectory cleaning
  -> baseline fit -> Spark loop -> held-out scoring
  -> equation graph / ghost overlay / replay / report.html -> runs/<run_id>/
```

1-DOF default; `--mode projectile` runs the coupled 2-DOF path;
`--mode spring/rolling` change only the coordinate and the prompts.

## Module map (`src/pixelphys/*.py`)

| module | responsibility | key public functions |
|---|---|---|
| `config.py` | settings, paths, model ids; fails fast on missing key | `load_settings`, `load_dotenv`; `SAM_MODEL_ID="sam-3.1"`, `SPARK_MODEL_ID="muse-spark-1.3"`, `DEFAULT_BASE_URL=https://api.meta.ai/v1` |
| `meta.py` | Meta API client, retries, JSONL cost log (no key logged) | `chat`, `log_call`, `compute_cost_usd`, `get_client` |
| `tracking.py` | sensors: SAM video, HSV color, frame-difference motion | `track_clip`, `upload_clip`, `track_video`, `track_color_clip`, `track_motion_clip`, `observations_from_records`, `clip_fps` |
| `calibrate.py` | scale/offset: px pass through, `/scale` to metres | `to_meters`, `describe_units` |
| `trajectory.py` | clean, resample, coordinate, split, trim | `clean_observations`, `split_by_gaps`, `resample_uniform`, `smooth_position`, `trim_free_flight`, `vertical_q`, `rest_relative_q`, `pendulum_angle`, `save_observations`, `load_observations` |
| `expr.py` | safe term grammar (AST whitelist, NumPy namespace) | `validate_terms`, `compile_term`, `basis_matrix`, `*_2d` variants; max 8 terms, 64 chars |
| `physics.py` | weak-form init + shooting fit + NRMSE/BIC scoring | `fit_terms`, `score_candidate`, `simulate`, `rollout_score`, `weak_init`, `fit_projectile`, `score_projectile`, `nrmse`, `bic`; `BASELINE_TERMS=[1,q,v,v**2]` |
| `scientist.py` | 3-tool Spark loop, both modes, submit guards | `ScientistTools`, `ProjectileTools`, `run_loop`, `default_chat_fn` |
| `synthetic.py` | known-truth renderers (toss, 2-D projectile, spring, rolling) | `render_clip`, `render_spring`, `render_rolling`, `integrate_toss/spring/rolling` |
| `render.py` | equation graph, ghost overlay, replay, report | `plot_equation`, `render_equation`, `render_overlay`, `render_replay_2d`, `render_replay_pixels`, `build_report` |
| `cli.py` | argparse + orchestration, plausibility gate, 1-D/2-D paths | `main`, `track_observations`, `to_coordinate`, `run_1d`, `run_projectile`, `projectile_coordinates` |
| `__init__.py` | package marker | — |

## The two Meta models

One family, one base URL + Bearer key (`MODEL_API_KEY`), two surfaces:

- **SAM 3.1 (sensor).** `POST /v1/responses` on the base URL (streaming
  SSE), parsed with `meta_sam_parser` (`parse_responses_stream`,
  `video_segmentation_format`, `decode_mask_to_raster`; mask encoding
  `one_bit`, `include_confidence`). Clips arrive via the **Files API**
  (`client.files.create`, purpose `user_data`) in `tracking.upload_clip`;
  `track_video` sends `{input_text prompt, input_video file_id}` and maps
  box+mask records to `Observation`s (most-present object wins).
- **Muse Spark (scientist).** Chat Completions via `meta.chat`
  (`client.chat.completions.create`), **`tool_choice="auto"` only**,
  `max_tokens=2048`, `reasoning_effort="low"` while tuning. Cost
  $1.25/$4.25 per Mtok in/out. Three tools: `fit_model` (structure in,
  fitted+scored model out), `plot_residuals` (fit-window residuals vs
  t/q/v as an image), `submit_model` (guard: beat baseline by >=2%
  held-NRMSE **or** BIC under v1; **and** + train-within-5% under v2).

## Trackers

- `sam_video` (primary): SAM 3.1 video segmentation, best quality, billed
  per frame (~$0.20/1k; CLI warns past 1500 frames).
- `color` (offline fallback): HSV largest-contour per frame; two presets,
  tennis-green `(25,40,40)-(85,255,60)`-ish and synthetic-dark
  `(0,0,0)-(180,255,60)`. Fails on clutter (locks onto furniture).
- `motion` (offline fallback): median-background frame differencing for
  static cameras; keeps the contour nearest the previous centroid.
- `--tracker auto`: SAM first, then the first *plausible* offline track
  (color, then motion). Plausibility gate (`cli._track_plausible`):
  >=10 obs, median step 2-80 px, spatial span > 40 px, jump share <= 0.2.
  Implausible tracks warn but still run rather than abort.

Coordinates: axis `y` = vertical toss (rows flipped so gravity is
negative), pivot = pendulum angle, `spring` = displacement from median
(up positive), `rolling` = distance along image-x from window start,
projectile = origin at window start, y up (raw pixels would flip
gravity's sign and poison the constant term).

## Data flow of a run

`pixelphys run` writes `runs/<timestamp>_<clip-stem>/`:

```
config.json        # clip, prompt, tracker, mode, axis, trim, scale, effort
observations.json  # cleaned tracked centroids
model.json         # baseline + gravity-only control + final terms/coeffs/scores
transcript.json    # Spark rounds, tool calls, costs, stop reason
plots/equation.png # measured + baseline + control + final (solid fit, dashed held)
plots/residual_*.png
overlay.mp4        # ghost prediction over real footage (blanked outside fit span)
replay*.mp4        # projectile: re-flown fitted equations, labelled fitted-model
report.html        # self-contained (base64 figures): equation, model table, cost
```

Plus the append-only cost log `runs/cost.jsonl` (every API call).
`scripts/`: `run_baseline.py`, `run_day4.py`, `run_ablation.py`,
`transfer_check.py`, `track_batch.py`, `plot_qt.py`, `smoke_*`.

## Cost model and guards

Per-loop: `MAX_ROUNDS=6`, `COST_CAP_USD=0.50`, `MAX_FIT_CALLS=8`,
loop fit budget 15 s, screen budget 5 s (one weak-init start, loose
tolerances; only submitted/best models pay for the tight refit),
`SUBMIT_IMPROVEMENT=0.02`. Per-fit: wall-clock budget (default 60 s),
`SIM_TIMEOUT_S=5` via SIGALRM against stiff-solver stalls,
`sign(...)` terms use BDF-only integration (Day-5 hang fix). Failures
(exploding rollout, exhausted budget, bad terms) return `ok=False` with
inf metrics and an error string — data, never a crash — and the CLI
refuses cleanly (exit 2, partial run kept) on missing key, short tracks,
or gaps. The gravity-only control is scored and drawn in every run so a
win over baseline can never hide a control win.

## Test strategy

- **Offline fixtures** (`tests/fixtures/*.json`): tracked observations and
  truth arrays; no API in tests. Gate scripts read fixtures, never live video.
- **Exact-coefficient identity tests**: synthetic truth recovery is exact —
  toss drag (`test_physics`), projectile coupling (`test_projectile`),
  Duffing cubic stiffness and Coulomb rolling friction
  (`test_spring_rolling`), expression grammar (`test_expr`), CLI
  coordinate/gate behavior (`test_cli`), rendering (`test_render`).
- **Slow markers**: full shooting fits are `@pytest.mark.slow`
  (`pyproject.toml`: skip with `-m 'not slow'`); the fast inner loop runs
  without them. API-free scripts run with `PYTHONPATH=src python3`.
