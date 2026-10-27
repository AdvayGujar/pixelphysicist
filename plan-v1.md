# Pixel Physicist: Plan

**One-liner:** Film a real moving object with a phone. SAM 3.1 tracks it, and Muse Spark acts as the scientist: it diagnoses why a first model fails, proposes the missing physics, and the system validates the discovered equation by predicting held-out motion.

**Event:** Meta Global AI Developer Hackathon, track *Push the Frontier*. Solo.
**Build window:** Tue Oct 27 to Tue Nov 3, 2026. Day 7 is Mon Nov 2, and Nov 3 is buffer and submission. **Check the exact submission cutoff time (and timezone, convert to IST) on Day 1.**
**Budget:** $150 Meta Model API credits. Hardware: a 16GB Mac, CPU only.

---

## 1. Why this project

- **New capability:** going from raw footage to a validated physical law, where the model diagnoses *why* a candidate law is wrong and fixes its structure, instead of only running regression.
- **Track fit:** "accelerate a scientific experiment" and "a better way to understand a physical system", with no lab hardware.
- **Meta models are central, not decorative:**
  - SAM 3.1 is the sensor: text-prompted object tracking in video.
  - Muse Spark is the researcher: it reads residual plots, proposes new model terms, and judges overlays.
- **Honest weak point to defend:** a classical fitter does the numeric fitting. Spark's value is *model-structure discovery*. The ablation (Section 8) exists to prove this with numbers.

---

## 2. Verified facts, assumptions, unknowns

Sourced from third-party docs and blogs on Oct 26. **Confirm on Day 1 in the dashboard and docs.**

| Item | What I found | Status |
|---|---|---|
| API base URL | `https://api.meta.ai/v1`, bearer key in `MODEL_API_KEY`, OpenAI-compatible | Verify; the OpenAI SDK will not read `MODEL_API_KEY` automatically, so pass `api_key` explicitly |
| Default model | `muse-spark-1.3`: 1M context, tool calling, structured output, image/video/PDF input | Verify |
| Spark pricing (Standard) | $1.25 in / $0.15 cached in / $4.25 out per 1M tokens. Reasoning tokens bill as output | Verify |
| Contributor mode | `muse-spark-1.3-contributor`: $0.10 in / $0.20 out, but Meta may train on your prompts and outputs | Optional for prompt tuning only (see Section 9) |
| Muse Wonder | `muse-wonder-1.0` is a frontier agent text model, $4 / $20 per 1M, not an image model | Optional escalation only |
| SAM 3.1 | $2.50 per 1,000 images, $0.20 per 1,000 frames; **minimum 50 frames billed per video request**; rate limit 3,000 RPM | Verified: Responses API, MP4/MOV H.264/H.265/MPEG-4, up to 15,000 frames/clip, up to 16 objects/frame |
| Spark limitations | No logprobs; `seed` best-effort; **`tool_choice` accepts only `"auto"`** (others → 400); `reasoning_effort="none"` → 400 (Muse Spark always reasons); context 1,048,576 tokens, max output 131,072 tokens | Verified for 1.3 |
| SAM 3.1 video request format | Responses API `POST /v1/responses`, `model=sam-3.1`, `input_text` concept prompt + `input_video` (URL or `file_id`); stream recommended; up to 15,000 frames, 16 objects/frame; formats MP4/MOV H.264/H.265/MPEG-4 | Resolved (docs) |
| **Spark token cost per image / per video frame** | **Unknown** | **Day 1 blocker test** |
| Reasoning effort parameter | Accepted: `minimal`, `low`, `medium`, `high`, `xhigh`; `none` not supported (→400); `max` level on Standard tier only (verify exact mapping on Day 1) | Verified enum |

Not using Muse Image, Muse Voice or the open-weight models. Adding them would be decorative.

---

## 3. Scope

### MVP (must work to submit)
1. A CLI `pixelphys run <clip> --prompt "<object>"` that does: track, calibrate, baseline fit, Spark scientist loop, prediction, overlay video, report.
2. At least **3 real clips** with different hidden physics, and **synthetic clips with known ground truth** for benchmarking.
3. One 1-DOF generalized coordinate per clip: pendulum angle, or position along a line.
4. An ablation table: baseline versus baseline plus Spark, measured by held-out prediction error.
5. A public repo with a README and a demo video of 60 to 90 seconds.

### Stretch (only after the feature freeze gate passes)
- 2-DOF projectile (x and y) with drag.
- Bouncing ball as a hybrid system: fit flights separately, estimate restitution from impacts.
- Text-only versus text-plus-plots Spark ablation, to show that vision matters.
- Bootstrap confidence intervals on the parameters.

### Non-goals
3D motion, moving cameras, multiple interacting objects, a web app, real-time processing, training any model.

---

## 4. Pipeline architecture

```
phone clip --> [ffmpeg frames] --> [Tracker backend] --> pixel trajectory (t, px, py)
                                       |                       |
                      SAM 3.1 video / SAM per-frame /          v
                      color tracker / synthetic          [Calibrate: px -> metres]
                                                               |
                                                               v
                                                  [Trajectory clean: smooth, 1-DOF q(t)]
                                                               |
                                    split: fit window (~60%)   |   held-out window (~40%)
                                                               v
                                         [Baseline fit: default term library]
                                                               |
                                                               v
                              +--------- [Scientist loop (Muse Spark, tool use)] ---------+
                              |  tools: fit_model, plot_residuals, compare_overlay,       |
                              |         submit_model                                       |
                              |  guards: safe-expression grammar, BIC penalty,             |
                              |          max rounds, per-run cost cap                      |
                              +-----------------------------+------------------------------+
                                                            v
                                          [Predict held-out window + new clip]
                                                            v
                      [Overlay renderer: ghost prediction over real footage] + [Spark judge] + [Report]
```

### Backend interface (so a Day-1 surprise never breaks the project)
`Tracker.track(video_path, prompt) -> list[Observation(t, cx, cy, bbox, mask_area)]`

Backends, in priority order:
1. `sam_video`: SAM 3.1 video segmentation (preferred).
2. `sam_frames`: SAM 3.1 per-image segmentation on downsampled frames.
3. `color`: HSV threshold tracker (guaranteed fallback for a high-contrast object).
4. `synthetic`: renders a known system with noise (ground truth for tests and ablations).

---

## 5. Component design

### 5.1 Capture protocol (filmed on Day 1, not before the event)
- Phone on a stack of books or tripod, camera perpendicular to the motion plane, locked exposure and focus, 60 fps if available, otherwise 30 fps.
- Plain contrasting background, and the object a saturated colour.
- A reference object of known size in the plane of motion (an A4 sheet or a ruler) for pixel-to-metre scaling.
- Measure ground-truth quantities with a ruler: pendulum length, slope angle.
- 3 takes per scenario with different initial conditions: one to fit, one held out.

### 5.2 Calibration
- Segment the reference object ("the A4 paper") with SAM 3.1 via a concept prompt, take its long-side pixel length, and compute the scale. Request `mask_encoding: "one_bit"` to keep payloads small. Manual two-click scale as a backup.
- Pendulum: pivot position from a manual click or a segmented marker, giving θ(t) = atan2 of bob relative to pivot.
- Report the scale uncertainty.

### 5.3 Trajectory cleaning
- Drop frames where the mask area jumps (tracking glitches), interpolate short gaps, apply Savitzky–Golay or spline smoothing.
- Choose the generalized coordinate q and output (t, q) plus velocity estimates for plotting only.

### 5.4 Model representation
Second-order 1-DOF system: **q'' = Σ cᵢ · φᵢ(q, v)** where each φᵢ is a term string.
- **Safe expression grammar:** parse with `sympy`/`ast` using a whitelist: `q, v`, numbers, `+ - * / **`, and `sin cos tanh sign abs sqrt exp`. Reject anything else, cap the length, no attribute access. Spark's expressions are never `eval`'d raw.
- **Default (baseline) library:** `1, q, v, q², v², sin(q), cos(q)`. Deliberately excludes `q·v`, `|v|·v` and `sign(v)`, so the hidden physics of the hero clips (quadratic drag, Coulomb friction) lives *outside* the baseline and Spark has something real to discover. `1` is kept so a constant gravitational component along a slope is captured; the spring-mass negative control still holds because that clip has no constant bias. If Day 3 shows a hero clip still not failing, shrink the library further (drop `v²`) rather than re-film, and never add a hero term to the baseline.
### 5.5 Fitting
1. **Initial coefficients:** weak-form or smoothed-derivative sparse regression (STLSQ, thresholding small coefficients).
2. **Refinement:** simulation-based least squares (`scipy.optimize.least_squares` with `solve_ivp`) over coefficients plus initial conditions (q0, v0), multi-start. With ≤5 terms each solve takes milliseconds on CPU.
3. **Selection:** score by held-out simulation RMSE (normalized by amplitude) plus a BIC-style penalty, so Spark cannot win by adding terms.

### 5.6 Scientist loop (Muse Spark)
**State shown to Spark each round:** a data summary, the current model (terms and coefficients), train and validation metrics, and attached images (trajectory, residual plots against q, v and t, ghost overlay frames).

**Tools** (all functions the model calls; remember `tool_choice` is `auto` only, so the system prompt must say "finish only by calling `submit_model`"):
| Tool | Purpose |
|---|---|
| `fit_model(terms: list[str], rationale: str)` | Fit a candidate term set, return metrics and a model id |
| `plot_residuals(model_id, versus: "q" or "v" or "t")` | Return a residual plot image |
| `compare_overlay(model_id)` | Return a frame strip with the predicted ghost over real frames |
| `read_plot(model_id, versus)` | Visual subagent: return a natural-language reading of a residual plot (KeplerAgent pattern) |
| `submit_model(model_id, explanation: str)` | Final answer, structured output with a JSON schema |

**Loop rules:** at most 8 rounds, stop early if validation error improves less than 2% for 2 rounds, per-run cost cap (see Section 9), a static system-prompt prefix to maximize cache hits, and every API call written to a JSONL cost log.

**Prompt design notes:** teach Spark a short residual-reading cheat sheet (residual proportional to v suggests viscous drag, to |v|v quadratic drag, to sign(v) Coulomb friction, to a sinusoid in t a driving force, a trend in q a wrong restoring term). It must justify each added term physically and name the quantity the coefficient represents.

### 5.7 Spark judge (qualitative)
Send a frame strip or short overlay video and ask "does the ghost track the real object, and where does it diverge?" The verdict is shown in the report but **never replaces the numeric validation metric.**

### 5.8 Outputs
- The equation (rendered cleanly) with coefficients and units.
- Held-out prediction RMSE and the overlay mp4: real footage with a ghost trajectory of the prediction.
- A static HTML report per run (the equation, the plots, Spark's reasoning transcript, the cost).
- A results CSV for the ablation.

---

## 6. Experiment set

| # | Scenario | Hidden structure (expected) | Role |
|---|---|---|---|
| 1 | Pendulum, small amplitude | θ'' = -(g/L) sin θ - cθ' | Sanity check; recover g from measured L |
| 2 | Light ball or paper ball **tossed upward** (up and down) | y'' = -g - k\|v\|v (quadratic drag, bidirectional) | Candidate "hero"; toss rather than drop so the unsigned `v²` baseline term cannot accidentally fit one-direction motion |
| 3 | Ball rolling on cloth or towel slope | constant gravity component along slope (captured by baseline `1`), plus **Coulomb friction `sign(v)` (not in baseline)** as the hero hidden term | Candidate "hero" |
| 4 | Vertical spring-mass | -(k/m)(y - y0) - cv - g | Linear baseline wins; shows the system doesn't hallucinate extra terms |
| 5 (stretch) | Bouncing ball | Flights with drag plus restitution at impacts | Hybrid system |
| S1–S4 | Synthetic versions with known parameters, noise levels, frame rates | Ground truth | Benchmarks, ablations, fallback |

The demo hero is whichever of #2 or #3 shows the clearest baseline failure followed by a clean Spark fix. **Do not promise which term Spark finds until measured.**

---

## 7. Success metrics

- **Parameter recovery** on synthetic clips (relative error of true coefficients).
- **Held-out prediction error:** RMSE over the last 40% of the clip, and on a separate take with different initial conditions.
- **Spark rounds to success** and **cost per run.**
- **Term correctness:** does the final structure match the ground-truth terms on synthetic clips?
- **Negative control:** on the clean spring-mass clip, Spark should *not* add spurious terms.

---

## 8. Ablation design

- **Condition A (baseline):** default library, sparse fit plus shooting refinement, no Spark.
- **Condition B (Spark):** the same starting point plus the scientist loop.
- Run **5 repeats per clip for condition B** (the LLM is non-deterministic and seed is best-effort). Condition A is deterministic given data, so 1 run per clip suffices; condition C (stretch) also 5. Report mean and spread for B/C, single values for A.
- Report: held-out RMSE, number of terms, correct-structure rate (synthetic only), cost.
- **Stretch condition C:** Spark with text-only residual statistics (no images) to show whether vision matters.
- **Success criterion for the claim "Spark matters":** B beats A on held-out error on at least 2 clips, and B ≈ A on the negative control.

---

## 9. Budget

Assumptions (token costs for images and video are unknown until Day 1): about 30 to 40K input tokens per scientist turn, mostly cached after the first turn, and about 3 to 4K output tokens including reasoning.

| Item | Estimate |
|---|---|
| One scientist run (about 8 turns) | $0.20 to $0.75 |
| Development and prompt tuning (about 60 to 100 runs) | $20 to $50 |
| Ablation (about 3 clips x 2 conditions x 5 repeats; only B costs) | $3 to $8 |
| SAM 3.1 (one request per clip; billed frames = max(50, clip frames); ~15K frames total at $0.20 per 1,000) | about $3 | Note: minimum 50 frames per video request, so never split a clip into many small requests |
| Judge calls, demo takes, buffer | $5 to $10 |
| **Total** | **about $35 to $75, reserve the rest** |

**Guards:** a per-run cap of $0.75, a global alert at $60, and a cost logger that records every call. Optionally use the Contributor model for early prompt iteration only if you're comfortable with Meta training on those prompts (never include anything private; footage of your own home is the main thing to consider).

---

## 10. Day-by-day schedule

Each day ends with a **gate**. If a gate slips by more than 3 hours, execute the fallback on the spot.

### Day 1, Tue Oct 27: Foundations and the blocker tests
- Public repo, `uv` or venv, `.env.example`, key not committed. Commit early and often (the commit history shows work done during the event).
- `meta_client.py` wrapper and a cost logger.
- **Smoke tests:** Spark text, tool calling and structured output; Spark with an image; Spark with a short video; SAM image; SAM video (via Responses API, parse with `meta_sam_parser`). Record token usage per input type, billed SAM frames (`usage.video_frames_processed`), latency and rate limits.
- Film the real clips (Section 6, #1 to #4) and render the first synthetic clip.
- **Gate G1:** at least one real clip yields a usable trajectory from *some* backend, and the SAM video and per-image paths are characterized. **Fallback:** color tracker plus synthetic data, with SAM used for the calibration object. If SAM is unusable entirely, rethink the SAM part of the pitch tonight rather than on Day 4.

### Day 2, Wed Oct 28: Perception
- Tracker interface and the 4 backends, calibration, cleaning, and unit tests on synthetic data.
- **Gate G2:** pendulum clip yields a trajectory whose period is within 2% of a manual measurement. Calibrated scale looks sane (g recovered within about 10% later is the downstream check).

### Day 3, Thu Oct 29: Modeling and baseline
- Expression grammar, simulator, sparse fit, shooting refinement, held-out validation and metrics.
- Run the baseline across all clips; identify which fail.
- **Gate G3:** baseline recovers the synthetic ground truth and **visibly fails on at least one real clip** (otherwise there is no story; re-film or choose a different scenario).

### Day 4, Fri Oct 30: Scientist loop v1
- Tools, prompts, cost cap, structured final output, residual and overlay plots.
- **Gate G4:** on a synthetic system with a hidden extra term, Spark finds it within 6 rounds in at least 4 of 5 runs. **Fallback:** simplify the loop (fewer tools, fixed term menu plus free-form proposals) rather than adding features.

### Day 5, Sat Oct 31: Real hard clips, ablation, judge
- Tune prompts on real clips, add the Spark judge, run the full ablation into `results/*.csv`.
- **Gate G5 / FEATURE FREEZE (end of day):** B beats A on at least 2 clips and the negative control is clean. No new features after this.

### Day 6, Sun Nov 1: Polish and generalization
- Overlay renderer, HTML report, CLI polish, README, tests, repo cleanup.
- Run an unseen clip end to end with no code changes (the generalization shot for the demo).
- Dry-run the demo script and re-film any weak footage.

### Day 7, Mon Nov 2: Demo video
- Record the screen, narrate, edit, upload.
- Final README pass and a clean-clone test (fresh clone, install, run on a bundled clip).

### Nov 3, Tue: Buffer and submission
- Submit early, well before the cutoff (check IST conversion). Only fix bugs or re-record, nothing new.

---

## 11. Demo storyline (60 to 90 seconds)

| Time | Shot |
|---|---|
| 0:00–0:08 | Hook: a real clip of the hero object, one line of narration: "Can a model discover the physics of this from video alone?" |
| 0:08–0:20 | SAM tracks the object; the trajectory and calibrated plot appear |
| 0:20–0:32 | The baseline fit runs, and the predicted ghost visibly drifts from reality (error number shown) |
| 0:32–0:55 | Spark inspects the residual plot; its reasoning is shown as a one-line diagnosis ("error grows with speed squared, suggesting quadratic drag"); it refits and the error collapses |
| 0:55–1:10 | The final equation with units; the ghost prediction of the held-out seconds now tracks the real footage |
| 1:10–1:25 | An unseen clip, no code changes, same pipeline, new equation. A closing card with the ablation table and per-run cost |

**Recording notes:** use a clean terminal (`rich` live panel) and the overlay mp4; keep narration to a script written on Day 6; show only what was built during the event; check music licensing or use none.

---

## 12. Repo structure

```
pixelphys/
  README.md  plan.md  LICENSE  pyproject.toml  .env.example
  src/pixelphys/
    cli.py  config.py  meta_client.py  costlog.py
    tracking/   base.py  sam_video.py  sam_frames.py  color.py  synthetic.py
    calibrate.py  trajectory.py
    models/     expr.py  library.py  fit.py  simulate.py  metrics.py
    scientist/  loop.py  tools.py  prompts.py  schemas.py  judge.py
    render/     overlay.py  report.py
  experiments/  ablation.py
  data/         clips/  synthetic/
  results/      ablation.csv  figures/
  runs/         (gitignored)
  tests/
```

**Stack:** Python 3.11+, `openai` SDK (base URL override) for Muse Spark; SAM 3.1 goes through the same SDK's Responses API; `meta-sam-parser` (Python, ≥3.10, zero deps) to parse SAM output and decode masks; `numpy`, `scipy`, `sympy`, `opencv-python`, `imageio-ffmpeg`, `matplotlib`, `rich`, `pytest`. (No `httpx` needed; SAM uses the OpenAI SDK.)

---

## 13. Risk register

| Risk | Likelihood | Impact | Mitigation and fallback |
|---|---|---|---|
| SAM video API unusable or opaque | Medium | High | Per-frame SAM, then color tracker; SAM still does calibration-object segmentation |
| Real-footage noise breaks the fit | Medium | Medium | Better filming protocol, calibration checks, synthetic clips as the benchmark backbone |
| Baseline does not visibly fail on any real clip | Medium | High | Re-film with a lighter or faster object; adjust which terms are in the default library (honestly) |
| Spark proposes unphysical or overfit terms | Medium | Medium | BIC penalty, held-out scoring, round cap, physical-justification requirement |
| Spark ignores `submit_model` or loops | Medium | Low | Re-prompt on plain-text replies, hard round cap, parse structured output defensively |
| Image or video token cost higher than assumed | Medium | Medium | Downscale images, send fewer frames, rely more on numeric summaries |
| Non-determinism makes the demo run differ from rehearsal | Medium | Medium | Record the demo from a saved good run; report ablation mean and spread honestly |
| Scope creep into stretch goals | High | High | Hard feature freeze at the end of Day 5 |
| API outage or rate limits near the deadline | Low | High | Cache every response; keep the recorded runs; submit early |

---

## 14. Rules and compliance checklist

- [ ] All project code written during Oct 27 to Nov 3 (no pre-written project code); third-party libraries are fine.
- [ ] Footage self-filmed during the event; no third-party video, music or assets without rights.
- [ ] Repo is public by submission; license added; no API keys in history.
- [ ] README states clearly which Meta models are used and what each does (Muse Spark and SAM 3.1).
- [ ] Demo video shows only what was built during the event.
- [ ] No banned category: this is not a medical, RAG, chat-with-your-data, or simple image-analysis project.
- [ ] Submission cutoff time confirmed in IST.

---

## 15. Definition of done

1. Fresh clone, install and run produce a report and overlay on a bundled clip.
2. The ablation table exists with the repeats, and the README reports it honestly, including where Spark does *not* help.
3. The demo video is uploaded and under 90 seconds.
4. The README documents limits: planar 1-DOF motion, static perpendicular camera, a single rigid object, a classical fitter doing the numeric fitting.

---

## 16. Honest-limits statement (for the README and the Q&A)

- The numeric fitting is classical. Spark's contribution is proposing and revising the *structure* of the model from residual evidence, and the ablation measures that contribution.
- The Spark judge is qualitative. Validation is numeric, on held-out data.
- Results come from a small number of clips filmed in controlled conditions, not a general-purpose physics engine.

---

## 17. Detailed module and submodule design

This section expands Sections 4, 5 and 12 into an implementable module-level plan. It is the source of truth for file layout, public APIs, data structures and build order. Nothing here changes the scope or the day-by-day gates; it only makes them concrete.

### 17.0 Conventions

- **Python:** 3.11+. Type hints everywhere. `from __future__ import annotations` in every module.
- **Data structures:** standard `dataclasses` (frozen where immutable) for internal types; no extra validation libs. Pydantic is not a dependency.
- **Errors:** a small exception hierarchy in `pixelphys.errors` (Section 17.1.4). Never swallow; wrap at boundaries with context.
- **I/O boundary:** all filesystem, network and subprocess calls live in dedicated modules (`meta_client`, trackers, render). Business logic (`models`, `trajectory`, `scientist`) is pure-ish and testable with injected data.
- **Naming:** modules are nouns; functions are verbs; booleans are questions. Constants are `UPPER_SNAKE`.
- **No globals:** settings and clients are constructed and passed in. The only module-level state is immutable pricing tables.
- **Logging:** `logging.getLogger(__name__)` per module; `costlog` is the single place that writes the JSONL cost file.

### 17.1 Core modules (`src/pixelphys/`)

#### 17.1.1 `config.py` — configuration and environment

**Purpose:** load and validate all configuration from env/files; single source of truth for paths and model IDs.

**Public API:**
- `@dataclass(frozen=True) class Settings`: `api_base_url`, `model_api_key`, `default_model`, `sam_model`, `contributor_model`, `wonder_model`, `reasoning_effort` (default `"medium"`; use `"low"` for prompt tuning, `"high"` only for the final demo run), `data_dir`, `runs_dir`, `results_dir`, `cost_cap_per_run_usd`, `cost_alert_global_usd`, `max_scientist_rounds`, `cache_enabled`.
- `Settings.from_env() -> Settings`: reads `MODEL_API_KEY`, `META_API_BASE_URL` (default `https://api.meta.ai/v1`), optional overrides; resolves paths relative to repo root; raises `ConfigError` on missing key.
- `Settings.from_file(path) -> Settings`: for tests.
- Constants: `DEFAULT_MODEL = "muse-spark-1.3"`, `SAM_MODEL = "sam-3.1"` (verify exact ID Day 1), `CONTRIBUTOR_MODEL = "muse-spark-1.3-contributor"`, `WONDER_MODEL = "muse-wonder-1.0"`.

**Validation:** fail fast on missing key; warn (not fail) if model IDs look wrong. Paths are created lazily by callers, not at load time.

#### 17.1.2 `meta_client.py` — Meta Model API client

**Purpose:** one wrapper over the OpenAI SDK with base-URL override, explicit `api_key`, retries, rate-limit handling, and usage extraction. Used by scientist loop, judge, and (if needed) SAM helpers.

**Public API:**
- `class MetaClient`:
  - `__init__(settings, costlog)`.
  - `chat(messages, model=None, temperature=..., reasoning_effort=None, response_format=None) -> ChatResult` (Chat Completions).
  - `responses(input, model=None, tools=None, reasoning=None, previous_response_id=None, stream=False) -> ResponseResult` (Responses API; recommended for the scientist loop because reasoning carries across tool turns).
  - `chat_with_tools(messages, tools, model=None, tool_choice="auto") -> ToolChatResult`.
  - `vision(messages_with_images, ...) -> ...` (convenience; same as chat with image parts).
- `@dataclass ChatResult`: `content: str`, `tool_calls: list[ToolCall]`, `usage: Usage`, `raw`.
- `@dataclass ToolCall`: `id`, `name`, `arguments: dict`.
- `@dataclass Usage`: `input_tokens`, `cached_input_tokens`, `output_tokens`, `reasoning_tokens`, `latency_s`, `model`.

**Behaviour:**
- Pass `api_key=settings.model_api_key` explicitly to the SDK (it will not read `MODEL_API_KEY`).
- Retry on 429/5xx with exponential backoff (max 3 tries); respect `Retry-After`.
- Optional disk cache keyed by a hash of (model, messages, temperature, reasoning_effort, tool schema). Enabled by `PIXELPHYS_CACHE=1` (default on for dev/demo, off for ablation runs via `--no-cache`) so a recorded good run can be replayed offline and the demo is immune to API outages. Cache hits are still logged to the cost log with `cost_usd=0`.
- On every call, invoke `costlog.log_call(...)` and check the per-run/global caps; raise `CostCapExceeded` before the call if the run total would exceed the cap (estimate from last call if needed; hard-check after).
- Attach a `run_id` and `round` via context (set by scientist loop).
- Never log prompt text to the cost log; only metadata and token counts.

**Submodules:** none; single file. If it grows past ~300 lines, split into `meta_client/chat.py` and `meta_client/usage.py`.

#### 17.1.3 `costlog.py` — cost logging and aggregation

**Purpose:** append-only JSONL record of every API call; aggregation helpers for caps and reporting.

**Public API:**
- `class CostLogger`:
  - `__init__(path)`.
  - `log_call(call_id, run_id, round, model, endpoint, usage, clip_id=None) -> None`.
  - `run_total(run_id) -> float`.
  - `global_total() -> float`.
  - `calls_for_run(run_id) -> list[dict]`.
- `compute_cost(model, usage) -> float`: uses a pricing table keyed by model ID; reasoning tokens billed as output.
- Pricing table (verify Day 1): `muse-spark-1.3`: 1.25 / 0.15 cached / 4.25 per 1M; contributor: 0.10 / 0.20; wonder: 4 / 20; SAM: per-image/per-frame (tracked separately, not via this logger's token formula — SAM calls log `cost_usd` directly).

**File format:** one JSON object per line with keys `ts, call_id, run_id, round, model, endpoint, clip_id, input_tokens, cached_input_tokens, output_tokens, reasoning_tokens, latency_s, cost_usd`.

#### 17.1.4 `errors.py`

- `class PixelPhysError(Exception)`.
- Subclasses: `ConfigError`, `TrackingError`, `CalibrationError`, `ExpressionError`, `FitError`, `CostCapExceeded`, `ScientistError`, `RenderError`.

### 17.2 Tracking module (`src/pixelphys/tracking/`)

#### 17.2.1 `tracking/base.py` — interface and data model

**Data model:**
- `@dataclass Observation`: `frame_idx: int`, `t: float`, `cx: float`, `cy: float`, `bbox: tuple[float,float,float,float]`, `mask_area: float`, `confidence: float = 1.0`.

**Interface:**
- `class Tracker(Protocol)`:
  - `name: str`
  - `track(video_path: Path, prompt: str) -> list[Observation]`.
- `class TrackerFactory`:
  - `create(name: str, settings: Settings) -> Tracker` (names: `sam_video`, `sam_frames`, `color`, `synthetic`).
  - `available() -> list[str]`.

**Validation (`validate_observations`):** non-empty, strictly increasing `t`, finite coords, no negative areas; raises `TrackingError` otherwise.

#### 17.2.2 `tracking/sam_video.py` — SAM 3.1 video backend (preferred)

- Uses the Responses API with `model="sam-3.1"`, `input_text` = concept prompt, `input_video` = uploaded `file_id` (upload once via Files API; export MP4 with faststart). Stream the response; parse with `meta_sam_parser.parse_video_stream`; decode masks with `decode_mask_to_raster`; compute per-frame centroid from the mask raster (or box centre as a cheaper fallback).
- One request per clip (whole clip) so the 50-frame minimum is paid once, not per chunk. Up to 15,000 frames/clip, up to 16 objects/frame; track the largest mask as the target object.
- Set `metadata.mask_encoding="one_bit"` and `metadata.include_confidence="true"` (optional, for filtering low-confidence frames).
- `track(video_path, prompt) -> list[Observation]`: one Observation per frame with cx,cy from centroid, bbox from box record, mask_area from raster sum, t = frame_idx/fps.

#### 17.2.3 `tracking/sam_frames.py` — SAM per-image backend (fallback)

- Extract frames with ffmpeg at a configurable fps (default 10 to control cost), upload each as base64 `input_image` (or upload once and reuse), call SAM image endpoint per frame, parse with `meta_sam_parser.parse_image_stream`.
- Concurrency: small thread pool (≤4); respect the 3,000 RPM limit. Use this only if the video endpoint fails; it costs more (per-image pricing, no 50-frame minimum but $2.50/1k images).

#### 17.2.4 `tracking/color.py` — HSV threshold fallback

- `class ColorTracker(Tracker)`:
  - `__init__(hsv_lower, hsv_upper, min_area)`.
  - Per frame: HSV convert, inRange, morphological open, largest contour, centroid. Skip frames with no contour (record gap, do not crash).

#### 17.2.5 `tracking/synthetic.py` — synthetic renderer

- `class SyntheticTracker(Tracker)` is a misnomer; instead `render_synthetic(spec: SyntheticSpec) -> (video_path, ground_truth: Trajectory, metadata)`.
- `SyntheticSpec`: `ode: str` (term expression), `params: dict`, `q0, v0`, `duration, fps, resolution, noise_px, seed`.
- Integrates the true ODE, renders a moving dot on a background, writes mp4, returns ground-truth trajectory.
- Used by tests and ablation; also produces the S1–S4 clips.

### 17.3 `calibrate.py` — pixel-to-metre calibration

**Public API:**
- `@dataclass CalibrationResult`: `scale_px_per_m: float`, `pivot: tuple[float,float] | None`, `method: str`, `uncertainty_px: float`.
- `calibrate_scale_from_reference(frames, prompt="the A4 paper") -> CalibrationResult`: segment reference with SAM, long-side pixel length, scale = pixels / 0.297 m (A4 long side). Uncertainty from mask bbox jitter across frames.
- `manual_scale(p1, p2, real_length_m) -> CalibrationResult`.
- `pendulum_angle(observations, pivot, scale) -> list[tuple[t,q]]`: q = atan2(bob-pivot) with vertical reference; unwrap with `np.unwrap`.
- `line_position(observations, origin, axis_unit, scale) -> list[tuple[t,q]]`.

**Notes:** pivot can be a manual click or a segmented marker; store both. Report scale uncertainty in the report.

### 17.4 `trajectory.py` — cleaning and generalized coordinate

**Data model:**
- `@dataclass Trajectory`: `t: np.ndarray`, `q: np.ndarray`, `v: np.ndarray`, `kind: str` (`pendulum`/`line`/`projectile`), `calibration: CalibrationResult`, `source_clip: Path`, `fps: float`, `metadata: dict`.

**Public API:**
- `clean(observations, fps)`: (1) drop frames whose `mask_area` z-score exceeds 3 (glitches); (2) interpolate gaps ≤ `MAX_GAP_FRAMES` (5) onto a uniform time grid so the data is evenly sampled; (3) apply Savitzky–Golay smoothing on (cx,cy) on that uniform grid (window 7 or largest odd ≤ n). Smoothing before interpolation, or on the original irregular grid, is a bug — uniform resampling first.
- `to_trajectory(observations, calibration, kind) -> Trajectory`: choose q, compute v via central differences (for plotting only; fitting uses its own derivative handling).
- `split(traj, fit_frac=0.6) -> (Trajectory, Trajectory)`: split by time, not index count, to respect fps; return fit and held-out.
- `resample(traj, fps) -> Trajectory`.

**Guards:** require ≥ 30 points; require monotonic t.

### 17.5 Models module (`src/pixelphys/models/`)

#### 17.5.1 `models/expr.py` — safe expression grammar

**Purpose:** parse and validate term strings; the only place strings become math. Spark output is never `eval`'d.

**Grammar (whitelist):**
- Variables: `q`, `v`.
- Constants: numeric literals (int/float), including scientific notation.
- Operators: `+ - * / **` (right-associative for `**`), unary minus.
- Functions: `sin, cos, tanh, sign, abs, sqrt, exp`.
- Parentheses.
- Reject: attribute access, calls to anything else, names other than `q,v` and the function whitelist, comprehensions, lambdas, imports, comparisons, assignments, subscripts.

**Public API:**
- `ALLOWED_NAMES = {"q","v"}`, `ALLOWED_FUNCS = {...}`.
- `parse_term(s: str) -> sympy.Expr`: raises `ExpressionError` on violation; also rejects if length > `MAX_TERM_LEN` (64) or term count in an expression > cap.
- `validate_term(s) -> tuple[bool, str]`: non-raising, returns (ok, reason).
- `validate_terms(terms: list[str]) -> None`: raises on any bad term; dedups.
- `term_to_callable(expr) -> Callable[[float,float], float]`: lambdify with numpy; cached.
- `render_equation(terms, coeffs, q0=None, v0=None) -> str`: e.g. `q'' = -9.81*sin(q) - 0.34*v`.
- `MAX_TERM_LEN = 64`, `MAX_TERMS = 8`.

**Implementation:** use `ast.parse(s, mode="eval")`, walk the tree with a visitor that only accepts the whitelisted node types. Then convert to sympy via a small recursive builder (do not use `sympy.sympify` on raw strings — it can execute).

#### 17.5.2 `models/library.py` — term libraries

- `BASELINE_TERMS = ["1", "q", "v", "q**2", "v**2", "sin(q)", "cos(q)"]`.
- `CORE_TERMS = ["1", "q", "v", "sin(q)", "cos(q)"]` (used if baseline is still too strong; decide Day 3 by checking the Day-3 gate, not by guessing).
- `function_library_for_prompt() -> str`: human-readable list with descriptions, sent to Spark.
- No candidate generation beyond these; Spark proposes free-form terms within the grammar.

#### 17.5.3 `models/simulate.py` — ODE integration

- `def derivatives(t, state, terms, coeffs)`: state = [q, v]; returns [v, sum(c_i * phi_i(q,v))].
- `def integrate(terms, coeffs, q0, v0, t_eval, method="DOP853", rtol=1e-8, atol=1e-10) -> np.ndarray`: returns q_pred aligned with t_eval. Default to DOP853 for oscillatory motion; if the step count exceeds a threshold (stiff/discontinuous term such as Coulomb `sign(v)`), retry with `BDF`; raise `FitError` on NaN or non-finite output. Never silently truncate the integration interval.
- `def integrate_with_events(...)` for bouncing (stretch).

#### 17.5.4 `models/fit.py` — fitting

- `@dataclass FitResult`: `terms: list[str]`, `coeffs: np.ndarray`, `q0: float`, `v0: float`, `success: bool`, `rmse_train: float`, `rmse_val: float | None`, `bic: float`, `n_iter: int`, `message: str`.
- `def sparse_fit(terms_library, traj: Trajectory, threshold=0.05) -> FitResult`:
  - Build design matrix A where column i = phi_i(q, v) computed from smoothed q,v; target = smoothed q'' (central differences).
  - Solve least squares; threshold coefficients with |c| < threshold * max(|c|); iterate (STLSQ) 5 times or until support stabilizes.
  - q0,v0 from first point.
- `def shooting_refine(terms, traj, init: FitResult, n_starts=5) -> FitResult`:
  - Parameters = coeffs + q0 + v0. Residual = q_pred(t) - q_obs(t) over fit window.
  - `scipy.optimize.least_squares` with multi-start (perturb init), bounds on coeffs (±1e3) and q0/v0 (±10% of range).
  - Returns best by train RMSE.
- `def fit_model(terms, traj) -> FitResult`: sparse_fit then shooting_refine; if sparse yields empty support, fall back to all terms with zero init.
- `def score(result, val_traj) -> FitResult`: fills rmse_val, bic.

**BIC:** `n * log(rmse²) + k * log(n)` (standard; document). Used for selection among candidate models.

#### 17.5.5 `models/metrics.py`

- `rmse(pred, truth)`, `nrmse(pred, truth)` (divide by amplitude of truth).
- `bic(n, k, sse)`, `aic(...)`.
- `parameter_recovery(pred_coeffs, true_coeffs, term_names) -> dict`: relative errors matched by term.

### 17.6 Scientist module (`src/pixelphys/scientist/`)

#### 17.6.1 `scientist/schemas.py` — structured outputs

- `SUBMIT_MODEL_SCHEMA`: JSON schema for final answer: `{model_id, explanation, equation, terms: list[str], coefficients: list[float], units: str, added_terms_rationale: list[{term, physical_justification}]}`.
- `TOOL_SCHEMAS`: OpenAI function-calling schemas for `fit_model`, `plot_residuals`, `compare_overlay`, `submit_model`.
- `validate_submit(payload) -> SubmitModel` dataclass or raise.

#### 17.6.2 `scientist/tools.py` — tool implementations

- `class ScientistTools`:
  - Holds registries: `models: dict[model_id, FitResult]`, `traj_fit, traj_val`, `image_store` (paths).
  - `fit_model(terms, rationale) -> dict`: validate terms via `expr.validate_terms`; call `fit.fit_model` on fit traj; score on val; store; return `{model_id, terms, coeffs, rmse_train, rmse_val, bic, n_terms}`.
  - `plot_residuals(model_id, versus) -> str` (base64 PNG): compute residuals (observed - predicted) on fit window, scatter vs q/v/t, return image path + base64; save to run dir.
  - `compare_overlay(model_id) -> str`: render a strip of ~6 frames with predicted ghost overlaid; return image.
  - `read_plot(model_id, versus) -> str`: visual subagent reading of a residual plot (natural language).
  - `submit_model(model_id, explanation) -> SubmitModel`: validate model exists; **reject (return an error, do not finalize) if the model does not beat the baseline on validation RMSE by at least 2% or on BIC** — this stops Spark from submitting a spurious or worse model. On success, build the structured output and mark the loop done.
  - `registry` exposed for loop.
- Each tool returns a compact JSON-serializable dict; images are returned as `{path, base64}` and also attached to the next user message by the loop.

#### 17.6.3 `scientist/prompts.py` — prompts

- `SYSTEM_PROMPT`: static prefix (cache-friendly) containing:
  - Role: physicist assistant.
  - Residual cheat sheet: residual ∝ v → viscous drag; ∝ |v|v → quadratic drag; ∝ sign(v) → Coulomb friction; sinusoid in t → driving force; trend in q → wrong restoring term.
  - Tool usage: must finish by calling `submit_model`; never guess coefficients (call fit_model).
  - Physical justification requirement: name the quantity each coefficient represents and its units.
  - Constraints: max 8 terms, BIC penalty, no overfitting.
- `build_user_message(round, data_summary, current_model, metrics, image_paths, instruction) -> list[content parts]`.
- `DATA_SUMMARY_TEMPLATE`: fps, n points, amplitude, coordinate kind, calibration scale.

#### 17.6.4 `scientist/loop.py` — the scientist state machine

- `@dataclass ScientistResult`: `final_model: FitResult`, `submit: SubmitModel`, `rounds: int`, `transcript: list[dict]`, `cost_usd: float`, `stopped_reason: str`.
- `class ScientistLoop`:
  - `__init__(client, tools, settings, run_id)`.
  - `run(baseline: FitResult, traj_fit, traj_val) -> ScientistResult`:
    - Initialize messages with system + first user message (baseline results + plots).
    - Round loop ≤ `max_rounds`:
      - Call `client.chat_with_tools`.
      - Append assistant message + tool results to transcript.
      - Dispatch tool calls; collect image attachments for next turn.
      - If `submit_model` called → finalize, break.
      - Early stop: if val RMSE improved < 2% for 2 consecutive rounds → break with best model (submit it).
      - Cost cap check before each call.
    - If loop ends without submit → submit the best model so far with a note.
    - Return result.
  - `best_model()` tracking by val RMSE (then BIC).
- **Round cap:** 8. **Early-stop threshold:** 2% relative improvement. **Max tool calls per round:** no explicit cap, but tool results are appended.

**Transport:** prefer the Responses API with `previous_response_id` so reasoning carries across tool turns (stronger multi-step, fewer tokens); fall back to Chat Completions with manual history only if needed.

**Message construction:** include an experience log (tool, args, outcome) and a workspace file listing each round (KeplerAgent pattern). Images attached as base64 data URLs in user messages; keep a rolling window? For 1M context, send all; but to save cost, send current residual plot only plus the overlay strip when requested. Follow prompt design: state summary each round is small.

#### 17.6.5 `scientist/judge.py` — qualitative judge

- `def judge_overlay(overlay_path, client) -> JudgeVerdict`: `{tracks: bool, divergence_description: str, confidence: str}`.
- Sends the overlay frame strip + question. Result is shown in report, never used as a metric.

### 17.7 Render module (`src/pixelphys/render/`)

#### 17.7.1 `render/overlay.py`

- `def render_overlay(video_path, pred_traj, out_path, fps=None) -> Path`: for each frame, draw predicted ghost position (from pred_traj at that t) as a translucent circle; write mp4 via imageio-ffmpeg. Handle t alignment (nearest).
- `def render_frame_strip(video_path, pred_traj, n_frames=6) -> np.ndarray`: returns a horizontal strip image for Spark tools/judge.

#### 17.7.2 `render/report.py`

- `def build_report(run_dir, clip, traj, baseline, spark_result, judge_verdict, cost) -> Path`: writes `report.html` (self-contained, inline images as base64 or relative paths) with equation, plots, transcript, metrics, cost.
- Template stored as a Python string (no Jinja dependency) or a small HTML template file in `render/templates/`.

### 17.8 `experiments/ablation.py`

- `def run_ablation(clips, conditions, repeats, out_csv) -> pd?` — use csv module, no pandas dependency.
- Conditions: `A` (baseline only), `B` (baseline + scientist), `C` (text-only scientist, stretch).
- For each (clip, condition, repeat): run pipeline, record `{clip, condition, repeat, heldout_rmse, n_terms, structure_correct (synthetic), cost_usd, rounds}`.
- Aggregates mean/std to `results/ablation.csv` plus a summary JSON.
- Concurrency: sequential (cost caps); optional `--parallel N` later.

### 17.9 `cli.py` — command-line interface

- Framework: `argparse` (no click/typer dependency) or `typer` if already in stack. Plan says stack has no CLI lib; use `argparse` to avoid deps. (Could add `rich` for output, already in stack.)
- Commands:
  - `pixelphys run <clip> --prompt TEXT [--tracker NAME] [--out DIR] [--scale PX_PER_M] [--pivot X,Y] [--fps N]`. Manual `--scale`/`--pivot` bypass SAM-based calibration when the reference object is missed, so a bad calibration never blocks a run.
  - `pixelphys track <clip> --prompt TEXT --out traj.json`.
  - `pixelphys calibrate ...` (optional).
  - `pixelphys ablation --config experiments/ablation.yaml`.
  - `pixelphys report --run DIR`.
- `run` orchestrates: track → calibrate → clean → split → baseline fit → scientist → predict held-out + new clip → overlay → judge → report. Writes everything under `runs/<run_id>/`.

### 17.10 Cross-cutting `types.py` (optional)

If dataclasses scatter, collect shared ones (`Trajectory` already in trajectory.py; `Observation` in tracking/base.py). Keep them where they're used; do not create a giant types file unless duplication appears.

### 17.11 Error handling and guards (summary)

- Safe expression grammar (17.5.1) — no raw eval.
- BIC penalty in selection (17.5.4).
- Round cap 8, early stop 2%, per-run cap $0.75, global alert $60.
- Input validation at boundaries: observations, trajectories, terms.
- Cost cap checked before each API call.

### 17.12 Testing strategy

- `tests/test_expr.py`: whitelist acceptance/rejection cases, injection attempts.
- `tests/test_fit.py`: recover known params on synthetic linear/quadratic systems; thresholding.
- `tests/test_trajectory.py`: cleaning, split by time.
- `tests/test_costlog.py`: aggregation, cost formula.
- `tests/test_synthetic.py`: render + track round-trip.
- `tests/test_integration.py`: end-to-end on a synthetic clip (no API calls; scientist mocked or skipped behind env flag).
- `tests/test_meta_client.py`: cache keying, cap enforcement, retry behaviour using a fake transport (no network). `MetaClient` accepts an injectable transport for this.
- Run with `pytest -q`. Target: unit tests < 5s, integration < 60s.

### 17.13 Implementation order (maps to days)

| Day | Modules to land | Gate |
|---|---|---|
| D1 | config, meta_client, costlog, errors, synthetic renderer, smoke scripts | G1 |
| D2 | tracking/* (all 4), calibrate, trajectory, tests | G2 |
| D3 | models/expr, library, simulate, fit, metrics, baseline run | G3 |
| D4 | scientist/schemas, tools, prompts, loop, judge (basic) | G4 |
| D5 | render/overlay, render/report, ablation, prompt tuning on real clips | G5 |
| D6 | cli polish, tests, unseen-clip run | — |
| D7 | demo assets only | — |

### 17.14 Module dependency graph (allowed imports)

```
errors (none)
config (errors)
costlog (errors)
meta_client (config, costlog, errors)
tracking/* (config, errors)
calibrate (tracking.base, errors)
trajectory (tracking.base, calibrate, errors)
models/expr (errors)
models/library (models/expr)
models/simulate (models/expr)
models/fit (models/*, trajectory)
models/metrics (numpy)
scientist/schemas (models/expr)
scientist/tools (models/*, trajectory, render/overlay? no—tools return paths; render imported lazily)
scientist/prompts (models/library)
scientist/loop (meta_client, scientist/*, models/fit)
scientist/judge (meta_client, render/overlay)
render/overlay (trajectory)
render/report (models, trajectory, costlog)
experiments/ablation (cli-level orchestration, all above)
cli (all)
```

Rule: `models` and `trajectory` never import `scientist` or `render`; `scientist` may import `render` lazily to avoid cycles.

### 17.15 Caching and cost control

- System prompt is a static string (no per-run interpolation) to maximize prefix cache hits.
- Data summaries and metrics go in the user message after the cached prefix.
- Images are downscaled to ≤ 1024px on the long side before encoding; frame strips are downscaled to ≤ 1600px wide. Never send raw 4K phone frames to the API.
- Every call logged; per-run cap enforced in `meta_client`.

### 17.16 Configuration and env (`.env.example`)

```
MODEL_API_KEY=...
META_API_BASE_URL=https://api.meta.ai/v1
DEFAULT_MODEL=muse-spark-1.3
COST_CAP_PER_RUN_USD=0.75
COST_ALERT_GLOBAL_USD=60
```

### 17.17 Performance budget

- Per fit (≤8 terms, shooting refinement, 5 starts): < 200 ms on 16GB Mac CPU for 200 points.
- Trajectory cleaning: < 50 ms.
- Overlay render (60s @ 30fps): < 10 s.
- Scientist run: dominated by API latency; target < 90 s wall clock for 8 rounds.

### 17.18 Security and safety

- No `eval`/`exec` on Spark or user input; AST whitelist only.
- No shell command construction from user input; use subprocess with list args.
- API keys only via env; `.env` gitignored; `.env.example` committed.
- Cost caps prevent runaway spend.

---

## 18. Data models and on-disk layout

### 18.1 Run directory (`runs/<run_id>/`)

```
runs/20261027_143022_pendulum_take1/
  config.json          # settings snapshot + CLI args
  trajectory_fit.json  # (t,q,v) arrays
  trajectory_val.json
  calibration.json
  baseline.json        # FitResult
  scientist.json       # ScientistResult (transcript, final)
  plots/               # residual plots, overlay strip
  overlay.mp4
  report.html
  cost.jsonl           # per-run slice of the global cost log (copied)
```

`run_id` format: `YYYYMMDD_HHMMSS_<clip_stem>`.

### 18.2 JSON schemas (abbreviated)

- `trajectory_*.json`: `{"t": [...], "q": [...], "v": [...], "kind": "pendulum", "fps": 60.0, "calibration": {...}, "metadata": {...}}`.
- `baseline.json` / model entries: `{"terms": [...], "coeffs": [...], "q0": ..., "v0": ..., "rmse_train": ..., "rmse_val": ..., "bic": ..., "success": true}`.
- `scientist.json`: `{"final_model": {...}, "submit": {...}, "rounds": 6, "stopped_reason": "submitted", "transcript": [...], "cost_usd": 0.42}`.

### 18.3 Cost log (global)

`runs/costlog.jsonl` (or `results/costlog.jsonl`) — append-only, one line per API call across all runs, for budget monitoring.

---

## 19. Pipeline data flow (concrete)

```
CLI run(clip, prompt)
  -> tracker.track(clip, prompt) -> list[Observation]
  -> validate_observations
  -> calibrate (scale + pivot) -> CalibrationResult
  -> trajectory.clean -> observations_clean
  -> trajectory.to_trajectory(kind) -> Trajectory
  -> split(fit_frac=0.6) -> (fit, val)
  -> models.fit.fit_model(BASELINE_TERMS, fit) -> baseline
  -> score(baseline, val)
  -> scientist.loop.run(baseline, fit, val) -> spark_result
  -> predict: integrate(spark_result.final_model, val.t) -> q_pred
  -> metrics.nrmse(q_pred, val.q)
  -> render.overlay(video, pred, out) -> overlay.mp4
  -> judge.judge_overlay(overlay) -> verdict
  -> render.report(...) -> report.html
  -> print summary (equation, rmse, cost)
```

Each arrow's output is persisted to the run directory before the next step, so a crash can be resumed (manual resume is a Day-6 nicety; not required for MVP).

---

## 20. Risk-to-module mapping

| Risk (Section 13) | Primary module | Mitigation in code |
|---|---|---|
| SAM video unusable | tracking/sam_video, sam_frames | TrackerFactory + CLI `--tracker` |
| Footage noise breaks fit | trajectory.clean, calibrate | z-score drop, SG smooth, scale uncertainty |
| Baseline doesn't fail | models/library | switch BASELINE_TERMS to CORE_TERMS (honest, documented) |
| Spark overfits | models/metrics (BIC), loop (caps) | BIC in fit_model tool, round cap |
| Spark ignores submit | scientist/loop | re-prompt on text reply, hard cap, parse defensively |
| Token cost high | meta_client, prompts | downscale images, static prefix |
| Non-determinism | ablation (repeats) | 5 repeats, mean+spread |
| Scope creep | cli (feature set) | no stretch code until G5 passes |

---

## 21. Open questions to resolve on Day 1 (blockers)

1. Exact SAM 3.1 video request/response shape and limits → determines `sam_video.py`.
2. Spark token cost per image / per video frame → determines image downsampling and frame counts in tools.
3. `reasoning_effort` support on 1.3 and its exact enum values.
4. `tool_choice` on 1.3 (auto only?).
5. Rate limits and max request size.
6. Exact model ID strings in the dashboard.

Record answers in `docs/DAY1_FINDINGS.md` (create on Day 1) and update `config.py` constants accordingly.

---

## 22. Acceptance criteria for "done" (maps to Section 15)

1. `pip install -e .` then `pixelphys run data/clips/pendulum_take1.mp4 --prompt "the pendulum bob"` produces `runs/.../report.html` and `overlay.mp4` from a fresh clone.
2. `results/ablation.csv` has 3 clips × 2 conditions × 5 repeats with mean+spread reported in README.
3. Demo video < 90 s uploaded.
4. README documents limits (Section 16).
5. `pytest -q` green on a machine with no API key (API-dependent tests skipped via env flag).

---

## 23. Related systems and comparison

Deep research on existing work that overlaps with Pixel Physicist. Use this to position the project and to borrow proven ideas; do not re-invent what already exists.

### 23.1 Landscape

| System | Year | Input | Perception | Discovery method | LLM role | Validation | Open source |
|---|---|---|---|---|---|---|---|
| **SINDy** (Brunton et al.) | 2016 | Time series (x, ẋ) | None (needs derivatives) | Sparse regression over a candidate library (STLSQ) | None | In-sample fit | Yes (PySINDy) |
| **AI Feynman** (Udrescu & Tegmark) | 2020 | Tables of (x, y) | None | Recursive physics-inspired symbolic regression (symmetry, separability) | None | In-sample fit | Yes |
| **Visual Physics** (Chari et al.) | 2019 | Video (synthetic) | Bounding boxes + learned latent params | Symbolic regression over latent params + positions | None | In-sample fit | Paper only |
| **PySR** (Cranmer) | 2023 | Tables of (x, y) | None | Genetic-programming SR (expression trees) | None | In-sample / CV | Yes |
| **LLM-SR** (Shojaee et al., ICLR 2025) | 2025 | Tables of (x, y) + text | None | LLM proposes equation *skeletons*; evolutionary search + param fit | Equation guesser (skeletons) | In-domain + out-of-domain NMSE | Yes |
| **Physics-informed LLM-SR** (2026) | 2026 | Tables + images | None | SR with an LLM-based score in the loss | Scorer (prior injection) | In-sample metrics | Paper |
| **KeplerAgent** (Yang et al., 2026) | 2026 | Tables (x, ẋ) | None | LLM agent orchestrating symmetry discovery, a visual subagent, PySINDy and PySR; experience log + workspace | Agent (tool orchestration, template proposals) | Symbolic accuracy + long-term sim error | Paper (code pending) |
| **Pixel Physicist (ours)** | 2026 | **Raw phone video** | **SAM 3.1 tracking + calibration + cleaning** | Sparse fit (STLSQ) + shooting refinement + **Spark scientist loop** | **Scientist reading residuals/plots, proposing structure, judging overlays** | **Held-out sim RMSE on last 40% + separate take + synthetic ground truth** | Public repo |

### 23.2 What each does better (learn from them)

- **SINDy / PySR / AI Feynman:** mature, fast, well-tested numerics. Our classical fitter is a re-implementation of their ideas (STLSQ + shooting). We should not claim novelty on the fitting side; the plan already says so honestly.
- **LLM-SR:** strong evidence that LLM-guided search beats black-box SR, especially out-of-domain. Its "skeleton + optimizer" pattern is exactly our `fit_model` tool. Its warning about memorization (Feynman equations in pretraining) is why we use synthetic *transformed* systems and held-out takes, not textbook equations alone.
- **KeplerAgent:** closest in spirit. Borrow its proven ideas:
  - An **experience log** recording every tool call, args and outcome, plus a **workspace file list** fed back into the agent each turn. We already log transcripts; add the workspace-file listing to the user message.
  - A **visual subagent** that reads plots and returns natural-language findings. Our Spark loop already attaches plots; make "read the residual plot" an explicit tool rather than passive context.
  - **Stopping criteria:** a hard max tool-call/round cap (we have 8 rounds) plus a success threshold, with graceful fallback when noise prevents reaching it (we have early stop).
  - **Long-term prediction error** (simulate the discovered ODE forward) as a metric in addition to pointwise RMSE — we already use held-out simulation RMSE; keep it and label it as long-term prediction.
  - **Template proposals** for PySR (e.g. `sin(f(x))`) — maps to our term-string proposals; keep.
- **Visual Physics:** validates that video → physics is feasible end-to-end; their limitation (synthetic videos, no real footage, no LLM, no ablation) is exactly our opening.

### 23.3 Our genuine differentiators (defend these)

1. **End-to-end from real phone footage.** Every comparable system starts from clean tables or synthetic video. We start from a hand-held clip, with SAM 3.1 as the sensor and a calibration step. This is the demo's hook.
2. **Perception is a first-class, swappable backend** (SAM video / SAM frames / color / synthetic) with a documented fallback, so a Day-1 API surprise never kills the project.
3. **The LLM is the scientist, not the fitter.** Spark proposes and revises *structure* from residual evidence (plots + overlays); a classical fitter does the numbers. The ablation (Section 8) isolates this contribution with numbers, not vibes.
4. **Vision as a variable.** Condition C (text-only residual stats) tests whether plots/images matter — directly addressing the "is the LLM decorative?" critique that KeplerAgent/LLM-SR never measure (they feed numbers only).
5. **Honest negative control** (spring-mass: Spark must *not* add terms) and an explicit "where Spark does not help" reporting requirement.
6. **Cost- and reproducibility-aware.** Per-run cap, JSONL cost log, cached prefixes, 5 repeats with mean+spread, mockable client for offline tests. Research systems rarely report this; it matters for a hackathon and for credibility.
7. **Held-out *time* prediction on a separate take.** Most SR work reports in-sample or random-split error; we report forward simulation on unseen seconds and an unseen clip.

### 23.4 Risks this comparison exposes

- **KeplerAgent is very recent (Feb 2026) and conceptually adjacent.** A reviewer may ask "how is this different from KeplerAgent?" Answer: video perception end-to-end, real footage, SAM, cost-guarded reproducibility, and an ablation that isolates vision — none of which KeplerAgent does. Put this in the README and the Q&A prep.
- **Our scope is 1-DOF.** KeplerAgent/LLM-SR handle coupled ODEs and PDEs. We deliberately stay 1-DOF for MVP (Section 3 non-goals); do not drift into multi-DOF before the feature freeze.
- **No symmetry/conservation tools.** KeplerAgent's symmetry discovery is a real advantage on ODE systems. We skip it for MVP (out of scope); if time permits on Day 6, a tiny "check for even/odd symmetry in residual" heuristic is a cheap win, but only after G5.
- **Memorization risk.** Like LLM-SRBench, we must use transformed/synthetic systems and held-out takes so Spark cannot recite a textbook equation. Already planned; keep it.

### 23.5 Concrete plan edits implied by this research

1. Add a workspace-file listing + experience log to the scientist user message (KeplerAgent pattern) — update `scientist/prompts.py` and `scientist/loop.py`.
2. Add a `read_plot`/`visual_subagent` tool so Spark can ask for a natural-language reading of a residual plot, rather than only receiving the image — update `scientist/tools.py`.
3. Label the held-out metric as "long-term prediction error (simulation)" in reports to match the literature's terminology.
4. Keep the term "symbolic accuracy" for synthetic clips (term-correctness rate) and report it alongside NMSE.
5. Add a one-paragraph "related work and positioning" subsection to the README (Day 6), citing Visual Physics, LLM-SR, KeplerAgent, SINDy and PySR, and stating the differentiators in 23.3.

**Sources:** Visual Physics (arXiv 1911.11893); AI Feynman (Science Advances 2020); SINDy (Brunton et al. 2016); PySR (Cranmer 2023); LLM-SR (Shojaee et al., ICLR 2025, arXiv 2404.18400); Knowledge-integrated physics-informed SR (Scientific Reports 2026); KeplerAgent (Yang et al., arXiv 2602.12259, 2026); LLM-SRBench (Shojaee et al. 2025).
