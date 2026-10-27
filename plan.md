# Pixel Physicist: 8-Day Plan (v2)

**One-liner:** Film a moving object with a phone. SAM 3.1 tracks it, and Muse Spark acts as the scientist: it reads the residual evidence, diagnoses why the first model fails, proposes the missing physics, and the system validates the result by predicting motion it never saw.

**Event:** Meta Global AI Developer Hackathon, track *Push the Frontier*. Solo.
**Build window:** Tue Oct 27 → Tue Nov 3, 2026 (8 days; Nov 3 is buffer + submission).
**Hardware:** 16GB Mac, CPU only. **Budget:** $150 Meta Model API credits.

> **Verdict on v1 (archived as `plan-v1.md`):** the design was sound but ~2–3× too large for a solo 8-day build. It had four tracking backends, twenty-odd modules, five scientist tools, a judge, a vision subagent, four real scenarios with three takes each, five ablation repeats, an HTML report, and only seven working days. v2 keeps the story and the honest-ablation idea, and cuts everything that does not appear on screen or in the README.

---

## 0. Reality check: flaws in v1 and the v2 fix

| v1 flaw | Why it hurts in 8 days | v2 fix |
|---|---|---|
| 4 tracker backends (SAM video, SAM frames, color, synthetic) | Each backend needs filming, debugging, parsing, and a fallback path | Build 2 real backends: **SAM video** + **HSV color**. Synthetic is a renderer, not a tracker. Drop SAM-frames unless SAM video fails on Day 1 |
| ~20 modules / 25 files | Too much scaffolding, too many interfaces to test | **11 modules + `__init__.py`** (Section 8) |
| 5 scientist tools + `read_plot` subagent + judge | Doubles API cost and debugging surface; the judge is decorative | **3 tools** and **no judge** (Section 4.5) |
| 4 real scenarios × 3 takes | ~12 clips to film, transfer, track, and label; most never reach the demo | **2 real scenarios × 2 takes** + 1 synthetic (Section 5) |
| "5 repeats per clip per condition" × 3 clips | ~18 LLM runs, hours of wall-clock, noisy results | **1 baseline run + 3 Spark repeats** on 2 real clips + 1 synthetic (Section 6) |
| Held-out RMSE over the last 40% of a clip | Phase drift makes the metric dominated by integration error, even for the right structure | **Fixed 2-second rollout horizon** + separate take; report term correctness too (Section 6) |
| Sparse regression on numerically differentiated `q''` | Differentiation amplifies phone-tracking noise; the coefficient estimate can be garbage before shooting starts | **Weak-form (integral) initialization**, then shooting refinement (Section 4.4) |
| Supporting both Chat Completions and Responses API | Two code paths, two tool schemas, two sets of bugs | **Chat Completions only**; Responses is a post-freeze experiment |
| Automated SAM calibration of the A4 reference | Another SAM failure mode; the scale is not needed to discover structure | **Manual two-point scale is primary**; SAM reference is optional. Structure/prediction work in pixel or normalized units |
| Day 1 overloaded (repo + client + cost log + 6 smoke tests + 12 takes + synthetic renderer) | Guarantees a late start on the risky part (tracking) | Day 1 is **de-risk in the morning, film in the afternoon, one synthetic clip in the evening** |
| "B beats A on at least 2 clips" | High bar; one bad clip kills the claim | **B beats A on the hero clip, and does not regress on the negative control** (Section 6) |
| 5 tools, 8 rounds, free-form terms | Spark can wander, burn budget, and fail the gate | **Menu-first terms**, 6 rounds, submit guard (Section 4.5) |

**What v2 keeps from v1:** the end-to-end video-to-law story, SAM + Spark as the two Meta models, a classical fitter with Spark doing structure discovery, the ablation, the negative control, the safe expression grammar, cost caps, synthetic ground truth, and the honest-limits statement.

---

## 1. The story in one paragraph

A phone clip is the only input. SAM 3.1 turns it into a tracked trajectory. A calibrator converts pixels to a generalized coordinate (angle or along-line position). A classical fitter fits a default physical model. If the model is wrong, the residual plot shows a structure the baseline cannot explain. Muse Spark reads that plot, names the missing physical effect, proposes the term (e.g. quadratic drag `|v|v` or Coulomb friction `sign(v)`), and the fitter refits. The win condition is not a prettier curve on the fit window: it is lower error on motion the model never saw.

---

## 2. Scope: what ships, what waits, what is cut

### MVP-0 — must exist at submission

1. `pixelphys run <clip> --prompt "<object>" [--scale PX_PER_M] [--pivot X,Y]`:
   track → calibrate (manual allowed) → clean → split → baseline fit → Spark loop → held-out prediction → overlay → report.
2. **One real hero clip** where the baseline visibly fails and Spark finds the missing term.
3. **One real negative control** (small-amplitude pendulum with viscous damping; easier to film than a spring) where Spark must not add terms.
4. **One synthetic clip** with known ground truth, used for the reproducible test and the ablation.
5. **Ablation table**: baseline vs baseline+Spark, 3 Spark repeats, held-out NRMSE + term correctness + cost.
6. Public repo, README with the table and honest limits, demo video 60–90 s.

### MVP-1 — only if MVP-0 is green before the Day-6 gate

- Second real hero clip (the other of tossed ball / slope).
- A separate take as an extra generalization check.
- `compare_overlay` tool for Spark.
- 5 repeats instead of 3.
- Automated SAM calibration of the reference object.

### Cut entirely (do not touch before the feature freeze)

2-DOF projectile, bouncing ball, bootstrap confidence intervals, text-only ablation, Muse Wonder, Contributor mode, web app, HTML dashboard, judge, `read_plot` subagent, SAM-frames backend, Responses API, moving camera, multiple objects.

**The cut list is part of the plan, not a failure.**

---

## 3. Architecture (lean)

```
phone clip
  → [tracking.py]  SAM 3.1 video (primary) / HSV color (fallback)
  → [calibrate.py] manual scale + pivot (optional SAM reference)
  → [trajectory.py] clean, resample, q(t), fit/held-out split
  → [physics.py]   baseline fit (weak-form init + shooting)
  → [scientist.py] Spark loop: fit_model, plot_residuals, submit_model
  → [physics.py]   predict held-out horizon
  → [render.py]    ghost overlay mp4 + report.html
  → [cli.py]       orchestration, run directory
```

**Two Meta models only:**
- **SAM 3.1** (`sam-3.1`) — sensor. Responses API, one request per clip, `metadata.mask_encoding="one_bit"`, parse with `meta_sam_parser`, decode masks, centroid per frame.
- **Muse Spark** (`muse-spark-1.3`) — scientist. Chat Completions, `reasoning_effort="medium"` (use `"low"` while tuning prompts, `"high"` only for the final recorded run).

**Why this is enough:** the model only ever sees a 1-DOF coordinate and a small term library. We do not need a general physics engine.

---

## 4. Technical approach

### 4.1 Capture (Day 1, during the event)

Two scenarios, two takes each (fit + held out):

- **Hero A — tossed light ball (up and down).** Quadratic drag is bidirectional, so the unsigned `v²` baseline term cannot accidentally fit it. Use a light ball or crumpled paper; plain background; bright object.
- **Hero B — ball rolling down a shallow towel/cloth slope.** Coulomb friction `sign(v)` is the missing term; the baseline can fit constant acceleration but not the stop/creep behavior.
- **Negative control — vertical spring-mass** (or a small-amplitude pendulum with viscous damping). The baseline should win; Spark must add nothing.

Protocol: phone on books/tripod, perpendicular to the motion plane, locked exposure/focus, 30 or 60 fps, A4 sheet or ruler in the plane for a manual scale, 8–15 s per take. Measure pendulum length or slope angle with a ruler. **Two takes per scenario is enough**; a third only if the first is visibly bad.

### 4.2 Tracking

`Observation(frame_idx, t, cx, cy, bbox, mask_area, confidence)`.

- **SAM video (primary):** upload the clip once via Files API (`purpose=user_data`, MP4 faststart), one Responses request with a short noun-phrase concept prompt, stream the output, parse with `meta_sam_parser.parse_video_stream`, decode masks with `decode_mask_to_raster`. Track the largest mask (or the one with the most stable area). One request per clip so the 50-frame minimum is paid once.
- **HSV color (fallback):** tuned threshold, largest contour, centroid. Guaranteed to work on a saturated object against a plain background. This is the Day-1 safety net.
- **Synthetic renderer:** integrate a known ODE, render a dot on a plain background, write MP4, return the exact ground-truth trajectory. Used for tests and the synthetic ablation.

### 4.3 Calibration and cleaning

- **Manual scale is primary:** two clicks on a known length (A4 long side 0.297 m, or a ruler), or `--scale` from a measured value. Automated SAM segmentation of the reference object is optional and never blocks a run.
- **Pivot:** manual click or `--pivot`. `q(t)` is the pendulum angle `atan2` or the along-line position.
- **Cleaning:** drop frames whose `mask_area` is a >3σ outlier; resample onto a uniform time grid; Savitzky–Golay smooth the position (window 7 or the largest odd ≤ n). Split by time, 60/40. Require ≥30 points.
- **Calibration is allowed to fail gracefully:** structure discovery and NRMSE work in pixel/normalized units; physical units are a bonus on the demo clip.

### 4.4 Fitting (the part that must be robust)

Model: `q'' = Σ cᵢ φᵢ(q, v)`, with a whitelisted term grammar.

1. **Weak-form initialization.** Multiply by a smooth compact-support test function `w(t)` and integrate by parts twice:
   `∫ w'' q dt = Σ cᵢ ∫ w φᵢ(q,v) dt`.
   This avoids differentiating noisy data and gives a linear least-squares estimate of the coefficients. Use 5–8 Gaussian test functions. This is the single biggest robustness improvement over v1: the weak/integral form of SINDy is well documented as far more noise-robust than pointwise derivatives (Schaeffer & McCalla; Messenger & Bortz), which matters because phone tracking gives noisy positions and we need second derivatives.
2. **Shooting refinement.** `scipy.optimize.least_squares` over coefficients + `q0, v0`, integrating with `solve_ivp` (DOP853; fall back to BDF for `sign(v)`). 3 starts. Bound coefficients to ±1e3 and initial conditions to ±10% of the observed range.
3. **Selection.** Score by held-out NRMSE (fixed horizon) plus a BIC-style penalty. The baseline and every Spark candidate use the same fitter and the same score.

**Default baseline library (fixed, intentionally incomplete):** `1, q, v, q², v², sin(q), cos(q)`.
It excludes `|v|v`, `sign(v)`, and `q·v` so the hero terms are genuinely outside the baseline. `1` is kept so a constant slope/gravity term is representable. If a hero clip still does not fail, shrink the library (drop `v²`) rather than adding a hero term.

**Safe expression grammar:** `ast.parse(mode="eval")` with a visitor that accepts only `q`, `v`, numbers, `+ - * / **`, unary minus, and `sin cos tanh sign abs sqrt exp`. No attribute access, calls to other names, comprehensions, lambdas, imports, comparisons, subscripts. Cap term length at 64 and term count at 8. Evaluate via a compiled whitelisted AST with a NumPy namespace; never `eval` a raw model string.

### 4.5 The Spark scientist loop

**Tools (3, not 5):**

| Tool | Purpose |
|---|---|
| `fit_model(terms, rationale)` | Validate terms, fit on the fit window, score on the held-out horizon, store a model id, return metrics |
| `plot_residuals(model_id, versus)` | Return a residual plot image against `q`, `v`, or `t` |
| `submit_model(model_id, explanation)` | Final structured answer; rejected if it does not beat the baseline by ≥2% NRMSE or BIC |

**Menu-first, free-form second.** Give Spark a curated candidate list in the prompt:
`abs(v)*v`, `sign(v)`, `q*v`, `v**3`, `sin(q)*v`, `cos(q)*v`.
It may propose another whitelisted term, but the prompt says "prefer the menu; justify each term physically and name the quantity it represents." This raises the success rate and keeps the demo legible.

**Loop:** at most 6 rounds; stop early if the best held-out NRMSE improves <2% for 2 rounds; per-run cap $0.50; a static system-prompt prefix for cache hits; every call logged. If the loop ends without `submit_model`, submit the best model with a note. If Spark submits something worse than the baseline, the tool rejects it and the loop continues.

**Prompt cheat sheet:** residual ∝ `v` → viscous drag; ∝ `|v|v` → quadratic drag; ∝ `sign(v)` → Coulomb friction; a sinusoid in `t` → driving force; a trend in `q` → wrong restoring term.

**No judge, no vision subagent.** Spark sees the plot directly; a second model reading the plot is redundant, expensive, and not on the critical path.

### 4.6 Outputs

- The equation with coefficients (and units when calibration is trustworthy).
- Held-out NRMSE and a ghost-overlay MP4.
- A single `report.html` with the equation, residual plot, Spark transcript, metrics, and cost.
- `results/ablation.csv` for the README.

---

## 5. Experiment set (3 clips, not 8)

| # | Clip | Hidden physics | Role |
|---|---|---|---|
| H | Tossed light ball **or** rolling ball on slope | `|v|v` **or** `sign(v)`, outside the baseline | Hero; baseline must visibly fail |
| N | Small-amplitude pendulum with viscous damping (easier to film than a spring) | `sin(q) ≈ q` restoring + `v` damping, inside the baseline | Negative control; Spark must add nothing |
| S | Synthetic toss with known drag | Known `|v|v` coefficient | Repro test, ablation, fallback if H is weak |

Film both H candidates if time allows (2 takes each); pick the one with the clearer baseline failure on Day 3. Do not promise which term Spark finds until it is measured.

### 5.1 Where to get clips (use datasets first, film only the demo)

**Do not spend Day 1 filming six takes.** Use an open dataset for development and benchmarking, and film only the one or two clips that appear in the demo.

| Source | License | What it has | Use |
|---|---|---|---|
| **Morpheus Real-World Physics Videos** (`physics-from-video/morpheus-real-world`, Hugging Face) | **MIT** | 124 real videos: `falling_ball`, `projectile`, `bouncing_ball`, `rolling_*`, `holonomic_pendulum`, `spring`, collisions; 1280×1024 MP4 + per-frame JPGs | **Primary dev/benchmark corpus.** Exactly the phenomena we need |
| **Physics-IQ / Physics-IQ Verified** (DeepMind) | Data **CC BY 4.0**, code Apache 2.0 | 396 real 4K/30fps videos, five physics categories, three angles, two takes | Extra held-out clips; attribution required |
| **CLEVRER** (MIT CSAIL) | **CC0** | 20k synthetic 5-second collision videos + annotations | Pipeline tests with no license friction |
| **Physion / Physion++** | MIT | Synthetic rolling/sliding/falling/colliding scenes | Stress tests; multi-object, not 1-DOF |
| **Kubric / PyBullet / Blender** | Apache 2.0 / own | Generate your own synthetic clips with exact ground truth | Synthetic hero + ground truth; safest for compliance |

**Recommended setup:** download Morpheus, use `projectile` or `falling_ball` as the drag hero, `holonomic_pendulum` as the negative control, and `rolling_*` for the Coulomb case. Use these for Days 2–5. Then film **two 10-second clips** yourself (one tossed ball, one pendulum) for the demo and to satisfy the "filmed during the event" framing. Total filming time: ~30 minutes.

**Compliance caveat:** check the hackathon rules before putting third-party footage in the submission video. MIT/CC BY allow reuse with attribution, but if the rules require event-filmed footage, use datasets only for development and film the demo clips yourself. Never use DAVIS (CC BY-NC, non-commercial) or YouTube-sourced clips.

---

## 6. Metrics and ablation (small but honest)

**Primary metric — held-out NRMSE over a fixed horizon.**
Fit on the first 60% of a clip. Predict the next **2 seconds** (not the whole 40%) from the true state at the split. Normalize by the amplitude of the held-out segment. A 2-second horizon is long enough to expose a wrong structure and short enough that phase drift does not dominate. If a second take exists, repeat the rollout over its first 2 seconds.

**Secondary metrics.**
- Term correctness on the synthetic clip (exact term set).
- Relative parameter error on the synthetic clip.
- BIC.
- Rounds and cost per run.

**Ablation.**

| Clip | A (baseline) | B (baseline + Spark) |
|---|---|---|
| H (hero) | 1 run | 3 repeats |
| N (negative control) | 1 run | 3 repeats |
| S (synthetic) | 1 run | 3 repeats |

12 runs total, ≈2–3 hours and ≈$6–12. Report mean and range for B, single values for A.

**Success criterion (scoped, defensible):**
- On H: B's mean NRMSE is ≥10% lower than A, and at least 2 of 3 repeats beat A.
- On N: B is within 5% of A and adds no terms in at least 2 of 3 repeats.
- On S: B recovers the true term in at least 2 of 3 repeats.

If H fails but S succeeds, report exactly that: "the synthetic benchmark shows the mechanism; the real clip was not clean enough." That is an honest result, not a failed demo.

---

## 7. The 8-day schedule

Each day has a **gate** and a **cut line**. If a gate slips by more than 3 hours, take the cut line immediately. Do not borrow from the next day.

### Day 1 — Tue Oct 27: De-risk, then film

**Morning (3h): foundation + blockers.**
- Public repo, `uv`/venv, `.env.example`, `.gitignore`, license. Commit early.
- `meta.py`: OpenAI client (base URL + explicit `MODEL_API_KEY`), one Chat Completions call, one tool call, one image call; cost log.
- Film a 5-second test clip first, then `tracking.py`: SAM video smoke test on it; record latency, billed frames, and whether `meta_sam_parser` decodes masks. If the video path fails, test the image path once.

**Afternoon (2h): clips.**
- Download Morpheus (`physics-from-video/morpheus-real-world`, MIT); confirm `projectile`, `holonomic_pendulum`, `rolling_*` play.
- Film only **two demo clips** (tossed ball, pendulum), ~10 s each, A4/ruler in frame, note the true length/angle. This is the "filmed during the event" footage; datasets carry development and benchmarking.

**Evening (2h): one synthetic clip + trajectory plot.**
- `synthetic.py`: render one known-drag toss with ground truth.
- Plot the SAM trajectory on the real clip; save it.

**Gate G1:** one real clip produces a usable trajectory from SAM **or** color, and the SAM video path is characterized (working, or definitively not). **Cut line:** if SAM video is unusable, use color tracking and keep SAM for the single-frame object/calibration mask; do not build SAM-frames.

### Day 2 — Wed Oct 28: Perception

- `tracking.py`: finish `SamVideoTracker` + `ColorTracker`; `Observation` validation.
- `calibrate.py`: manual scale + pivot; optional SAM reference.
- `trajectory.py`: clean, resample, `q(t)`, split; plot fit/held-out.
- Tests on the synthetic clip.

**Gate G2:** the hero clip yields a smooth `q(t)` with a plausible period/scale; the synthetic tracker recovers its truth within a few pixels. **Cut line:** manual scale only; drop calibration uncertainty; drop the optional SAM reference.

### Day 3 — Thu Oct 29: Physics core

- `expr.py`: AST validation + evaluation.
- `physics.py`: simulator, weak-form init, shooting fit, NRMSE/BIC.
- Run the baseline on H, N, S.

**Gate G3:** baseline recovers the synthetic truth, and **visibly fails on H**. **Cut line:** if neither H candidate fails, switch the hero to the synthetic clip for the discovery story and keep the real clip for the tracking story; shrink the baseline library (drop `v²`) before re-filming.

### Day 4 — Fri Oct 30: Spark loop v1

- `scientist.py`: schemas, 3 tools, prompt, loop, cost cap, transcript.
- Chat Completions with manual history; 6 rounds; submit guard.

**Gate G4:** on the synthetic hidden-term clip, Spark adds the correct term within 6 rounds in at least 3 of 5 runs. **Cut line:** switch to menu-only (no free-form terms), reduce to 4 rounds, and keep the diagnosis text in the transcript even if the term choice is constrained.

### Day 5 — Sat Oct 31: Hero runs + ablation

- Tune prompts on the real hero clip (morning).
- Run the 12-run ablation (afternoon); write `results/ablation.csv`.
- Write the README table and the honest-limits paragraph.

**Gate G5 / FEATURE FREEZE:** H meets the success criterion above, and N is clean. **Cut line:** if the real hero fails, run the ablation on S only and present H as the tracking demo; be explicit in the README. No new features after today.

### Day 6 — Sun Nov 1: Polish + generalization

- `render.py`: ghost overlay MP4 + simple `report.html`.
- CLI polish, error messages, `--scale`/`--pivot` overrides.
- README, license, tests (`expr`, `fit`, `trajectory`), clean-clone test.
- Run one unseen clip end-to-end with no code changes.

**Gate G6:** a fresh clone installs and runs on a bundled clip, producing a report and an overlay.

### Day 7 — Mon Nov 2: Demo

- Write the 60–90 s script; record the screen from a **saved good run** (not a live API call); edit; upload.
- Final README pass, links, compliance checklist.

**Gate G7:** demo is uploaded and under 90 s.

### Day 8 — Tue Nov 3: Buffer + submit

- Submit early. Confirm the exact cutoff in IST. Only bug fixes or a re-record; nothing new.

---

## 8. Repo structure (11 modules + `__init__.py`)

```
pixelphys/
  README.md  LICENSE  pyproject.toml  .env.example  .gitignore
  src/pixelphys/
    __init__.py
    cli.py          # argparse, run orchestration, run dir
    config.py       # settings, paths, model ids
    meta.py         # OpenAI client + cost log + retries
    tracking.py     # Observation, Tracker, SamVideo, Color
    synthetic.py    # render known ODE + ground truth
    calibrate.py    # manual scale/pivot, optional SAM reference
    trajectory.py   # clean, resample, q(t), split
    expr.py         # safe AST grammar + evaluation
    physics.py      # simulate, weak init, shooting, metrics
    scientist.py    # schemas, tools, prompts, loop
    render.py       # overlay mp4 + report.html
  experiments/ablation.py
  data/clips/  data/synthetic/
  results/  runs/  tests/
```

**Stack:** Python 3.11+, `openai`, `meta-sam-parser`, `numpy`, `scipy`, `opencv-python`, `imageio-ffmpeg`, `matplotlib`, `rich`, `pytest`. No sympy (the AST whitelist plus NumPy is enough), no pandas, no Jinja, no web framework.

**Why 11 files:** each file is one responsibility and small enough to debug in an evening. If a file grows past ~300 lines, split it then, not now.

### 8.1 Module × day work breakdown

Module IDs are used in the schedule; each module has one owner (you) and one acceptance check.

| ID | Module | Purpose | Depends on | Built | Hardened | Done when |
|---|---|---|---|---|---|---|
| M0 | `data/` | Morpheus download + two filmed demo clips | — | D1 | D1 | clips play locally |
| M1 | `config.py` | settings, paths, model IDs | — | D1 | D1 | loads env, fails fast on missing key |
| M2 | `meta.py` | OpenAI client, cost log, retries | M1 | D1 | D1–D4 | one text, one tool, one image call all logged |
| M3 | `tracking.py` | `Observation`, SAM video, HSV color | M1, M2 | D1 smoke | D2 | hero clip → a plot of the trajectory |
| M4 | `synthetic.py` | render a known ODE + ground truth | M1, M8 (simulator) | D1 | D2–D3 | MP4 + truth JSON produced |
| M5 | `calibrate.py` | manual scale/pivot, optional SAM reference | M3 | D2 | D2 | scale/pivot applied to `q(t)` |
| M6 | `trajectory.py` | clean, resample, `q(t)`, split | M3, M5 | D2 | D2–D3 | smooth `q(t)`, 60/40 split |
| M7 | `expr.py` | safe AST grammar + evaluation | — | D3 | D3 | valid/invalid/injection tests pass |
| M8 | `physics.py` | simulate, weak-form init, shooting, metrics | M6, M7 | D3 | D3–D4 | recovers synthetic truth; baseline fails on H |
| M9 | `scientist.py` | schemas, 3 tools, prompts, loop | M2, M8 | D4 | D4–D5 | finds the hidden term in ≥3/5 synthetic runs |
| M10 | `render.py` | ghost overlay MP4 + `report.html` | M6, M8 | D6 | D6 | overlay + report generated |
| M11 | `cli.py` | argparse, run orchestration | all | D1 skeleton | D6 | one command runs H end-to-end |
| M12 | `experiments/ablation.py` | A/B runner, CSV | M9, M11 | D5 | D5 | 12 rows in `results/ablation.csv` |
| M13 | `tests/` | unit + one integration test | M3–M9 | D2 | D6 | `pytest -q` green without an API key |

**Daily module deliverables:**

| Day | Modules touched | Deliverable | Gate |
|---|---|---|---|
| D1 | M0, M1, M2, M3 (smoke), M4 (first), M11 (skeleton) | repo + working API smoke test + one tracked clip + one synthetic clip | G1 |
| D2 | M3, M5, M6, M13 | smooth `q(t)` for H/N and a synthetic run | G2 |
| D3 | M7, M8, M13 | baseline fit on H/N/S; H visibly fails | G3 |
| D4 | M9 | Spark loop adds the synthetic hidden term | G4 |
| D5 | M9 (tuning), M12 | 12-run ablation CSV + README table | G5 / freeze |
| D6 | M10, M11, M13 | overlay + report + clean-clone run | G6 |
| D7 | none (demo only) | 60–90 s video uploaded | G7 |
| D8 | none (buffer) | submission confirmed | — |

**Critical path:** `M2 → M3 → M6 → M8 → M9 → M10 → demo`. If a day slips, cut work from the non-critical modules (M12 repeats, M10 polish, M13 tests), never from M3/M6/M8/M9.

**Solo + AI assistance:** you are the only human, but `M10` (render/report), `M13` (tests), and the README can be delegated to background agents while you build `M8`/`M9`. Review and integrate their output yourself; do not let them touch the fitting math.

---

## 9. Budget and time budget

| Item | Estimate |
|---|---|
| One Spark run (6 rounds) | $0.15–0.50 |
| Prompt tuning (~40 runs) | $10–20 |
| Ablation (12 runs) | $6–12 |
| SAM (one request per clip, ~10 clips × 300–900 frames) | ~$1–2 |
| Demo takes, buffer | $5–10 |
| **Total** | **≈$25–45** of $150 |

**Guards:** per-run cap $0.50, global alert $40, JSONL log of every call. Use `reasoning_effort="low"` for tuning and `"medium"` for the recorded run. Do not use Contributor mode (the footage is your home; not worth the training-data trade).

**Time budget is the real constraint.** Assume 8 focused hours per day, 6 on Day 7, 2 on Day 8. If a task is not on the critical path to the demo or the README table, it is cut.

---

## 10. Risks and fallbacks

| Risk | Likelihood | Impact | Fallback |
|---|---|---|---|
| SAM video unusable / loses the object | Medium | High | Color tracker (Day-1 safety net); SAM image for the calibration object only |
| Real tracking too noisy for the weak form | Medium | Medium | More smoothing, shorter fit window, synthetic clip as the benchmark backbone |
| Baseline does not visibly fail on a real clip | Medium | High | Switch hero to the synthetic clip; shrink the baseline library; re-film a lighter/faster object |
| Spark does not find the term | Medium | High | Menu-only prompt, 4 rounds, keep the diagnosis text; report the synthetic success honestly |
| Spark submits a worse model | Low | Medium | `submit_model` guard (must beat baseline by 2% or BIC) |
| Held-out 2 s horizon still noisy | Medium | Medium | Report both 1 s and 2 s horizons; term correctness is the primary claim on synthetic |
| Third-party dataset not allowed in the submission video | Low | High | Use datasets for development only; film the two demo clips yourself; attribute any dataset used in the README |
| API outage near the deadline | Low | High | Disk cache for dev/demo; record the demo from a saved run; submit early |
| Scope creep | High | High | Hard feature freeze at the end of Day 5; the cut list in Section 2 is binding |
| Day 1 overruns | Medium | High | Film in the afternoon regardless; if the client is not ready, use the color tracker and a synthetic clip for the evening gate |

---

## 11. Demo storyline (60–90 s)

| Time | Shot |
|---|---|
| 0:00–0:08 | Hook: the real hero clip; "Can a model discover the physics of this from video alone?" |
| 0:08–0:20 | SAM tracks the object; the calibrated/plot trajectory appears |
| 0:20–0:32 | Baseline fit; the ghost drifts from reality; NRMSE shown |
| 0:32–0:55 | Spark reads the residual plot; one-line diagnosis ("error grows with speed squared → quadratic drag"); refit; error collapses |
| 0:55–1:10 | Final equation; held-out ghost now tracks the real footage |
| 1:10–1:25 | The synthetic clip and the ablation table; per-run cost; closing card |

Record from a saved run, keep narration to a script written on Day 6, show only event-time work, use no music (or a licensed track).

---

## 12. Definition of done, compliance, honest limits

**Done:**
1. Fresh clone → install → `pixelphys run` on a bundled clip → `report.html` + `overlay.mp4`.
2. `results/ablation.csv` has the 12 runs; README reports the mean/range and says where Spark does **not** help.
3. Demo video is uploaded and under 90 s.
4. README documents the limits: planar 1-DOF, static perpendicular camera, one rigid object, a classical fitter doing the numbers.

**Compliance checklist:**
- [ ] Code written Oct 27–Nov 3 only; third-party libraries fine.
- [ ] Footage self-filmed during the event; no third-party video/music.
- [ ] Public repo, license, no keys in history.
- [ ] README names Muse Spark and SAM 3.1 and what each does.
- [ ] Demo shows only event-time work.
- [ ] Not a banned category (medical/RAG/chat-with-data/simple image analysis).
- [ ] Cutoff confirmed in IST on Day 1.

**Honest limits (README + Q&A):**
- The numeric fitting is classical. Spark's contribution is proposing and revising model *structure* from residual evidence; the ablation measures that contribution.
- The pipeline handles one 1-DOF coordinate, one object, one static camera, and short clips. It is not a general physics engine.
- Results come from a small number of controlled clips. Where Spark fails, the README says so.

---

## 13. Related work and positioning (short)

Closest systems: **SINDy** (sparse regression on derivatives; PySINDy), **PySR** (genetic-programming symbolic regression), **AI Feynman** (physics-inspired SR), **Visual Physics** (video → latent params → SR, synthetic only), **LLM-SR** (LLM proposes equation skeletons; tabular data), and **KeplerAgent** (LLM agent orchestrating symmetry tools + PySINDy/PySR; tabular ODE data).

**Our differentiator:** end-to-end from *real phone video* (SAM 3.1 perception + calibration + cleaning), with Spark as a residual-reading scientist, and an ablation that isolates the LLM's structure contribution, including a negative control. None of the above combines real-video perception with LLM-guided structure discovery and a held-out validation. Keep this paragraph in the README and the Q&A.

---

## 14. Day-1 facts to confirm (append findings here)

- [ ] Exact SAM 3.1 request/response works with `meta_sam_parser`; billed frames via `usage.video_frames_processed`.
- [ ] Spark Chat Completions tool calling works with `tool_choice="auto"`; `reasoning_effort` values.
- [ ] Spark token cost per image at our downscaled size (≤1024 px long side).
- [ ] Rate limits and any max request size.
- [ ] Submission cutoff time in IST.

**If any of these fail, the fallback is in Section 10, not a redesign.**
