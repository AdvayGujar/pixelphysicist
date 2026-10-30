"""M9: Spark scientist loop - schemas, 3 tools, prompts, loop. (Day 4.)

Prepared ahead of the live runs: the tool layer and loop mechanics are
complete and tested offline with a scripted chat function. The loop only
ever fits through ``physics.score_candidate`` (never ``fit_terms``
directly), so a bad candidate -- exploding rollout, exhausted budget,
invalid terms -- comes back as data with an error string, not a crash.

Protocol decisions carried in from Day 3 (see plan.md audit responses):
- The shipped baseline is ``[1, q, v, v**2]``; quote it, not the plan's
  original 7-term library.
- Tune against the synthetic clip and the hero split protocol. Do not use
  cross-take (``rollout_score``) rankings as evidence about drag.
- Gravity-only (``["1"]``) is the published control on real clips: on the
  real hero it beats the drag model, so a winning submission must beat the
  *baseline*, and the transcript must not claim more than that.
"""
from __future__ import annotations

import base64
import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import physics as P

MENU_TERMS = ["abs(v)*v", "sign(v)", "q*v", "v**3", "sin(q)*v", "cos(q)*v"]
MAX_ROUNDS = 6
COST_CAP_USD = 0.50
SUBMIT_IMPROVEMENT = 0.02  # beat baseline held-NRMSE by >= 2%, or win on BIC
# Audit A3: the in-loop fit budget. Committed baseline runners keep 60 s
# (P.DEFAULT_FIT_BUDGET_S); loop candidates that need more are not viable.
LOOP_FIT_BUDGET_S = 15.0
# Audit A4: screen-stage budget. One weak-init start, loose tolerances.
SCREEN_BUDGET_S = 5.0
# Audit A5: fit_model calls per loop, counted across rounds.
MAX_FIT_CALLS = 8

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "fit_model",
            "description": "Fit a candidate term set on the fit window and "
                           "score it on the held-out horizon.",
            "parameters": {
                "type": "object",
                "properties": {
                    "terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Term strings in q and v, e.g. "
                                       "[\"1\", \"abs(v)*v\"]. Max 8 terms.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Physical justification for this "
                                       "structure, naming what each added "
                                       "term represents.",
                    },
                },
                "required": ["terms", "rationale"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_residuals",
            "description": "Render the residuals (observed - predicted) of "
                           "a fitted model on the fit window, against t, q, "
                           "or v. Returns a plot image.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "versus": {"type": "string", "enum": ["t", "q", "v"]},
                },
                "required": ["model_id", "versus"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_model",
            "description": "Submit a fitted model as the final answer. "
                           "Rejected unless it beats the baseline by >= 2% "
                           "held-out NRMSE or on BIC (submitting the "
                           "baseline itself is always accepted).",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_id": {"type": "string"},
                    "explanation": {
                        "type": "string",
                        "description": "What the final structure says "
                                       "physically, and why the evidence "
                                       "supports it.",
                    },
                },
                "required": ["model_id", "explanation"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a physicist studying one-dimensional motion q(t) extracted from video.

You work through tools only: fit_model proposes a structure and fits it
classically (you never guess coefficients), plot_residuals shows you where
a model fails, and submit_model ends the run with your final structure.

The starting point is the baseline model with terms [1, q, v, v**2]
(terms are expressions in q and v; q'' = sum of coefficient * term). Its
unsigned v**2 term cannot express drag that acts in both directions of
motion. Candidate missing terms, prefer these before free-form proposals:
abs(v)*v (quadratic drag), sign(v) (Coulomb friction), q*v, v**3,
sin(q)*v, cos(q)*v. You may propose any other term in the same grammar
(q, v, numbers, + - * / **, sin cos tanh sign abs sqrt exp), at most 8
terms total. Replace, do not just add: the baseline's unsigned v**2 is
often the wrong term, so try each candidate INSTEAD of v**2 (e.g.
[1, abs(v)*v]) as well as added to the baseline. If a fit returns an
integration error, retry a sparser version carrying the same new term
(drop q/v/v**2 first) before moving on.
Parsimony first (prompt v2): when you suspect quadratic drag, your
FIRST drag fit must be the sparse replacement [1, abs(v)*v] -- constant
plus drag, nothing else. Adding drag to the full baseline lets q and v
absorb its effect and hides the comparison. Only add q/v back if the
sparse model's residuals demand it, and say what in them demands it.

Reading residuals (observed - predicted) on the fit window:
- residual proportional to v suggests viscous drag (a v term),
- growing like |v|*v / speed-squared suggests quadratic drag: abs(v)*v,
- a roughly constant offset that flips sign with velocity suggests
  Coulomb friction: sign(v),
- a sinusoid against t suggests a periodic driving force,
- a trend against q suggests the restoring (position) term is wrong.

Rules:
- Justify each added term physically and name the quantity its
  coefficient represents (e.g. drag per unit mass, in 1/length).
- Prefer fewer terms: selection uses held-out NRMSE and BIC, so an extra
  term must earn its place on motion the fit never saw.
- submit_model is rejected unless the model beats the baseline by at
  least 2% held-out NRMSE or on BIC. When a fitted model beats the
  baseline, call submit_model promptly with the physical explanation;
  do not delay submission for extra plots. If nothing beats the
  baseline, submit the baseline itself and explain why the baseline
  suffices.
- Finish by calling submit_model. Do not end with a plain-text answer."""


# Audit B6: v2 submit rule (AND, plus train check). Derived from the v1
# prompt by exact replacement so the two can never drift apart silently.
_V1_RULE = """- submit_model is rejected unless the model beats the baseline by at
  least 2% held-out NRMSE or on BIC. When a fitted model beats the
  baseline, call submit_model promptly with the physical explanation;
  do not delay submission for extra plots. If nothing beats the
  baseline, submit the baseline itself and explain why the baseline
  suffices."""
_V2_RULE = """- submit_model is rejected unless the model beats the baseline by at
  least 2% held-out NRMSE AND on BIC, with train RMSE no worse than 5%
  above the baseline's. A held-out win that loses on BIC or on train is
  held-window luck, not physics: do not submit it; keep searching or
  submit the baseline. When a fitted model beats the baseline on all
  three, call submit_model promptly with the physical explanation;
  do not delay submission for extra plots. If nothing beats the
  baseline on all three, submit the baseline itself and explain why the
  baseline suffices."""
assert _V1_RULE in SYSTEM_PROMPT, "v1 submit rule drifted; fix _V1_RULE"
V2_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(_V1_RULE, _V2_RULE)

# Guard v3 (pre-registered in docs/maths.md §5 and the Day-5 v2 write-up):
# held-NRMSE AND BIC, dropping the train clause that misfired in v2 (the
# overfit baseline owns the fit window, so sparse truth loses on train by
# construction). Same exact-replacement versioning as v2.
_V3_RULE = """- submit_model is rejected unless the model beats the baseline by at
  least 2% held-out NRMSE AND on BIC. A held-out win that loses on BIC is
  held-window luck, not physics: do not submit it; keep searching or
  submit the baseline. Train RMSE is not part of this decision because the
  overfit baseline owns the fit window. When a fitted model beats the
  baseline on both, call submit_model promptly with the physical
  explanation; do not delay submission for extra plots. If nothing beats
  the baseline on both, submit the baseline itself and explain why the
  baseline suffices."""
assert _V1_RULE in SYSTEM_PROMPT, "v1 submit rule drifted; fix _V1_RULE"
V3_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(_V1_RULE, _V3_RULE)

_V1_SUBMIT_DESC = ("Submit a fitted model as the final answer. "
                   "Rejected unless it beats the baseline by >= 2% "
                   "held-out NRMSE or on BIC (submitting the "
                   "baseline itself is always accepted).")


def v2_schemas() -> list[dict[str, Any]]:
    """Tool schemas with the v2 submit rule (audit B6)."""
    schemas = copy.deepcopy(TOOL_SCHEMAS)
    assert schemas[2]["function"]["name"] == "submit_model"
    assert schemas[2]["function"]["description"] == _V1_SUBMIT_DESC, \
        "v1 submit description drifted; fix _V1_SUBMIT_DESC"
    schemas[2]["function"]["description"] = (
        "Submit a fitted model as the final answer. "
        "Rejected unless it beats the baseline by >= 2% held-out NRMSE "
        "AND on BIC, with train RMSE within 5% of the baseline's "
        "(submitting the baseline itself is always accepted).")
    return schemas


def v3_schemas() -> list[dict[str, Any]]:
    """Tool schemas with the v3 submit rule (held AND BIC, no train)."""
    schemas = copy.deepcopy(TOOL_SCHEMAS)
    assert schemas[2]["function"]["name"] == "submit_model"
    assert schemas[2]["function"]["description"] == _V1_SUBMIT_DESC, \
        "v1 submit description drifted; fix _V1_SUBMIT_DESC"
    schemas[2]["function"]["description"] = (
        "Submit a fitted model as the final answer. "
        "Rejected unless it beats the baseline by >= 2% held-out NRMSE "
        "AND on BIC (train RMSE is not part of the decision; "
        "submitting the baseline itself is always accepted).")
    return schemas


SPRING_MENU_TERMS = ["q**3", "sign(v)", "abs(v)*v", "q*v", "v**3"]

SPRING_SYSTEM_PROMPT = """You are a physicist studying a vertical spring-mass oscillator. The coordinate q(t) is the displacement from equilibrium (up positive), already extracted from video.

You work through tools only: fit_model proposes a structure and fits it
classically (you never guess coefficients), plot_residuals shows you where
a model fails, and submit_model ends the run with your final structure.

The baseline is [1, q, v, v**2]: constant (gravity/equilibrium offset),
linear stiffness (q), viscous damping (v), and an unsigned quadratic
term. A linear spring with viscous damping is fully inside this
baseline -- if that is all the physics there is, submit the baseline.

Candidate missing physics, prefer these before free-form proposals:
q**3 (nonlinear Duffing stiffness: the spring stiffens or softens at
large stretch), sign(v) (dry friction in the guide/bearing),
abs(v)*v (quadratic air drag), q*v, v**3. Terms are expressions in
q and v (numbers, + - * / **, sin cos tanh sign abs sqrt exp), at most
8 total. Replace, do not just add: try each candidate INSTEAD of the
baseline's v**2 as well as added to the baseline.

Reading residuals (observed - predicted) on the fit window:
- a trend growing with |q| / a cubic-shaped residual vs q suggests
  nonlinear stiffness: q**3,
- an offset that flips sign with velocity suggests dry friction: sign(v),
- growth with speed suggests drag: v**3 or abs(v)*v,
- if residuals look like structureless noise, the baseline suffices.

Rules:
- Justify each added term physically and name what its coefficient
  represents (e.g. cubic stiffness per unit mass).
- Prefer fewer terms: held-out NRMSE and BIC decide, so an extra term
  must earn its place on motion the fit never saw.
- submit_model is rejected unless the model beats the baseline by at
  least 2% held-out NRMSE or on BIC. When a fitted model beats the
  baseline, submit promptly. If nothing beats it, submit the baseline
  itself and explain why linear spring physics suffices.
- Finish by calling submit_model. Do not end with a plain-text answer."""

ROLLING_MENU_TERMS = ["sign(v)", "abs(v)*v", "v**3", "q*v"]

ROLLING_SYSTEM_PROMPT = """You are a physicist studying a ball rolling on a slope. The coordinate q(t) is the distance along the slope (up-slope positive), already extracted from video.

You work through tools only: fit_model proposes a structure and fits it
classically (you never guess coefficients), plot_residuals shows you where
a model fails, and submit_model ends the run with your final structure.

The baseline is [1, q, v, v**2]. The constant term is the downslope
gravity component; it acts the same whichever way the ball rolls.
Candidate missing physics, prefer these before free-form proposals:
sign(v) (Coulomb rolling friction: a roughly constant deceleration that
FLIPS direction with the velocity -- unlike the constant term, it
opposes motion in both directions), abs(v)*v (quadratic drag), v**3,
q*v. Terms are expressions in q and v (numbers, + - * / **,
sin cos tanh sign abs sqrt exp), at most 8 total. Replace, do not
just add: try each candidate INSTEAD of the baseline's v**2 as well.
Position along the slope is NOT a restoring coordinate -- there is no
spring pulling the ball back -- so try the sparse forms first:
[1, sign(v)] and [1, sign(v), v] before any set containing q.

Reading residuals (observed - predicted) on the fit window:
- a roughly constant offset that flips sign when the velocity flips
  (e.g. at the top of the roll) suggests Coulomb friction: sign(v),
- growth with speed suggests drag: v**3 or abs(v)*v,
- a trend with q suggests the slope/gravity model is wrong,
- if residuals look like structureless noise, the baseline suffices.

Rules:
- Justify each added term physically and name what its coefficient
  represents (e.g. rolling-friction deceleration, in length/time^2).
- Prefer fewer terms: held-out NRMSE and BIC decide, so an extra term
  must earn its place on motion the fit never saw. Note the ball
  reversing direction is what separates sign(v) from the constant
  term; a one-direction roll cannot distinguish them.
- submit_model is rejected unless the model beats the baseline by at
  least 2% held-out NRMSE or on BIC. When a fitted model beats the
  baseline, submit promptly. If nothing beats it, submit the baseline
  itself and explain why.
- Finish by calling submit_model. Do not end with a plain-text answer."""


def _num(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


@dataclass
class _Entry:
    scored: Any  # P.ScoredFit (1-DOF) or P.ScoredFit2D (projectile mode)
    rationale: str = ""
    tight: bool = False  # audit A4: screen entries become tight on submit/best


class ScientistTools:
    """The 3 scientist tools over one fit/held-out split."""

    def __init__(self, t_fit: np.ndarray, q_fit: np.ndarray,
                 t_held: np.ndarray, q_held: np.ndarray,
                 out_dir: Path, budget_s: float = LOOP_FIT_BUDGET_S,
                 baseline_terms: list[str] | None = None,
                 baseline: P.ScoredFit | None = None,
                 fit_cache: dict[str, Any] | None = None,
                 guard: str = "v1") -> None:
        self.t_fit = np.asarray(t_fit, dtype=float)
        self.q_fit = np.asarray(q_fit, dtype=float)
        self.t_held = np.asarray(t_held, dtype=float)
        self.q_held = np.asarray(q_held, dtype=float)
        self.out_dir = Path(out_dir)
        self.budget_s = budget_s
        self.models: dict[str, _Entry] = {}
        self._next_id = 0
        self.submitted_id: str | None = None
        self.submit_explanation = ""
        # Audit A1: memoize fits keyed by exact term tuple. Data is fixed
        # per tools instance; the runner may share one dict across repeats
        # of a condition. Values are {"screen": ..., "tight": ...}.
        self._fit_cache: dict[str, Any] = fit_cache if fit_cache is not None else {}
        # Audit B6: submit-guard version. "v1" is the Day-4/5 OR rule;
        # "v2" requires held-NRMSE AND BIC AND train-within-5%.
        # "v3" requires held-NRMSE AND BIC, dropping train (maths.md §5).
        if guard not in ("v1", "v2", "v3"):
            raise ValueError(f"guard must be 'v1', 'v2' or 'v3', got {guard!r}")
        self.guard = guard
        if baseline is None:
            # Audit A2: the runner precomputes condition A once and passes
            # it in, instead of refitting per repeat.
            baseline = P.score_candidate(list(baseline_terms or P.BASELINE_TERMS),
                                         self.t_fit, self.q_fit,
                                         self.t_held, self.q_held,
                                         budget_s=budget_s)
            self.baseline_id = self._register(baseline, "baseline library", tight=True)
        else:
            self.baseline_id = self._register(baseline, "precomputed baseline",
                                              tight=True)

    def _register(self, scored: P.ScoredFit, rationale: str,
                  tight: bool = False) -> str:
        model_id = f"model-{self._next_id}"
        self._next_id += 1
        self.models[model_id] = _Entry(scored, rationale, tight)
        return model_id

    @property
    def baseline(self) -> P.ScoredFit:
        return self.models[self.baseline_id].scored

    def summary(self) -> dict[str, Any]:
        base = self.baseline
        return {
            "n_fit": len(self.t_fit),
            "n_held": len(self.q_held),
            "baseline_id": self.baseline_id,
            "baseline_terms": base.terms,
            "baseline_ok": base.ok,
            "baseline_train_rmse": _num(base.train_rmse),
            "baseline_held_nrmse": _num(base.held_nrmse),
            "baseline_bic": _num(base.bic),
            "baseline_error": base.error,
            "menu_terms": MENU_TERMS,
        }

    @staticmethod
    def _cache_key(terms: Any) -> tuple[str, ...]:
        if isinstance(terms, list) and all(isinstance(t, str) for t in terms):
            return tuple(t.strip() for t in terms)
        return ("__invalid__", repr(terms))

    def _screen(self, terms: Any) -> P.ScoredFit:
        """Cheap advisory fit: one weak-init start, loose tolerances."""
        return P.score_candidate(terms, self.t_fit, self.q_fit,
                                 self.t_held, self.q_held,
                                 budget_s=SCREEN_BUDGET_S, fast=True)

    def tighten(self, model_id: str) -> P.ScoredFit | None:
        """Full-settings refit of a registered model; None when it fails.

        Audit A4: only submitted / best models pay for tight scores.
        Success replaces the entry so reported numbers stay comparable
        to the Day-3/4/5 tight fits.
        """
        entry = self.models.get(model_id)
        if entry is None or not entry.scored.ok or entry.tight:
            return entry.scored if entry is not None and entry.scored.ok else None
        tight = P.score_candidate(entry.scored.terms, self.t_fit, self.q_fit,
                                  self.t_held, self.q_held,
                                  budget_s=self.budget_s)
        key = self._cache_key(entry.scored.terms)
        slot = self._fit_cache.get(key)
        if isinstance(slot, dict):
            slot["tight"] = tight
        else:
            self._fit_cache[key] = {"screen": entry.scored, "tight": tight}
        if tight.ok:
            entry.scored = tight
            entry.tight = True
            return tight
        return None

    def best_bic(self) -> tuple[float | None, str | None]:
        """Lowest BIC over ok models (audit B7: Spark's BIC frontier)."""
        best_id: str | None = None
        best_score = float("inf")
        for model_id, entry in self.models.items():
            scored = entry.scored
            if scored.ok and math.isfinite(scored.bic) and scored.bic < best_score:
                best_id, best_score = model_id, scored.bic
        return (_num(best_score) if best_id is not None else None, best_id)

    def fit_model(self, terms: Any, rationale: str = "") -> dict[str, Any]:
        key = self._cache_key(terms)
        slot = self._fit_cache.get(key)
        tight_cached = (slot.get("tight") if isinstance(slot, dict) else None)
        if tight_cached is not None and tight_cached.ok:
            scored, screening = tight_cached, False
        else:
            screen = (slot.get("screen") if isinstance(slot, dict) else None)
            if screen is None:
                screen = self._screen(terms)
                self._fit_cache[key] = {"screen": screen, "tight": None}
            scored, screening = screen, True
        model_id = self._register(scored, rationale, tight=not screening)
        base_nrmse = self.baseline.held_nrmse
        improvement = None
        if scored.ok and math.isfinite(base_nrmse) and base_nrmse > 0:
            improvement = (base_nrmse - scored.held_nrmse) / base_nrmse
        best_bic, best_bic_id = self.best_bic()
        return {
            "model_id": model_id,
            "ok": scored.ok,
            "terms": scored.terms,
            "coeffs": [_num(c) for c in scored.coeffs],
            "train_rmse": _num(scored.train_rmse),
            "held_nrmse": _num(scored.held_nrmse),
            "bic": _num(scored.bic),
            "n_scored": scored.n_scored,
            "held_nrmse_improvement_vs_baseline": improvement,
            "screening": screening,
            "best_bic_so_far": best_bic,
            "best_bic_model_id": best_bic_id,
            "error": scored.error,
        }

    def plot_residuals(self, model_id: str, versus: str = "t") -> dict[str, Any]:
        entry = self.models.get(model_id)
        if entry is None:
            return {"model_id": model_id, "error": f"unknown model_id: {model_id}"}
        scored = entry.scored
        if not scored.ok:
            return {"model_id": model_id,
                    "error": f"model did not fit: {scored.error}"}
        if versus not in ("t", "q", "v"):
            return {"model_id": model_id, "error": f"versus must be t, q, or v, got {versus!r}"}
        pred, _ = P.simulate(scored.terms, scored.coeffs, self.t_fit,
                             (scored.q0, scored.v0))
        residuals = self.q_fit - pred
        x = {"t": self.t_fit, "q": self.q_fit,
             "v": P.estimate_velocity(self.t_fit, self.q_fit)}[versus]
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.scatter(x, residuals, s=12)
        ax.axhline(0.0, linestyle="--", linewidth=0.8)
        ax.set_xlabel(versus)
        ax.set_ylabel("residual (observed - predicted)")
        ax.set_title(f"{model_id} residuals vs {versus}: q'' = "
                     + " + ".join(f"{c:.4g}*{t}" for c, t in
                                  zip(scored.coeffs, scored.terms)))
        plot_dir = self.out_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        path = plot_dir / f"residual_{model_id}_{versus}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return {"model_id": model_id, "versus": versus, "path": str(path)}

    def _guard_reference(self):
        """What a submission must beat: the baseline, or gravity-only.

        Benchmark loophole (closed): when the baseline's own rollout
        exploded, the old guard accepted ANY fitted model -- a model
        worse than plain gravity was submitted on two Morpheus clips.
        Gravity-only is the standing control, so it becomes the
        reference; only if it also fails is there nothing to beat.
        """
        if self.baseline.ok:
            return self.baseline, "baseline"
        cached = getattr(self, "_gravity_reference", None)
        if cached is None:
            cached = P.score_candidate(["1"], self.t_fit, self.q_fit,
                                       self.t_held, self.q_held,
                                       budget_s=self.budget_s)
            self._gravity_reference = cached
        if cached.ok:
            return cached, "gravity-only (baseline failed)"
        return None, "none"

    def submit_model(self, model_id: str, explanation: str = "") -> dict[str, Any]:
        entry = self.models.get(model_id)
        if entry is None:
            return {"accepted": False, "error": f"unknown model_id: {model_id}"}
        scored = entry.scored
        if not scored.ok:
            return {"accepted": False,
                    "error": f"model did not fit: {scored.error}"}
        if model_id == self.baseline_id:
            self.submitted_id = model_id
            self.submit_explanation = explanation
            return {"accepted": True, "model_id": model_id,
                    "screening": False,
                    "terms": scored.terms,
                    "coeffs": [_num(c) for c in scored.coeffs],
                    "held_nrmse": _num(scored.held_nrmse),
                    "bic": _num(scored.bic),
                    "train_rmse": _num(scored.train_rmse)}
        base, reference_name = self._guard_reference()
        if base is None:
            self.submitted_id = model_id
            self.submit_explanation = explanation
            return {"accepted": True, "model_id": model_id,
                    "screening": False,
                    "reference": reference_name,
                    "terms": scored.terms,
                    "coeffs": [_num(c) for c in scored.coeffs],
                    "held_nrmse": _num(scored.held_nrmse),
                    "bic": _num(scored.bic),
                    "train_rmse": _num(scored.train_rmse)}
        # Audit A4: the guard runs on tight scores, never on the screen.
        tight = self.tighten(model_id)
        if tight is None:
            entry = self.models[model_id]
            return {"accepted": False,
                    "error": "tight refit failed: "
                             f"{entry.scored.error or 'no tight score'}"}
        scored = tight
        beats_nrmse = scored.held_nrmse <= (1.0 - SUBMIT_IMPROVEMENT) * base.held_nrmse
        beats_bic = scored.bic < base.bic
        if self.guard == "v2":
            train_ok = (math.isfinite(scored.train_rmse)
                        and math.isfinite(base.train_rmse) and base.train_rmse > 0
                        and scored.train_rmse <= 1.05 * base.train_rmse)
            if not (beats_nrmse and beats_bic and train_ok):
                return {
                    "accepted": False, "reference": reference_name,
                    "error": "does not beat the baseline by >= 2% held-out "
                             "NRMSE AND on BIC with train RMSE within 5%",
                    "held_nrmse": _num(scored.held_nrmse),
                    "baseline_held_nrmse": _num(base.held_nrmse),
                    "bic": _num(scored.bic),
                    "baseline_bic": _num(base.bic),
                    "train_rmse": _num(scored.train_rmse),
                    "baseline_train_rmse": _num(base.train_rmse),
                }
        elif self.guard == "v3":
            if not (beats_nrmse and beats_bic):
                return {
                    "accepted": False, "reference": reference_name,
                    "error": "does not beat the baseline by >= 2% held-out "
                             "NRMSE AND on BIC (train RMSE is not part of "
                             "the v3 decision)",
                    "held_nrmse": _num(scored.held_nrmse),
                    "baseline_held_nrmse": _num(base.held_nrmse),
                    "bic": _num(scored.bic),
                    "baseline_bic": _num(base.bic),
                }
        elif not (beats_nrmse or beats_bic):
            return {
                "accepted": False, "reference": reference_name,
                "error": "does not beat the baseline by >= 2% held-out "
                         "NRMSE or on BIC",
                "held_nrmse": _num(scored.held_nrmse),
                "baseline_held_nrmse": _num(base.held_nrmse),
                "bic": _num(scored.bic),
                "baseline_bic": _num(base.bic),
            }
        self.submitted_id = model_id
        self.submit_explanation = explanation
        return {"accepted": True, "model_id": model_id,
                "screening": False,
                "terms": scored.terms,
                "coeffs": [_num(c) for c in scored.coeffs],
                "held_nrmse": _num(scored.held_nrmse), "bic": _num(scored.bic),
                "train_rmse": _num(scored.train_rmse)}

    def best_model_id(self) -> str:
        """Lowest held-NRMSE fitted model (the baseline always qualifies)."""
        best_id, best_score = self.baseline_id, float("inf")
        for model_id, entry in self.models.items():
            scored = entry.scored
            if scored.ok and scored.held_nrmse < best_score:
                best_id, best_score = model_id, scored.held_nrmse
        return best_id


@dataclass
class ScientistResult:
    final_model_id: str
    terms: list[str]
    coeffs: list[float | None]
    submitted: bool
    explanation: str
    rounds: int
    cost_usd: float
    stopped_reason: str
    transcript: list[dict[str, Any]] = field(default_factory=list)
    axis_terms: dict[str, list[str]] | None = None
    axis_coeffs: dict[str, list[float | None]] | None = None
    # Final-model scores, so committed runner rows are numerically
    # re-auditable without the (gitignored) transcripts.
    held_nrmse: float | None = None
    bic: float | None = None
    train_rmse: float | None = None


def default_chat_fn(reasoning_effort: str = "low") -> Callable[..., Any]:
    """Chat Completions via meta.chat; tool_choice is always "auto"."""
    from .meta import chat

    def call(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Any:
        return chat(messages, tools=tools, tool_choice="auto",
                    max_tokens=2048, reasoning_effort=reasoning_effort)

    return call


def _usage_cost(response: Any) -> float:
    from .meta import compute_cost_usd

    usage = getattr(response, "usage", None)
    if usage is None:
        return 0.0
    return compute_cost_usd(int(getattr(usage, "prompt_tokens", 0) or 0),
                            int(getattr(usage, "completion_tokens", 0) or 0))


# --- Projectile (2-DOF) scientist mode: Section 4.5b ------------------------

PROJECTILE_MENU_X = [P.DRAG_TERM_X, "vx**3", "vx"]
PROJECTILE_MENU_Y = [P.DRAG_TERM_Y, "vy**3", "vy"]

PROJECTILE_SYSTEM_PROMPT = """You are a physicist studying projectile motion in two dimensions, x(t) and y(t), extracted from video.

You work through tools only: fit_model proposes per-axis structures and
fits them classically (you never guess coefficients), plot_residuals
shows you where a model fails, and submit_model ends the run.

Coordinates: x is right and y is UP (already converted from image
rows), so gravity is a negative constant term on y and no force on x.
The state is [x, y, vx, vy]. Each axis has its own acceleration model:
x'' = sum of coefficient * term, y'' = sum of coefficient * term, where
terms are expressions in x, y, vx, vy (numbers, + - * / **, and
sin cos tanh sign abs sqrt exp). The speed is |v| = sqrt(vx**2 + vy**2).

The baseline is x'': [1, x, vx, vx**2] and y'': [1, y, vy, vy**2]. Its
unsigned squared terms cannot express drag coupled through the speed.
The candidate missing physics is quadratic drag, which acts along each
velocity component in proportion to the total speed:
  x'': sqrt(vx**2+vy**2)*vx     y'': sqrt(vx**2+vy**2)*vy
with the same (negative) coefficient expected on both axes. Linear drag
(vx on x, vy on y) is the weaker alternative. Gravity belongs on y only.

Rules:
- Justify each added term physically; a drag coefficient represents
  drag per unit mass (1/length for quadratic, 1/time for linear).
- The same drag coefficient should appear on both axes; if your fitted
  coefficients disagree wildly in sign, the structure is probably wrong.
- Prefer fewer terms: selection uses held-out NRMSE and BIC over both
  axes, so extra terms must earn their place on unseen motion.
- submit_model is rejected unless the model beats the baseline by at
  least 2% held-out NRMSE or on BIC. Gravity-only on y (no horizontal
  force) is the control every drag claim is implicitly compared against.
- Finish by calling submit_model. Do not end with a plain-text answer."""

TOOL_SCHEMAS_2D: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "fit_model",
            "description": "Fit per-axis candidate term sets on the fit "
                           "window and score on the held-out horizon.",
            "parameters": {
                "type": "object",
                "properties": {
                    "terms_x": {"type": "array", "items": {"type": "string"},
                                "description": "Terms for x'' in x, y, vx, vy."},
                    "terms_y": {"type": "array", "items": {"type": "string"},
                                "description": "Terms for y'' in x, y, vx, vy."},
                    "rationale": {"type": "string",
                                  "description": "Physical justification."},
                },
                "required": ["terms_x", "terms_y", "rationale"],
            },
        },
    },
    TOOL_SCHEMAS[1],
    TOOL_SCHEMAS[2],
]


class ProjectileTools:
    """The 3 scientist tools over a 2-DOF fit/held-out split."""

    def __init__(self, t_fit: np.ndarray, x_fit: np.ndarray,
                 y_fit: np.ndarray, t_held: np.ndarray,
                 x_held: np.ndarray, y_held: np.ndarray,
                 out_dir: Path, budget_s: float = LOOP_FIT_BUDGET_S,
                 fit_cache: dict[str, Any] | None = None,
                 baseline: P.ScoredFit2D | None = None,
                 ) -> None:
        self.t_fit = np.asarray(t_fit, dtype=float)
        self.x_fit = np.asarray(x_fit, dtype=float)
        self.y_fit = np.asarray(y_fit, dtype=float)
        self.t_held = np.asarray(t_held, dtype=float)
        self.x_held = np.asarray(x_held, dtype=float)
        self.y_held = np.asarray(y_held, dtype=float)
        self.out_dir = Path(out_dir)
        self.budget_s = budget_s
        self.models: dict[str, _Entry] = {}
        self._next_id = 0
        self.submitted_id: str | None = None
        self.submit_explanation = ""
        # Audit A1/A3: same memo and loop-budget treatment as ScientistTools.
        # Two-stage scoring (A4) and guard v2 (B6) are 1-DOF only for now.
        self._fit_cache: dict[str, Any] = fit_cache if fit_cache is not None else {}
        # Audit A2 (extended to 2-DOF): the 8-term projectile baseline needs
        # the full 60 s to converge on 47-pt windows; the caller precomputes
        # it instead of refitting per loop at the loop budget.
        if baseline is None:
            baseline = P.score_projectile(P.PROJECTILE_BASELINE_X,
                                          P.PROJECTILE_BASELINE_Y,
                                          self.t_fit, self.x_fit, self.y_fit,
                                          self.t_held, self.x_held, self.y_held,
                                          budget_s=budget_s)
            self.baseline_id = self._register(baseline, "baseline library")
        else:
            self.baseline_id = self._register(baseline, "precomputed baseline")

    def _register(self, scored: P.ScoredFit2D, rationale: str) -> str:
        model_id = f"model-{self._next_id}"
        self._next_id += 1
        self.models[model_id] = _Entry(scored, rationale)
        return model_id

    @property
    def baseline(self) -> P.ScoredFit2D:
        return self.models[self.baseline_id].scored

    def summary(self) -> dict[str, Any]:
        base = self.baseline
        return {
            "mode": "projectile-2d",
            "n_fit": len(self.t_fit),
            "n_held": len(self.x_held),
            "baseline_id": self.baseline_id,
            "baseline_terms_x": base.terms_x,
            "baseline_terms_y": base.terms_y,
            "baseline_ok": base.ok,
            "baseline_held_nrmse": _num(base.held_nrmse),
            "baseline_bic": _num(base.bic),
            "baseline_error": base.error,
            "menu_terms_x": PROJECTILE_MENU_X,
            "menu_terms_y": PROJECTILE_MENU_Y,
        }

    def fit_model(self, terms_x: Any = None, terms_y: Any = None,
                  rationale: str = "", terms: Any = None) -> dict[str, Any]:
        key = (ScientistTools._cache_key(terms_x),
               ScientistTools._cache_key(terms_y))
        cached = self._fit_cache.get(key)
        if cached is not None and cached.ok:
            scored = cached
        else:
            scored = P.score_projectile(terms_x, terms_y,
                                        self.t_fit, self.x_fit, self.y_fit,
                                        self.t_held, self.x_held, self.y_held,
                                        budget_s=self.budget_s)
            self._fit_cache[key] = scored
        model_id = self._register(scored, rationale)
        base_nrmse = self.baseline.held_nrmse
        improvement = None
        if scored.ok and math.isfinite(base_nrmse) and base_nrmse > 0:
            improvement = (base_nrmse - scored.held_nrmse) / base_nrmse
        return {
            "model_id": model_id, "ok": scored.ok,
            "terms_x": scored.terms_x, "terms_y": scored.terms_y,
            "coeffs_x": [_num(c) for c in scored.coeffs_x],
            "coeffs_y": [_num(c) for c in scored.coeffs_y],
            "train_rmse": _num(scored.train_rmse),
            "held_nrmse": _num(scored.held_nrmse), "bic": _num(scored.bic),
            "n_scored": scored.n_scored,
            "held_nrmse_improvement_vs_baseline": improvement,
            "error": scored.error,
        }

    def plot_residuals(self, model_id: str, versus: str = "t") -> dict[str, Any]:
        entry = self.models.get(model_id)
        if entry is None:
            return {"model_id": model_id, "error": f"unknown model_id: {model_id}"}
        scored = entry.scored
        if not scored.ok:
            return {"model_id": model_id,
                    "error": f"model did not fit: {scored.error}"}
        if versus not in ("t", "x", "y"):
            return {"model_id": model_id,
                    "error": f"versus must be t, x, or y, got {versus!r}"}
        px, py, _, _ = P.simulate_projectile(
            scored.terms_x, scored.terms_y, scored.coeffs_x,
            scored.coeffs_y, self.t_fit, scored.state0)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, sharex=True)
        coord = {"t": self.t_fit, "x": self.x_fit, "y": self.y_fit}[versus]
        axes[0].scatter(coord, self.x_fit - px, s=12)
        axes[0].axhline(0.0, linestyle="--", linewidth=0.8)
        axes[0].set_ylabel("x residual")
        axes[1].scatter(coord, self.y_fit - py, s=12)
        axes[1].axhline(0.0, linestyle="--", linewidth=0.8)
        axes[1].set_ylabel("y residual")
        axes[1].set_xlabel(versus)
        fig.suptitle(f"{model_id} residuals vs {versus}")
        plot_dir = self.out_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        path = plot_dir / f"residual_{model_id}_{versus}.png"
        fig.savefig(path, dpi=100)
        plt.close(fig)
        return {"model_id": model_id, "versus": versus, "path": str(path)}

    def submit_model(self, model_id: str, explanation: str = "") -> dict[str, Any]:
        entry = self.models.get(model_id)
        if entry is None:
            return {"accepted": False, "error": f"unknown model_id: {model_id}"}
        scored = entry.scored
        if not scored.ok:
            return {"accepted": False,
                    "error": f"model did not fit: {scored.error}"}
        base = self.baseline
        reference_name = "baseline"
        if model_id != self.baseline_id and not base.ok:
            # Same loophole as the 1-DOF guard: an exploded baseline
            # must not make every candidate submittable. Gravity-only
            # (no horizontal force) is the standing control.
            gravity = P.score_projectile(["0"], ["1"], self.t_fit,
                                         self.x_fit, self.y_fit,
                                         self.t_held, self.x_held,
                                         self.y_held,
                                         budget_s=self.budget_s)
            if gravity.ok:
                base, reference_name = gravity, "gravity-only (baseline failed)"
        if model_id != self.baseline_id and base.ok:
            beats_nrmse = scored.held_nrmse <= (1.0 - SUBMIT_IMPROVEMENT) * base.held_nrmse
            beats_bic = scored.bic < base.bic
            if not (beats_nrmse or beats_bic):
                return {"accepted": False,
                        "error": "does not beat the baseline by >= 2% "
                                 "held-out NRMSE or on BIC",
                        "reference": reference_name,
                        "held_nrmse": _num(scored.held_nrmse),
                        "baseline_held_nrmse": _num(base.held_nrmse),
                        "bic": _num(scored.bic),
                        "baseline_bic": _num(base.bic)}
        self.submitted_id = model_id
        self.submit_explanation = explanation
        return {"accepted": True, "model_id": model_id,
                "terms_x": scored.terms_x, "terms_y": scored.terms_y,
                "coeffs_x": [_num(c) for c in scored.coeffs_x],
                "coeffs_y": [_num(c) for c in scored.coeffs_y],
                "held_nrmse": _num(scored.held_nrmse),
                "bic": _num(scored.bic)}

    def best_model_id(self) -> str:
        best_id, best_score = self.baseline_id, float("inf")
        for model_id, entry in self.models.items():
            scored = entry.scored
            if scored.ok and scored.held_nrmse < best_score:
                best_id, best_score = model_id, scored.held_nrmse
        return best_id


def _dispatch(tools: ScientistTools, name: str,
              arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "fit_model":
        if isinstance(tools, ProjectileTools):
            return tools.fit_model(arguments.get("terms_x"),
                                   arguments.get("terms_y"),
                                   str(arguments.get("rationale", "")))
        return tools.fit_model(arguments.get("terms"),
                               str(arguments.get("rationale", "")))
    if name == "plot_residuals":
        return tools.plot_residuals(str(arguments.get("model_id", "")),
                                    str(arguments.get("versus", "t")))
    if name == "submit_model":
        return tools.submit_model(str(arguments.get("model_id", "")),
                                  str(arguments.get("explanation", "")))
    return {"error": f"unknown tool: {name}"}


def run_loop(tools: Any, chat_fn: Callable[..., Any],
             max_rounds: int = MAX_ROUNDS,
             cost_cap_usd: float = COST_CAP_USD,
             system_prompt: str | None = None,
             schemas: list[dict[str, Any]] | None = None,
             max_fit_calls: int = MAX_FIT_CALLS) -> ScientistResult:
    """Drive the scientist loop; never raises for model/tool failures.

    Works for both ScientistTools (1-DOF) and ProjectileTools (Section
    4.5b): pass PROJECTILE_SYSTEM_PROMPT and TOOL_SCHEMAS_2D for the
    projectile mode. Defaults preserve the 1-DOF behaviour.
    Audit A5: at most ``max_fit_calls`` fit_model dispatches per loop;
    further ones fail fast as data and are surfaced in the transcript.
    """
    projectile = isinstance(tools, ProjectileTools)
    summary = tools.summary()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt or (
            PROJECTILE_SYSTEM_PROMPT if projectile else SYSTEM_PROMPT)},
        {"role": "user", "content":
         "Data summary (fit / held-out split of one tracked clip): "
         + json.dumps(summary)
         + "\nThe baseline model's residual plots are attached below. "
           "Start by reading them to diagnose where the baseline fails, "
           "then call fit_model with a physically motivated structure, "
           "inspect residuals with plot_residuals, and finish "
           "with submit_model."},
    ]
    # Attach the baseline failure evidence up front so the first fit is a
    # diagnosis of something Spark has seen, not a guess from the menu.
    # Failures here must never break the loop (baseline may not have fit).
    baseline_plots: list[dict[str, Any]] = []
    try:
        for versus in (["t"] if projectile else ["t", "v"]):
            plotted = tools.plot_residuals(tools.baseline_id, versus)
            if plotted.get("path"):
                baseline_plots.append(plotted)
    except Exception:
        baseline_plots = []
    for plotted in baseline_plots:
        encoded = base64.b64encode(
            Path(plotted["path"]).read_bytes()).decode()
        messages.append({
            "role": "user",
            "content": [
                {"type": "text",
                 "text": "Baseline (model-0) residual plot vs "
                         f"{plotted['versus']}: diagnose this failure mode "
                         "before proposing a structure."},
                {"type": "image_url", "image_url":
                 {"url": f"data:image/png;base64,{encoded}"}},
            ],
        })
    schemas = schemas or (TOOL_SCHEMAS_2D if projectile else TOOL_SCHEMAS)
    transcript: list[dict[str, Any]] = [{
        "event": "start", "summary": summary,
        "baseline_plots": [p.get("path") for p in baseline_plots],
    }]
    cost_usd = 0.0
    stopped = "max rounds reached"
    stagnant = 0
    best_seen = float("inf")
    rounds_used = 0
    fit_calls = 0
    fit_cap_hit = False

    for round_no in range(1, max_rounds + 1):
        rounds_used = round_no
        if cost_usd >= cost_cap_usd:
            stopped = f"cost cap ${cost_cap_usd:.2f} reached"
            break
        response = chat_fn(messages, schemas)
        cost_usd += _usage_cost(response)
        message = response.choices[0].message
        tool_calls = list(getattr(message, "tool_calls", None) or [])
        transcript.append({"event": "round", "round": round_no,
                           "content": getattr(message, "content", None),
                           "tool_calls": [c.function.name for c in tool_calls],
                           "cost_usd": round(cost_usd, 6)})
        if not tool_calls:
            messages.append({"role": "assistant",
                             "content": getattr(message, "content", None) or ""})
            messages.append({"role": "user",
                             "content": "Call a tool: fit_model, "
                                        "plot_residuals, or submit_model."})
            continue
        messages.append({
            "role": "assistant",
            "content": getattr(message, "content", None),
            "tool_calls": [{
                "id": c.id, "type": "function",
                "function": {"name": c.function.name,
                             "arguments": c.function.arguments},
            } for c in tool_calls],
        })
        for call in tool_calls:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}
            if call.function.name == "fit_model" and fit_calls >= max_fit_calls:
                # Audit A5: fail fast as data; the API contract still
                # needs one tool message per tool call.
                result = {"model_id": None,
                          "error": f"fit-call budget exhausted ({max_fit_calls} "
                                   "fit_model calls per loop); inspect the fits "
                                   "so far and submit."}
                if not fit_cap_hit:
                    fit_cap_hit = True
                    transcript.append({"event": "fit_cap",
                                       "limit": max_fit_calls,
                                       "round": round_no})
            else:
                if call.function.name == "fit_model":
                    fit_calls += 1
                result = _dispatch(tools, call.function.name, arguments)
            transcript.append({"event": "tool", "round": round_no,
                               "name": call.function.name, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result)})
            if result.get("path"):
                encoded = base64.b64encode(
                    Path(result["path"]).read_bytes()).decode()
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": f"Residual plot from {call.function.name}:"},
                        {"type": "image_url", "image_url":
                         {"url": f"data:image/png;base64,{encoded}"}},
                    ],
                })
        if tools.submitted_id is not None:
            stopped = "submitted"
            break
        # Plot-only rounds are diagnosis, not failed improvement: they must
        # not consume the 2-round stagnation budget (Day-5 pilot H2: Spark
        # found the winning model in round 2, plotted in round 4, and was
        # stopped before submitting).
        if not any(c.function.name == "fit_model" for c in tool_calls):
            continue
        current_best = tools.models[tools.best_model_id()].scored.held_nrmse
        if math.isfinite(current_best) and current_best < 0.98 * best_seen:
            best_seen, stagnant = current_best, 0
        else:
            stagnant += 1
            if stagnant >= 2:
                stopped = "held-out NRMSE improved < 2% for 2 rounds"
                break

    submitted = tools.submitted_id is not None
    final_id = tools.submitted_id or tools.best_model_id()
    if fit_cap_hit and not submitted:
        stopped += "; fit-call cap reached"
    final = tools.models[final_id].scored
    tighten = getattr(tools, "tighten", None)
    if callable(tighten):
        # Audit A4: reported scores are always tight. Submitted models
        # were tightened on submit; the fallback best is tightened here.
        tight = tighten(final_id)
        if tight is not None:
            final = tight
        elif not submitted:
            final_id = tools.baseline_id
            final = tools.models[final_id].scored
            stopped += "; best model failed tight refit, fell back to baseline"
    if not submitted:
        stopped += "; submitted best-so-far model as fallback"
    if hasattr(final, "terms_x"):  # projectile mode
        axis_terms = {"x": final.terms_x, "y": final.terms_y}
        axis_coeffs = {"x": [_num(c) for c in final.coeffs_x],
                       "y": [_num(c) for c in final.coeffs_y]}
        flat_terms = final.terms_x + final.terms_y
        flat_coeffs = axis_coeffs["x"] + axis_coeffs["y"]
    else:
        axis_terms = axis_coeffs = None
        flat_terms = final.terms
        flat_coeffs = [_num(c) for c in final.coeffs]
    return ScientistResult(
        final_model_id=final_id,
        terms=flat_terms,
        coeffs=flat_coeffs,
        submitted=submitted,
        explanation=tools.submit_explanation if submitted else
        "loop ended without submit_model; best model by held-out NRMSE",
        rounds=rounds_used,
        cost_usd=cost_usd,
        stopped_reason=stopped,
        transcript=transcript,
        axis_terms=axis_terms,
        axis_coeffs=axis_coeffs,
        held_nrmse=_num(final.held_nrmse),
        bic=_num(final.bic),
        train_rmse=_num(final.train_rmse),
    )
