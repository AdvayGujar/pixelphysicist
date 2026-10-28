"""Reproduce the Day-3 gate numbers (audit F3): baseline + controls on S/H/N.

Split protocol fits on the first 60% and scores the min(2 s, tail) horizon
via score_candidate. Cross-take protocol (F4) fits coefficients on all of
hero take 5's free flight and scores them on take 4 via rollout_score.

Usage:
    .venv/bin/python scripts/run_baseline.py [--out results/baseline.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixelphys import physics as P  # noqa: E402
from pixelphys import trajectory as T  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
FRICTIONLESS = ["1"]
DRAG = ["1", "abs(v)*v"]


def load_free_flight(fixture: str, axis: str, trim: bool) -> tuple[np.ndarray, np.ndarray]:
    obs = T.load_observations(FIXTURES / fixture)
    obs = T.clean_observations(obs)
    obs = max(T.split_by_gaps(obs), key=len)
    t, cx, cy = T.resample_uniform(obs, 30.0)
    raw = cx if axis == "x" else cy
    q = T.rest_relative_q(T.smooth_position(raw)) if axis == "x" \
        else T.vertical_q(T.smooth_position(raw))
    if trim:
        start, end = T.trim_free_flight(t, q)
        t, q = t[start:end], q[start:end]
    return t, q


def split_scored(label: str, t: np.ndarray, q: np.ndarray,
                 models: dict[str, list[str]]) -> list[dict]:
    cut = int(len(t) * 0.6)
    rows = []
    for name, terms in models.items():
        scored = P.score_candidate(terms, t[:cut], q[:cut], t[cut:], q[cut:])
        rows.append({
            "clip": label, "protocol": "split-60/40",
            "model": name, "terms": scored.terms, "ok": scored.ok,
            "coeffs": [float(c) for c in scored.coeffs],
            "q0": scored.q0, "v0": scored.v0,
            "train_rmse": scored.train_rmse, "held_nrmse": scored.held_nrmse,
            "bic": scored.bic, "n_fit": cut, "n_scored": scored.n_scored,
            "cost": scored.cost, "start_index": scored.start_index,
            "error": scored.error,
        })
        print(f"{label:12s} split {name:8s} held {scored.held_nrmse:.4f} "
              f"bic {scored.bic:9.1f} ok {scored.ok}", flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproduce the Day-3 gate numbers.")
    parser.add_argument("--out", default="results/baseline.json")
    args = parser.parse_args(argv)

    rows: list[dict] = []
    truth = json.loads((FIXTURES / "toss_drag.json").read_text())
    t_s = np.array(truth["t"])
    q_s = np.array(truth["q"])
    rows += split_scored("synthetic", t_s, q_s, {
        "baseline": list(P.BASELINE_TERMS),
        "gravity-only": FRICTIONLESS,
        "true-terms": DRAG,
    })

    t5, q5 = load_free_flight("obs_hero_toss_take5.json", "y", trim=True)
    print(f"hero take5: {len(t5)} free-flight points", flush=True)
    rows += split_scored("hero-take5", t5, q5, {
        "baseline": list(P.BASELINE_TERMS),
        "gravity-only": FRICTIONLESS,
        "drag-term": DRAG,
    })

    t4, q4 = load_free_flight("obs_hero_toss_take4.json", "y", trim=True)
    print(f"hero take4: {len(t4)} free-flight points", flush=True)
    for name, terms in [("baseline", list(P.BASELINE_TERMS)),
                        ("gravity-only", FRICTIONLESS),
                        ("drag-term", DRAG)]:
        fit = P.fit_terms(terms, t5, q5)
        held, n_scored = P.rollout_score(list(fit.terms), fit.coeffs, t4, q4)
        train_pred, _ = P.simulate(list(fit.terms), fit.coeffs, t5, (fit.q0, fit.v0))
        rss = float(np.sum((train_pred - q5) ** 2))
        rows.append({
            "clip": "hero-take4", "protocol": "cross-take-fit-take5",
            "model": name, "terms": list(fit.terms), "ok": True,
            "coeffs": [float(c) for c in fit.coeffs],
            "q0": fit.q0, "v0": fit.v0,
            "train_rmse": float(np.sqrt(rss / len(q5))), "held_nrmse": held,
            "bic": P.bic(len(q5), rss, len(fit.terms) + 2),
            "n_fit": len(q5), "n_scored": n_scored,
            "cost": fit.cost, "start_index": fit.start_index, "error": None,
        })
        print(f"hero-take4  x-take {name:8s} held {held:.4f} coeffs "
              f"{np.round(fit.coeffs, 4)}", flush=True)

    t_n, q_n = load_free_flight("obs_pendulum_take1.json", "x", trim=False)
    print(f"pendulum: {len(t_n)} points", flush=True)
    rows += split_scored("pendulum", t_n, q_n, {
        "baseline": list(P.BASELINE_TERMS),
        "gravity-only": FRICTIONLESS,
    })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "date": date.today().isoformat(),
        "horizon": "min(2 s, clean tail)",
        "rows": rows,
    }, indent=1))
    print(f"wrote {out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
