"""M12: Day-5 ablation runner -- baseline (A) vs baseline+Spark (B).

For each frozen condition in experiments.ablation (H/N/S), scores the
baseline once (condition A) and runs N_REPEATS live Spark loops
(condition B, reasoning_effort="low"). Writes results/ablation.csv
(one row per run) and results/ablation.json (gate evaluation).

Usage:
    .venv/bin/python scripts/run_ablation.py [--repeats 3] [--effort low]
        [--conditions H,N,S] [--out-csv results/ablation.csv]
        [--out-json results/ablation.json] [--prefix ablation]

Every B row carries terms, coeffs, held_nrmse, bic, rounds, cost --
the same fields as the Day-4 rows. The synthetic (S) condition runs
from TRACKED observations through the Day-3 pipeline, never from
truth arrays (Day-4 audit finding 2).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from ablation import CONDITIONS, EFFORT, load_condition  # noqa: E402
from pixelphys import physics as P  # noqa: E402
from pixelphys import scientist as S  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def split_60_40(t: np.ndarray, q: np.ndarray):
    cut = int(len(t) * 0.6)
    return (t[:cut], q[:cut], t[cut:], q[cut:])


def run_condition(name: str, repeats: int, effort: str, prefix: str,
                  guard: str = "v1") -> list[dict]:
    t, q = load_condition(name)
    t_fit, q_fit, t_held, q_held = split_60_40(t, q)
    rows: list[dict] = []

    base = P.score_candidate(list(P.BASELINE_TERMS), t_fit, q_fit, t_held, q_held)
    rows.append({
        "clip": name, "repeat": 0, "mode": "A",
        "submitted": True,
        "terms": list(base.terms),
        "coeffs": [float(c) for c in base.coeffs],
        "held_nrmse": float(base.held_nrmse),
        "bic": float(base.bic),
        "train_rmse": float(base.train_rmse),
        "rounds": 0, "cost_usd": 0.0,
        "stopped": "baseline (condition A)",
        "explanation": "baseline library [1, q, v, v**2], no Spark",
        "baseline_held_nrmse": float(base.held_nrmse),
        "baseline_bic": float(base.bic),
    })
    print(f"{name}: A baseline held={base.held_nrmse:.4f} bic={base.bic:.1f} "
          f"(n_fit={len(t_fit)} n_held={len(t_held)})", flush=True)

    # Audit A1: one memo dict shared across repeats of this condition
    # (identical data and splits, so term-tuple keys are unambiguous).
    fit_cache: dict[str, Any] = {}
    for i in range(1, repeats + 1):
        out_dir = REPO / f"runs/{prefix}_{name}_r{i}"
        existing = out_dir / "transcript.json"
        if existing.exists():
            # Resume: a previous invocation completed this repeat; reuse it
            # so a transport crash (e.g. a transient 400) never loses
            # finished runs or repeats their API cost.
            prior = json.loads(existing.read_text()).get("result", {})
            row = {
                "clip": name, "repeat": i, "mode": "B",
                "submitted": bool(prior.get("submitted")),
                "terms": list(prior.get("terms", [])),
                "coeffs": list(prior.get("coeffs", [])),
                "held_nrmse": float(prior.get("held_nrmse", float("inf"))),
                "bic": float(prior.get("bic", float("inf"))),
                "train_rmse": float(prior.get("train_rmse", float("inf"))),
                "rounds": int(prior.get("rounds", 0)),
                "cost_usd": float(prior.get("cost_usd", 0.0)),
                "stopped": str(prior.get("stopped", "resumed")),
                "explanation": str(prior.get("explanation", "")),
                "baseline_held_nrmse": float(prior.get(
                    "baseline_held_nrmse", float("inf"))),
                "baseline_bic": float(prior.get("baseline_bic", float("inf"))),
            }
            print(f"{name} B r{i}: resumed from {existing} "
                  f"terms={row['terms']} held={row['held_nrmse']:.4f}", flush=True)
            rows.append(row)
            continue
        (out_dir / "plots").mkdir(parents=True, exist_ok=True)
        # Transport errors (e.g. a transient 400 from the API) are not
        # data: retry the repeat once after a short wait instead of
        # losing the whole ablation. Two consecutive transport failures
        # are recorded as an error row so the remaining repeats still run.
        result, tools, attempt_error = None, None, ""
        for attempt in (1, 2):
            try:
                # Audit A1/A2: shared memo plus the precomputed condition-A
                # baseline instead of one refit per repeat.
                tools = S.ScientistTools(t_fit, q_fit, t_held, q_held, out_dir,
                                         baseline=base, fit_cache=fit_cache,
                                         guard=guard)
                if guard == "v2":
                    result = S.run_loop(tools, S.default_chat_fn(effort),
                                        system_prompt=S.V2_SYSTEM_PROMPT,
                                        schemas=S.v2_schemas())
                else:
                    result = S.run_loop(tools, S.default_chat_fn(effort))
                attempt_error = ""
                break
            except RuntimeError as exc:
                attempt_error = str(exc)[:300]
                print(f"{name} B r{i}: attempt {attempt} transport error: "
                      f"{attempt_error}", flush=True)
                if attempt == 1:
                    import time as _time
                    _time.sleep(15)
        if result is None:
            rows.append({
                "clip": name, "repeat": i, "mode": "B",
                "submitted": False, "terms": [], "coeffs": [],
                "held_nrmse": float("inf"), "bic": float("inf"),
                "train_rmse": float("inf"), "rounds": 0, "cost_usd": 0.0,
                "stopped": f"transport error after 2 attempts: {attempt_error}",
                "explanation": "no result; transport failed twice",
                "baseline_held_nrmse": float("inf"),
                "baseline_bic": float("inf"),
            })
            continue
        row = {
            "clip": name, "repeat": i, "mode": "B",
            "submitted": bool(result.submitted),
            "terms": list(result.terms),
            "coeffs": [float(c) if c is not None else float("nan")
                       for c in result.coeffs],
            "held_nrmse": float(result.held_nrmse),
            "bic": float(result.bic),
            "train_rmse": float(result.train_rmse),
            "rounds": int(result.rounds), "cost_usd": round(float(result.cost_usd), 6),
            "stopped": result.stopped_reason,
            "explanation": result.explanation,
            "baseline_held_nrmse": float(tools.baseline.held_nrmse),
            "baseline_bic": float(tools.baseline.bic),
        }
        (out_dir / "transcript.json").write_text(json.dumps(
            {"result": row, "transcript": result.transcript}, indent=1))
        print(f"{name} B r{i}: terms={result.terms} held={result.held_nrmse:.4f} "
              f"(base {tools.baseline.held_nrmse:.4f}) submitted={result.submitted} "
              f"rounds={result.rounds} cost=${result.cost_usd:.4f}", flush=True)
        rows.append(row)
    return rows


def evaluate(rows: list[dict]) -> dict:
    """G5 gate evaluation per plan Section 6 (amended Day-3 decision).

    Majority rule scales with repeats: 5-repeat conditions need 3 wins,
    3-repeat conditions need 2 (the pre-registered v1 bar).
    """
    by_clip: dict[str, list[dict]] = {}
    for r in rows:
        by_clip.setdefault(r["clip"], []).append(r)

    def mean_held(rs: list[dict]) -> float:
        vals = [r["held_nrmse"] for r in rs if np.isfinite(r["held_nrmse"])]
        return float(np.mean(vals)) if vals else float("inf")

    gates: dict[str, dict] = {}
    for name in ("H", "N", "S"):
        cr = by_clip.get(name, [])
        a = [r for r in cr if r["mode"] == "A"]
        b = [r for r in cr if r["mode"] == "B"]
        a_held = a[0]["held_nrmse"] if a else float("inf")
        b_mean = mean_held(b)
        n_beat = sum(1 for r in b if r["held_nrmse"] < a_held) if np.isfinite(a_held) else 0
        need = 3 if len(b) >= 5 else 2
        if name == "H":
            # B mean >=10% lower than A, and a majority of repeats beat A.
            # No claim vs gravity-only on real arcs (Day-3 decision).
            improvement = (a_held - b_mean) / a_held if a_held > 0 else 0.0
            passed = bool(np.isfinite(a_held) and improvement >= 0.10 and n_beat >= need)
            gates[name] = {"a_held": a_held, "b_mean": b_mean,
                           "improvement": improvement, "n_beat": f"{n_beat}/{len(b)}",
                           "gate": "GREEN" if passed else "RED"}
        elif name == "N":
            # B within 5% of A and no added terms in a majority of repeats.
            rel = abs(b_mean - a_held) / a_held if a_held > 0 else float("inf")
            n_clean = sum(1 for r in b
                          if list(r["terms"]) == list(P.BASELINE_TERMS)) if b else 0
            passed = bool(np.isfinite(a_held) and rel <= 0.05 and n_clean >= need)
            gates[name] = {"a_held": a_held, "b_mean": b_mean,
                           "rel_gap": rel, "n_clean": f"{n_clean}/{len(b)}",
                           "gate": "GREEN" if passed else "RED"}
        else:
            n_correct = sum(1 for r in b
                            if r["submitted"] and "abs(v)*v" in list(r["terms"]))
            passed = bool(n_correct >= need)
            gates[name] = {"a_held": a_held, "b_mean": b_mean,
                           "n_correct": f"{n_correct}/{len(b)}",
                           "gate": "GREEN" if passed else "RED"}
    g5 = all(g["gate"] == "GREEN" for g in gates.values())
    return {"gates": gates, "g5": "GREEN" if g5 else "RED"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Day-5 ablation.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--repeats-h", type=int, default=None,
                        help="override repeats for H (default: --repeats)")
    parser.add_argument("--repeats-n", type=int, default=None,
                        help="override repeats for N (default: --repeats)")
    parser.add_argument("--repeats-s", type=int, default=None,
                        help="override repeats for S (default: --repeats)")
    parser.add_argument("--guard", default="v1", choices=("v1", "v2"),
                        help="submit-guard version (v2 = audit B6 AND-rule)")
    parser.add_argument("--effort", default=EFFORT)
    parser.add_argument("--conditions", default="H,N,S")
    parser.add_argument("--out-csv", default="results/ablation.csv")
    parser.add_argument("--out-json", default="results/ablation.json")
    parser.add_argument("--prefix", default="ablation")
    args = parser.parse_args(argv)

    names = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for n in names:
        if n not in CONDITIONS:
            parser.error(f"unknown condition {n!r} (choose from {sorted(CONDITIONS)})")

    all_rows: list[dict] = []
    crashed: str = ""
    rep = {"H": args.repeats_h or args.repeats,
           "N": args.repeats_n or args.repeats,
           "S": args.repeats_s or args.repeats}
    try:
        for n in names:
            all_rows.extend(run_condition(n, rep[n], args.effort, args.prefix,
                                          guard=args.guard))
    except Exception as exc:  # noqa: BLE001 - always persist partial rows
        crashed = f"{type(exc).__name__}: {exc}"[:300]
        print(f"ablation crashed with {len(all_rows)} rows: {crashed}", flush=True)

    def _finite(value: float) -> float | None:
        return float(value) if np.isfinite(value) else None

    csv_path = REPO / args.out_csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["clip", "repeat", "mode", "submitted", "terms", "coeffs",
              "held_nrmse", "bic", "train_rmse", "rounds", "cost_usd",
              "stopped", "explanation", "baseline_held_nrmse", "baseline_bic"]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in all_rows:
            out = dict(r)
            out["terms"] = json.dumps(out["terms"])
            out["coeffs"] = json.dumps(out["coeffs"])
            for key in ("held_nrmse", "bic", "train_rmse", "cost_usd",
                        "baseline_held_nrmse", "baseline_bic"):
                if not np.isfinite(out[key]):
                    out[key] = ""
            writer.writerow(out)

    summary = {
        "date": date.today().isoformat(),
        "effort": args.effort,
        "repeats": args.repeats,
        "repeats_by_condition": rep,
        "guard": args.guard,
        "horizon": "min(2 s, clean tail)",
        "loop_fit_budget_s": S.LOOP_FIT_BUDGET_S,
        "screen_budget_s": S.SCREEN_BUDGET_S,
        "max_fit_calls": S.MAX_FIT_CALLS,
        "total_cost_usd": round(sum(r["cost_usd"] for r in all_rows
                                    if np.isfinite(r["cost_usd"])), 6),
        "crashed": crashed,
        **evaluate(all_rows),
        "rows": [{**r,
                  "held_nrmse": _finite(r["held_nrmse"]),
                  "bic": _finite(r["bic"]),
                  "train_rmse": _finite(r["train_rmse"]),
                  "cost_usd": _finite(r["cost_usd"]),
                  "baseline_held_nrmse": _finite(r["baseline_held_nrmse"]),
                  "baseline_bic": _finite(r["baseline_bic"])}
                 for r in all_rows],
    }
    json_path = REPO / args.out_json
    json_path.write_text(json.dumps(summary, indent=1))
    print(f"G5: {summary['g5']} "
          + " ".join(f"{k} {v['gate']}" for k, v in summary["gates"].items())
          + f"; total ${summary['total_cost_usd']:.4f}; "
          f"wrote {args.out_csv} + {args.out_json}"
          + (f"; CRASHED: {crashed}" if crashed else ""))
    if crashed:
        return 2
    return 0 if summary["g5"] == "GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
