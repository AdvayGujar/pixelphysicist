"""Day-4 live gates G4 (1-DOF) and G4b (2-DOF projectile).

Runs 5 live Spark loops per gate on synthetic truth (reasoning_effort=low),
saves per-run transcripts under runs/day4_*, and writes results/day4.json.

Usage:
    .venv/bin/python scripts/run_day4.py [--runs 5] [--out results/day4.json]
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
from pixelphys import scientist as S  # noqa: E402
from pixelphys import synthetic as syn  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"


def run_g4(n_runs: int, effort: str) -> list[dict]:
    truth = json.loads((FIXTURES / "toss_drag.json").read_text())
    t = np.array(truth["t"])
    q = np.array(truth["q"])
    cut = int(len(t) * 0.6)
    rows = []
    for i in range(1, n_runs + 1):
        out_dir = REPO / f"runs/day4_g4_run{i}"
        (out_dir / "plots").mkdir(parents=True, exist_ok=True)
        tools = S.ScientistTools(t[:cut], q[:cut], t[cut:], q[cut:], out_dir)
        result = S.run_loop(tools, S.default_chat_fn(effort))
        ok_term = "abs(v)*v" in result.terms
        row = {
            "run": f"g4-{i}", "submitted": result.submitted,
            "terms": result.terms, "coeffs": result.coeffs,
            "rounds": result.rounds, "cost_usd": round(result.cost_usd, 6),
            "stopped": result.stopped_reason,
            "correct_term": bool(ok_term and result.submitted),
            "explanation": result.explanation,
        }
        (out_dir / "transcript.json").write_text(json.dumps(
            {"result": row, "transcript": result.transcript}, indent=1))
        print(f"G4 run {i}: terms={result.terms} submitted={result.submitted} "
              f"rounds={result.rounds} cost=${result.cost_usd:.4f} "
              f"stopped={result.stopped_reason}", flush=True)
        rows.append(row)
    return rows


def run_g4b(n_runs: int, effort: str) -> list[dict]:
    times, xs, ys, _, _ = syn.integrate_projectile(syn.ProjectileParams())
    sl = slice(None, None, 8)
    t = np.array(times[sl])
    x = np.array(xs[sl])
    y = np.array(ys[sl])
    cut = int(len(t) * 0.6)
    rows = []
    for i in range(1, n_runs + 1):
        out_dir = REPO / f"runs/day4_g4b_run{i}"
        (out_dir / "plots").mkdir(parents=True, exist_ok=True)
        tools = S.ProjectileTools(t[:cut], x[:cut], y[:cut],
                                  t[cut:], x[cut:], y[cut:], out_dir)
        result = S.run_loop(tools, S.default_chat_fn(effort),
                            system_prompt=S.PROJECTILE_SYSTEM_PROMPT,
                            schemas=S.TOOL_SCHEMAS_2D)
        ok_term = (result.axis_terms is not None
                   and P.DRAG_TERM_X in result.axis_terms["x"]
                   and P.DRAG_TERM_Y in result.axis_terms["y"])
        row = {
            "run": f"g4b-{i}", "submitted": result.submitted,
            "axis_terms": result.axis_terms, "axis_coeffs": result.axis_coeffs,
            "rounds": result.rounds, "cost_usd": round(result.cost_usd, 6),
            "stopped": result.stopped_reason,
            "correct_term": bool(ok_term and result.submitted),
            "explanation": result.explanation,
        }
        (out_dir / "transcript.json").write_text(json.dumps(
            {"result": row, "transcript": result.transcript}, indent=1))
        print(f"G4b run {i}: terms={result.axis_terms} submitted={result.submitted} "
              f"rounds={result.rounds} cost=${result.cost_usd:.4f} "
              f"stopped={result.stopped_reason}", flush=True)
        rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Day-4 live gates.")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--effort", default="low")
    parser.add_argument("--out", default="results/day4.json")
    parser.add_argument("--skip-g4", action="store_true")
    parser.add_argument("--skip-g4b", action="store_true")
    args = parser.parse_args(argv)

    all_rows: list[dict] = []
    g4 = [] if args.skip_g4 else run_g4(args.runs, args.effort)
    g4b = [] if args.skip_g4b else run_g4b(args.runs, args.effort)
    all_rows = g4 + g4b
    total_cost = round(sum(r["cost_usd"] for r in all_rows), 6)
    summary = {
        "date": date.today().isoformat(),
        "effort": args.effort,
        "g4_pass": f"{sum(r['correct_term'] for r in g4)}/{len(g4)}" if g4 else "skipped",
        "g4b_pass": f"{sum(r['correct_term'] for r in g4b)}/{len(g4b)}" if g4b else "skipped",
        "g4_gate": "GREEN" if g4 and sum(r["correct_term"] for r in g4) >= 3 else ("skipped" if not g4 else "RED"),
        "g4b_gate": "GREEN" if g4b and sum(r["correct_term"] for r in g4b) >= 3 else ("skipped" if not g4b else "RED"),
        "total_cost_usd": total_cost,
        "rows": all_rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print(f"G4: {summary['g4_pass']} -> {summary['g4_gate']}; "
          f"G4b: {summary['g4b_pass']} -> {summary['g4b_gate']}; "
          f"total ${total_cost:.4f}; wrote {out}")
    return 0 if summary["g4_gate"] in ("GREEN", "skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
