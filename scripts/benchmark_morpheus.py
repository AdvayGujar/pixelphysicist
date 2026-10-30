"""Tier-2 Morpheus benchmark: one table across categories (runner + JSON).

Runs the existing CLI (``pixelphys run``) with ``--tracker motion`` (SAM
is the normal path on phone footage; motion is the offline fallback used
throughout the Morpheus work) on a fixed clip list, then records one row
per clip from each run's ``model.json``. Exit-2 refusals and failed fits
are rows too, with the error string -- never a crash. Falling-ball and
pendulum rows are cited from their committed result files, not rerun.

Usage:
    .venv/bin/python scripts/benchmark_morpheus.py [--run] [--out results/benchmark.json]
    Without --run, only assembles cited rows (no API calls).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENV_PY = REPO / ".venv" / "bin" / "pixelphys"

# Fixed clip list: local paths only (never commit footage).
LIVE_CLIPS: list[dict] = [
    {"key": "proj_a", "clip": "/tmp/stress/proj1.mp4",
     "experiment": "projectile (Morpheus /tmp/stress/proj1)",
     "mode": "projectile", "trim": True, "prompt": "small dark ball",
     "out": "runs/benchmark_proj_a"},
    {"key": "proj_b", "clip": "/tmp/stress2/proj5.mp4",
     "experiment": "projectile/video_5 (Morpheus)",
     "mode": "projectile", "trim": True, "prompt": "small dark ball",
     "out": "runs/benchmark_proj_b"},
    {"key": "roll_orange", "clip": "/tmp/stress/roll0.mp4",
     "experiment": "rolling_orange (Morpheus /tmp/stress/roll0)",
     "mode": "rolling", "trim": False, "prompt": "orange",
     "out": "runs/benchmark_roll_orange"},
    {"key": "spring_long", "clip": "/tmp/stress/spring0.mp4",
     "experiment": "spring/video_0 (Morpheus, 31 s)",
     "mode": "spring", "trim": False, "prompt": "hanging metal weight",
     "out": "runs/benchmark_spring_long"},
]


def _sanitize(value):
    """Strict-JSON output: non-finite floats become null, never Infinity."""
    import math

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return value


def _row_from_model(spec: dict) -> dict:
    """Assemble a benchmark row from an existing run directory (no API)."""
    out_dir = REPO / spec["out"]
    model_path = out_dir / "model.json"
    base_row = {"key": spec["key"], "experiment": spec["experiment"],
                "mode": spec["mode"], "tracker": "motion",
                "source": "live CLI run"}
    if not model_path.exists():
        return {**base_row, "submitted": False, "final_terms": [],
                "final_held_nrmse": None, "control_held_nrmse": None,
                "baseline_held_nrmse": None, "rounds": 0, "cost_usd": 0.0,
                "error": "no model.json in %s" % spec["out"]}
    try:
        model = json.loads(model_path.read_text())
    except (OSError, ValueError) as exc:
        return {**base_row, "submitted": False, "final_terms": [],
                "final_held_nrmse": None, "control_held_nrmse": None,
                "baseline_held_nrmse": None, "rounds": 0, "cost_usd": 0.0,
                "error": f"unreadable model.json: {exc}"[:300]}
    fin = model.get("final", {})
    if "terms_x" in fin or "terms_y" in fin:
        terms = (list(fin.get("terms_x", []))
                 + ["|"]
                 + list(fin.get("terms_y", [])))
    else:
        terms = list(fin.get("terms", []))
    base = model.get("baseline", {})
    return {**base_row,
            "tracker": model.get("tracker", "motion"),
            "submitted": bool(model.get("submitted", False)),
            "final_terms": terms,
            "final_held_nrmse": fin.get("held_nrmse"),
            "control_held_nrmse": (model.get("gravity_only", {}) or {}).get(
                "held_nrmse"),
            "baseline_held_nrmse": base.get("held_nrmse"),
            "rounds": model.get("rounds", 0),
            "cost_usd": model.get("cost_usd", 0.0),
            "error": None}


def run_cli(spec: dict) -> dict:
    """Run one CLI invocation; return the benchmark row (failure = row)."""
    cmd = [str(VENV_PY), "run", spec["clip"], "--prompt", spec["prompt"],
           "--tracker", "motion", "--mode", spec["mode"],
           "--effort", "low", "--out", spec["out"]]
    if spec["trim"]:
        cmd.append("--trim")
    for attempt in (1, 2):
        proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                              timeout=1200)
        tail = (proc.stdout + proc.stderr)[-2000:]
        if "transport or API error" in tail and attempt == 1:
            time.sleep(15)
            continue
        break
    row = _row_from_model(spec)
    if proc.returncode != 0 and row["error"] is None and row["final_held_nrmse"] is None:
        tail = (proc.stdout + proc.stderr)[-2000:]
        row["error"] = "exit %d; %s" % (
            proc.returncode, tail.strip().splitlines()[-1]
            if tail.strip() else "no output")[:300]
    return row


def cited_rows() -> list[dict]:
    """Falling-ball + pendulum rows cited from committed files (no rerun)."""
    rows: list[dict] = []
    mining = json.loads((REPO / "results" / "dataset_mining.json").read_text())
    for key in ("ball5", "ball8"):
        entry = mining["clips"][key]
        drag = entry["scores"]["drag"]
        grav = entry["scores"]["gravity"]
        base = entry["scores"]["baseline"]
        rows.append({
            "key": key, "experiment": entry["experiment"],
            "mode": "1d", "tracker": "motion", "submitted": None,
            "final_terms": ["1", "abs(v)*v"],
            "final_held_nrmse": drag["held_nrmse"],
            "control_held_nrmse": grav["held_nrmse"],
            "baseline_held_nrmse": base["held_nrmse"],
            "rounds": None, "cost_usd": 0.0, "error": None,
            "source": "cited from results/dataset_mining.json (offline "
                      "score_candidate; live Spark runs in docs/results.md §7 "
                      "submitted non-drag variants under the v1 prompt and "
                      "sparse drag under prompt v2)"})
    pend = json.loads((REPO / "results" / "pendulum_consistency.json")
                      .read_text())
    agree = {k: v["rel_err_corrected"] for k, v in pend["clips"].items()}
    rows.append({"key": "pendulum_x5",
                 "experiment": "holonomic_pendulum/video_1-5 (Morpheus)",
                 "mode": "1d (angle-period consistency, not a Spark run)",
                 "tracker": "motion", "submitted": None,
                 "final_terms": ["sin(q) period check"],
                 "final_held_nrmse": None,
                 "control_held_nrmse": None, "baseline_held_nrmse": None,
                 "rounds": None, "cost_usd": 0.0, "error": None,
                 "source": "cited from results/pendulum_consistency.json: "
                           "5/5 period agreement within 1%% (%s)" % ", ".join(
                               f"{k} {v:.2%}" for k, v in agree.items())})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Morpheus benchmark table.")
    parser.add_argument("--run", action="store_true",
                        help="execute the live CLI runs (API calls)")
    parser.add_argument("--out", default="results/benchmark.json")
    args = parser.parse_args(argv)
    rows: list[dict] = []
    if args.run:
        for spec in LIVE_CLIPS:
            print(f"benchmark: running {spec['key']} ...", flush=True)
            rows.append(run_cli(spec))
    else:
        # No API: assemble live rows from existing run dirs (if present).
        for spec in LIVE_CLIPS:
            out_dir = REPO / spec["out"]
            if out_dir.exists():
                rows.append(_row_from_model(spec))
    rows.extend(cited_rows())
    total_cost = round(sum(float(r.get("cost_usd") or 0.0) for r in rows), 6)
    payload = {"date": date.today().isoformat(),
               "tracker": "motion (SAM is the normal path on phone footage; "
                          "motion is the offline fallback for dataset clips)",
               "effort": "low", "guard": "v1 (CLI default)",
               "total_cost_usd": total_cost,
               "rows": rows}
    out_path = REPO / args.out
    out_path.write_text(json.dumps(_sanitize(payload), indent=1))
    print(f"wrote {args.out} ({len(rows)} rows, ${total_cost:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
