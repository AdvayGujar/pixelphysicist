"""Audit B8: transfer checks for submitted ablation models (offline, no API).

Scores each submitted S model on a second synthetic toss (v0=7.5, tracked
the same way) and each submitted N model on pendulum take 2, via
``rollout_score`` (coefficients transfer, ICs refit on the new clip).
Spurious terms are expected to fail transfer; true terms to hold.

Usage:
    .venv/bin/python scripts/transfer_check.py [--out results/transfer.json]

Reads results/ablation.json and results/ablation_v2.json when present and
reports both side by side. Transfer is a separate column from G5: it never
rewrites a gate.
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

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"

EXPERIMENTS = ("ablation", "ablation_v2")


def load_q(fixture: str, axis: str) -> tuple[np.ndarray, np.ndarray]:
    """Same Day-3 pipeline as the frozen conditions (mirror, not edit)."""
    obs = T.load_observations(FIXTURES / fixture)
    gaps = np.diff([o.t for o in obs])
    fps = 1.0 / float(np.median(gaps))
    obs = T.clean_observations(obs)
    obs = max(T.split_by_gaps(obs), key=len)
    t, cx, cy = T.resample_uniform(obs, fps)
    raw = cx if axis == "x" else cy
    q = (T.rest_relative_q(T.smooth_position(raw)) if axis == "x"
         else T.vertical_q(T.smooth_position(raw)))
    return np.asarray(t, dtype=float), np.asarray(q, dtype=float)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run transfer checks.")
    parser.add_argument("--out", default="results/transfer.json")
    args = parser.parse_args(argv)

    t_s2, q_s2 = load_q("obs_synthetic_toss_drag_v0b.json", "y")
    t_n2, q_n2 = load_q("obs_pendulum_take2.json", "x")
    print(f"S take-2: {len(t_s2)} pts span {t_s2[-1]-t_s2[0]:.2f}s; "
          f"N take-2: {len(t_n2)} pts span {t_n2[-1]-t_n2[0]:.2f}s", flush=True)
    transfer = {"S": (t_s2, q_s2), "N": (t_n2, q_n2)}

    rows: list[dict] = []
    for exp in EXPERIMENTS:
        path = REPO / f"results/{exp}.json"
        if not path.exists():
            print(f"skip {exp}: {path} not present", flush=True)
            continue
        summary = json.loads(path.read_text())
        for row in summary.get("rows", []):
            if row.get("mode") != "B" or not row.get("submitted"):
                continue
            clip = row.get("clip")
            if clip not in transfer:
                continue
            t2, q2 = transfer[clip]
            coeffs = np.array(row["coeffs"], dtype=float)
            held, n = P.rollout_score(list(row["terms"]), coeffs, t2, q2)
            rows.append({
                "experiment": exp, "clip": clip, "repeat": row.get("repeat"),
                "terms": list(row["terms"]),
                "held_nrmse": row.get("held_nrmse"),
                "transfer_nrmse": float(held), "transfer_n_scored": int(n),
            })
            print(f"{exp} {clip} r{row.get('repeat')}: {list(row['terms'])} "
                  f"held={row.get('held_nrmse'):.4f} "
                  f"transfer={held:.4f} (n={n})", flush=True)

    def median(vals: list[float]) -> float | None:
        finite = [v for v in vals if np.isfinite(v)]
        return float(np.median(finite)) if finite else None

    summary_out = {
        "date": date.today().isoformat(),
        "note": ("Transfer (rollout_score, ICs refit) is a separate column "
                 "from G5; it never rewrites a gate. S take-2 shares the "
                 "renderer pixel scale exactly; N take-2 assumes the "
                 "same-session scale."),
        "datasets": {"S": "obs_synthetic_toss_drag_v0b.json (v0=7.5)",
                     "N": "obs_pendulum_take2.json"},
        "by_experiment_clip": {
            f"{exp}/{clip}": {
                "n": sum(1 for r in rows
                         if r["experiment"] == exp and r["clip"] == clip),
                "median_transfer_nrmse": median(
                    [r["transfer_nrmse"] for r in rows
                     if r["experiment"] == exp and r["clip"] == clip]),
            }
            for exp in EXPERIMENTS for clip in ("S", "N")
        },
        "rows": rows,
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary_out, indent=1))
    print(f"wrote {args.out} ({len(rows)} transfer rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
