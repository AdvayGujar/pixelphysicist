"""Unseen-clip check: offline physics scores on clips never used before.

Complements the Morpheus benchmark with clips that played no part in
any tuning, gate, or prompt decision (downloaded 2026-10-31). No Spark
calls: this measures the tracking + classical fitting layer on fresh
footage. Per category:
- falling_*: 1-DOF offline mining rule (baseline / gravity / drag,
  winner = drag <= 0.8*gravity AND < baseline AND coeff < 0 AND all ok)
- projectile: 2-DOF offline scores (baseline / gravity-only / drag)
- pendulum: scripts/pendulum_consistency.analyse (period agreement)
- rolling/sliding: 1-DOF along-x scores (baseline / constant / sign(v))

Clips live outside the repo (MIT dataset); pass their directory.

Usage:
    .venv/bin/python scripts/unseen_check.py [--clips-dir /tmp/stress3]
        [--out results/unseen_clips.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixelphys import physics as P  # noqa: E402
from pixelphys import trajectory as T  # noqa: E402
from pixelphys.cli import projectile_coordinates  # noqa: E402
from pixelphys.tracking import track_motion_clip  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

FALLING = ["fall3", "fall6", "fall9", "apple0"]
PROJECTILE = ["proj2", "proj3"]
ALONG_X = ["can0", "slide0"]


def sanitize(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    return value


def clean_grid(local: Path):
    obs = track_motion_clip(local)
    obs = T.clean_observations(obs)
    obs = max(T.split_by_gaps(obs), key=len)
    gaps = np.diff([o.t for o in obs])
    fps = 1.0 / float(np.median(gaps))
    t, cx, cy = T.resample_uniform(obs, fps)
    return t, T.smooth_position(cx), T.smooth_position(cy)


def held(scored) -> float | None:
    return float(scored.held_nrmse) if scored.ok else None


def check_falling(local: Path) -> dict:
    t, _, cy = clean_grid(local)
    q = T.vertical_q(cy)
    cut = int(len(t) * 0.6)
    scores = {}
    for name, terms in (("baseline", list(P.BASELINE_TERMS)),
                        ("gravity", ["1"]),
                        ("drag", ["1", "abs(v)*v"])):
        sc = P.score_candidate(terms, t[:cut], q[:cut], t[cut:], q[cut:])
        scores[name] = {"ok": bool(sc.ok), "held_nrmse": held(sc),
                        "coeffs": [float(c) for c in sc.coeffs] if sc.ok else [],
                        "error": sc.error}
    drag, grav, base = scores["drag"], scores["gravity"], scores["baseline"]
    winner = bool(drag["ok"] and grav["ok"] and base["ok"]
                  and drag["held_nrmse"] <= 0.8 * grav["held_nrmse"]
                  and drag["held_nrmse"] < base["held_nrmse"]
                  and drag["coeffs"][1] < 0)
    return {"n": len(t), "scores": scores, "drag_winner": winner}


def check_projectile(local: Path) -> dict:
    t, cx, cy = clean_grid(local)
    try:
        qv = T.vertical_q(cy)
        s, e = T.trim_free_flight(t, qv)
        t, cx, cy = t[s:e], cx[s:e], cy[s:e]
    except ValueError:
        pass
    x, y, _ = projectile_coordinates(cx, cy)
    cut = int(len(t) * 0.6)
    out = {"n": len(t)}
    for name, tx, ty in (("baseline", P.PROJECTILE_BASELINE_X,
                          P.PROJECTILE_BASELINE_Y),
                         ("gravity", ["0"], ["1"]),
                         ("drag", [P.DRAG_TERM_X], ["1", P.DRAG_TERM_Y])):
        sc = P.score_projectile(tx, ty, t[:cut], x[:cut], y[:cut],
                                t[cut:], x[cut:], y[cut:])
        out[name] = {"ok": bool(sc.ok), "held_nrmse": held(sc),
                     "error": sc.error}
    drag, grav = out["drag"], out["gravity"]
    out["drag_beats_gravity"] = bool(
        drag["ok"] and grav["ok"]
        and drag["held_nrmse"] < grav["held_nrmse"])
    return out


def check_along_x(local: Path) -> dict:
    t, cx, _ = clean_grid(local)
    q = cx - cx[0]
    cut = int(len(t) * 0.6)
    out = {"n": len(t)}
    for name, terms in (("baseline", list(P.BASELINE_TERMS)),
                        ("constant", ["1"]),
                        ("coulomb", ["1", "sign(v)"])):
        sc = P.score_candidate(terms, t[:cut], q[:cut], t[cut:], q[cut:])
        out[name] = {"ok": bool(sc.ok), "held_nrmse": held(sc),
                     "error": sc.error}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline unseen-clip check.")
    parser.add_argument("--clips-dir", default="/tmp/stress3")
    parser.add_argument("--out", default="results/unseen_clips.json")
    args = parser.parse_args(argv)
    clips_dir = Path(args.clips_dir)

    rows: dict[str, dict] = {}
    for name in FALLING:
        try:
            rows[name] = {"kind": "falling", **check_falling(clips_dir / f"{name}.mp4")}
        except (ValueError, RuntimeError) as exc:
            rows[name] = {"kind": "falling", "error": f"{type(exc).__name__}: {exc}"}
        print(name, rows[name].get("drag_winner", rows[name].get("error")), flush=True)
    for name in PROJECTILE:
        try:
            rows[name] = {"kind": "projectile", **check_projectile(clips_dir / f"{name}.mp4")}
        except (ValueError, RuntimeError) as exc:
            rows[name] = {"kind": "projectile", "error": f"{type(exc).__name__}: {exc}"}
        print(name, rows[name].get("drag_beats_gravity", rows[name].get("error")), flush=True)
    for name in ALONG_X:
        try:
            rows[name] = {"kind": "along_x", **check_along_x(clips_dir / f"{name}.mp4")}
        except (ValueError, RuntimeError) as exc:
            rows[name] = {"kind": "along_x", "error": f"{type(exc).__name__}: {exc}"}
        print(name, rows[name].get("error", "scored"), flush=True)

    spec = importlib.util.spec_from_file_location(
        "pendulum_consistency", REPO / "scripts" / "pendulum_consistency.py")
    assert spec and spec.loader
    pc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pc)
    try:
        entry = pc.analyse(clips_dir / "pend7.mp4")
        rows["pend7"] = {"kind": "pendulum", **entry}
    except (ValueError, RuntimeError) as exc:
        rows["pend7"] = {"kind": "pendulum",
                         "error": f"{type(exc).__name__}: {exc}"}
    print("pend7", rows["pend7"].get("pass_5pct", rows["pend7"].get("error")),
          flush=True)

    doc = {"note": "Offline (no Spark) scores on clips downloaded "
                   "2026-10-31 that played no part in tuning or gates. "
                   "Clips are Morpheus (MIT), kept outside the repo.",
           "clips": rows}
    out_path = REPO / args.out
    out_path.write_text(json.dumps(sanitize(doc), indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
