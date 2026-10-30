"""Task A runner: mine Morpheus clips for drag-beats-gravity (offline).

For each candidate clip (motion tracker, 1-DOF vertical coordinate,
60/40 split), scores baseline vs gravity-only vs drag with
physics.score_candidate and applies the pre-registered winner rule:
drag held-NRMSE <= 0.8 * gravity AND < baseline AND drag coeff < 0 AND
all three fits ok. Numbers go to results/dataset_mining.json
(non-finite values are written as null, never Infinity).

Clips live OUTSIDE the repo (third-party, MIT): pass a directory
holding the downloads (default /tmp/stress2).

Usage:
    .venv/bin/python scripts/mine_dataset.py [--clips-dir /tmp/stress2]
        [--out results/dataset_mining.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixelphys import physics as P  # noqa: E402
from pixelphys import trajectory as T  # noqa: E402
from pixelphys.tracking import track_motion_clip  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

CLIPS = {
    "tape0": "falling_tape/video_0_fps30",
    "tape1": "falling_tape/video_1_fps30",
    "tape2": "falling_tape/video_2_fps30",
    "marker0": "falling_marker/video_0_fps30",
    "marker1": "falling_marker/video_1_fps30",
    "marker2": "falling_marker/video_2_fps30",
    "ball0": "falling_ball/video_0_fps30",
    "ball5": "falling_ball/video_5_fps30",
    "ball8": "falling_ball/video_8_fps30",
    "proj5": "projectile/video_5_fps30_cropped",
    "proj6": "projectile/video_6_fps30_cropped",
    "proj7": "projectile/video_7_fps30_cropped",
}

MODELS = {
    "baseline": list(P.BASELINE_TERMS),
    "gravity": ["1"],
    "drag": ["1", "abs(v)*v"],
}


def sanitize(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    return value


def load_q(clip: Path) -> tuple[np.ndarray, np.ndarray]:
    obs = track_motion_clip(clip)
    obs = T.clean_observations(obs)
    obs = max(T.split_by_gaps(obs), key=len)
    gaps = np.diff([o.t for o in obs])
    fps = 1.0 / float(np.median(gaps))
    t, cx, cy = T.resample_uniform(obs, fps)
    return t, T.vertical_q(T.smooth_position(cy))


def score_row(scored) -> dict:
    return {
        "ok": bool(scored.ok),
        "held_nrmse": float(scored.held_nrmse),
        "bic": float(scored.bic),
        "train_rmse": float(scored.train_rmse),
        "coeffs": [float(c) for c in scored.coeffs] if scored.ok else [],
        "n_scored": int(scored.n_scored),
        "error": scored.error,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mine Morpheus for drag winners.")
    parser.add_argument("--clips-dir", default="/tmp/stress2")
    parser.add_argument("--out", default="results/dataset_mining.json")
    args = parser.parse_args(argv)
    clips_dir = Path(args.clips_dir)

    out_clips: dict[str, dict] = {}
    for name, experiment in CLIPS.items():
        local = clips_dir / f"{name}.mp4"
        entry: dict = {"experiment": experiment, "local": str(local)}
        try:
            t, q = load_q(local)
            cut = int(len(t) * 0.6)
            scores = {}
            for model_name, terms in MODELS.items():
                scored = P.score_candidate(terms, t[:cut], q[:cut],
                                           t[cut:], q[cut:])
                scores[model_name] = score_row(scored)
            drag = scores["drag"]
            winner = bool(
                all(s["ok"] for s in scores.values())
                and drag["held_nrmse"] <= 0.8 * scores["gravity"]["held_nrmse"]
                and drag["held_nrmse"] < scores["baseline"]["held_nrmse"]
                and drag["coeffs"] and drag["coeffs"][1] < 0)
            entry.update({"n": len(t), "n_fit": cut,
                          "q_span_px": float(q.max() - q.min()),
                          "scores": scores, "winner": winner})
            print(f"{name}: winner={winner} "
                  + " ".join(f"{k}={v['held_nrmse']:.4f}" if v["ok"] else f"{k}=FAIL"
                             for k, v in scores.items()), flush=True)
        except (ValueError, RuntimeError) as exc:
            entry.update({"error": f"{type(exc).__name__}: {exc}",
                          "winner": False})
            print(f"{name}: ERROR {exc}", flush=True)
        out_clips[name] = entry

    doc = {
        "criterion": "drag held-NRMSE <=0.8*gravity AND drag<baseline, "
                     "with drag coeff<0 (pre-registered, unchanged)",
        "models": MODELS,
        "split": "60/40 standard, horizon min(2s,tail)",
        "coordinate": "1-DOF vertical_q (up-positive px), no trim/scale, "
                      "motion tracker",
        "clips": out_clips,
    }
    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(sanitize(doc), indent=1))
    winners = [n for n, c in out_clips.items() if c.get("winner")]
    print(f"winners: {winners}; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
