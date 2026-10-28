"""M11: argparse + run orchestration. Skeleton Day 1, wired Day 6 (+MVP-2 Phase A).

``pixelphys run <clip> --prompt ...`` executes the pipeline end to end
with no manual steps: track -> clean -> 1-DOF coordinate -> 60/40 split ->
baseline fit -> Spark loop -> equation graph -> ghost overlay ->
``model.json`` + ``report.html``. Every run lands in ``runs/<run_id>/``.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from . import physics as P
from . import render as R
from . import scientist as S
from . import trajectory as T


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pixelphys",
                                     description="Video-to-physical-law discovery.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Track, fit, discover, predict, render.")
    run.add_argument("clip", help="Path to input clip.")
    run.add_argument("--prompt", required=True, help="Object description for SAM.")
    run.add_argument("--tracker", default="auto", choices=("auto", "sam_video", "color"),
                     help="Sensor backend (auto tries SAM, falls back to color).")
    run.add_argument("--axis", default="y", choices=("x", "y"),
                     help="1-DOF coordinate axis (y = vertical toss, x = rest-relative).")
    run.add_argument("--trim", action="store_true",
                     help="Trim to the free-flight window (tossed ball).")
    run.add_argument("--scale", type=float, default=None,
                     help="Pixels per meter (default: pixel units).")
    run.add_argument("--effort", default="low",
                     help="Spark reasoning effort (low while tuning).")
    run.add_argument("--out", default=None,
                     help="Run directory (default: runs/<timestamp>_<clip stem>).")
    return parser


def track_observations(clip: Path, prompt: str, tracker: str):
    """SAM video primary, HSV color fallback. Returns (observations, used)."""
    from . import tracking as tr

    if tracker in ("sam_video", "auto"):
        try:
            fps = tr.clip_fps(clip)
            obs = tr.track_clip(clip, prompt, fps)
            if len(obs) < 2:
                raise ValueError(f"SAM tracked only {len(obs)} frames")
            return obs, "sam_video"
        except Exception as exc:  # noqa: BLE001 - fallback is the design
            if tracker == "sam_video":
                raise
            print(f"SAM tracking failed ({exc}); falling back to color.", flush=True)
    obs = None
    for low, high in ((tr.TENNIS_GREEN_LOW, tr.TENNIS_GREEN_HIGH),
                      (tr.SYNTHETIC_DARK_LOW, tr.SYNTHETIC_DARK_HIGH)):
        obs = tr.track_color_clip(clip, low=low, high=high)
        if len(obs) >= 2:
            break
    assert obs is not None
    if len(obs) < 2:
        raise ValueError(f"color tracker found only {len(obs)} frames")
    return obs, "color"


def to_coordinate(obs, axis: str, trim: bool):
    """Cleaned uniform grid plus 1-DOF coordinate and pixel back-mapping."""
    gaps = np.diff([o.t for o in obs])
    fps = 1.0 / float(np.median(gaps))
    obs = T.clean_observations(obs)
    obs = max(T.split_by_gaps(obs), key=len)
    t, cx, cy = T.resample_uniform(obs, fps)
    cx_s, cy_s = T.smooth_position(cx), T.smooth_position(cy)
    if axis == "y":
        q = T.vertical_q(cy_s)
        ref = float(cy_s[0])

        def to_pixels(q_pred: np.ndarray) -> np.ndarray:
            x0 = float(np.median(cx_s))
            return np.column_stack([np.full_like(q_pred, x0), ref - q_pred])
    else:
        q = T.rest_relative_q(cx_s)
        ref = float(np.median(cx_s))

        def to_pixels(q_pred: np.ndarray) -> np.ndarray:
            y0 = float(np.median(cy_s))
            return np.column_stack([q_pred + ref, np.full_like(q_pred, y0)])
    if trim:
        start, end = T.trim_free_flight(t, q)
        t, q = t[start:end], q[start:end]
    return t, q, to_pixels, fps


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    clip = Path(args.clip)
    if not clip.exists():
        parser.error(f"clip not found: {clip}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path("runs") / f"{stamp}_{clip.stem}"
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    config = {"clip": str(clip), "prompt": args.prompt, "tracker": args.tracker,
              "axis": args.axis, "trim": args.trim, "scale": args.scale,
              "effort": args.effort}
    (out_dir / "config.json").write_text(json.dumps(config, indent=1))

    obs, used = track_observations(clip, args.prompt, args.tracker)
    T.save_observations(obs, out_dir / "observations.json")
    t, q, to_pixels, fps = to_coordinate(obs, args.axis, args.trim)
    cut = int(len(t) * 0.6)
    t_fit, q_fit, t_held, q_held = t[:cut], q[:cut], t[cut:], q[cut:]
    units = "px" if args.scale is None else "m"
    scale_note = "pixels" if args.scale is None else f"{args.scale} px/m"

    tools = S.ScientistTools(t_fit, q_fit, t_held, q_held, out_dir)
    result = S.run_loop(tools, S.default_chat_fn(args.effort))
    base, final_entry = tools.baseline, tools.models[result.final_model_id].scored
    (out_dir / "transcript.json").write_text(json.dumps(
        {"result": {"terms": result.terms, "held_nrmse": result.held_nrmse,
                    "bic": result.bic, "rounds": result.rounds,
                    "cost_usd": result.cost_usd, "submitted": result.submitted,
                    "explanation": result.explanation},
         "transcript": result.transcript}, indent=1))

    grid = np.concatenate([t_fit, t_held])
    fin_q, _ = P.simulate(list(final_entry.terms), np.asarray(final_entry.coeffs),
                          grid, (float(final_entry.q0), float(final_entry.v0)))
    eq_path = R.plot_equation(
        t_fit, q_fit, t_held, q_held,
        base.terms, base.coeffs, base.q0, base.v0, base.held_nrmse,
        final_entry.terms, final_entry.coeffs, final_entry.q0, final_entry.v0,
        final_entry.held_nrmse, out_dir / "plots" / "equation.png",
        units=units, title=f"{clip.name} ({used}, {scale_note})")
    overlay_path = R.render_overlay(clip, grid, to_pixels(fin_q),
                                    out_dir / "overlay.mp4", fps=fps)
    model = {
        "clip": str(clip), "prompt": args.prompt, "tracker": used,
        "axis": args.axis, "trim": args.trim, "units": units,
        "px_per_m": args.scale,
        "baseline": {"terms": list(base.terms),
                     "coeffs": [float(c) for c in base.coeffs],
                     "q0": base.q0, "v0": base.v0,
                     "held_nrmse": base.held_nrmse, "bic": base.bic},
        "final": {"terms": list(final_entry.terms),
                  "coeffs": [float(c) for c in final_entry.coeffs],
                  "q0": final_entry.q0, "v0": final_entry.v0,
                  "held_nrmse": final_entry.held_nrmse, "bic": final_entry.bic,
                  "train_rmse": final_entry.train_rmse},
        "rounds": result.rounds, "cost_usd": result.cost_usd,
        "submitted": result.submitted, "stopped": result.stopped_reason,
        "explanation": result.explanation,
    }
    (out_dir / "model.json").write_text(json.dumps(model, indent=1))
    summary_lines = [
        f"baseline {list(base.terms)} held {base.held_nrmse:.4f} bic {base.bic:.1f}",
    ]
    for model_id, entry in tools.models.items():
        s = entry.scored
        if model_id != tools.baseline_id and s.ok:
            summary_lines.append(
                f"{model_id} {list(s.terms)} held {s.held_nrmse:.4f} bic {s.bic:.1f}"
                f"{' SUBMITTED' if model_id == result.final_model_id and result.submitted else ''}")
    summary_lines += ["", result.explanation, "", f"stopped: {result.stopped_reason}"]
    R.build_report(
        out_dir / "report.html", f"pixelphys run: {clip.name}",
        R.render_equation(final_entry.terms, final_entry.coeffs), units,
        [{"model": "baseline (A)", "terms": list(base.terms),
          "held_nrmse": base.held_nrmse, "bic": base.bic,
          "train_rmse": base.train_rmse, "rounds": 0, "submitted": True},
         {"model": "final (B)", "terms": list(final_entry.terms),
          "held_nrmse": final_entry.held_nrmse, "bic": final_entry.bic,
          "train_rmse": final_entry.train_rmse, "rounds": result.rounds,
          "submitted": result.submitted}],
        [eq_path], transcript="\n".join(summary_lines),
        cost_usd=result.cost_usd)
    print(f"equation: {R.render_equation(final_entry.terms, final_entry.coeffs)} [{units}]")
    print(f"held NRMSE: baseline {base.held_nrmse:.4f} -> final {final_entry.held_nrmse:.4f}")
    print(f"submitted={result.submitted} rounds={result.rounds} "
          f"cost=${result.cost_usd:.4f} tracker={used}")
    print(f"wrote {out_dir} (model.json, report.html, overlay.mp4)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
