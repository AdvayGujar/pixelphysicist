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
    run.add_argument("--pivot", default=None,
                     help="Pivot as X,Y in pixels (pendulum angle coordinate).")
    run.add_argument("--mode", default="1d", choices=("1d", "projectile"),
                     help="1-DOF coordinate fit (default) or coupled 2-DOF projectile.")
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


def to_coordinate(obs, axis: str, trim: bool, scale: float | None = None,
                  pivot: tuple[float, float] | None = None):
    """Cleaned uniform grid plus 1-DOF coordinate and pixel back-mapping.

    With ``scale`` (px/m) the coordinate is converted to metres; without
    it the pipeline runs in pixels (Finding 3: never label pixels "m").
    With ``pivot`` the coordinate is the pendulum angle (radians).
    """
    gaps = np.diff([o.t for o in obs])
    fps = 1.0 / float(np.median(gaps))
    obs = T.clean_observations(obs)
    obs = max(T.split_by_gaps(obs), key=len)
    t, cx, cy = T.resample_uniform(obs, fps)
    cx_s, cy_s = T.smooth_position(cx), T.smooth_position(cy)
    to_m = 1.0 if scale is None else 1.0 / scale
    if pivot is not None:
        q = T.pendulum_angle(cx_s, cy_s, pivot)
        px0, py0 = float(pivot[0]), float(pivot[1])
        r = float(np.median(np.hypot(cx_s - px0, cy_s - py0)))

        def to_pixels(q_pred: np.ndarray) -> np.ndarray:
            return np.column_stack([px0 + r * np.sin(q_pred),
                                    py0 + r * np.cos(q_pred)])
    elif axis == "y":
        q = T.vertical_q(cy_s) * to_m
        ref = float(cy_s[0])

        def to_pixels(q_pred: np.ndarray) -> np.ndarray:
            x0 = float(np.median(cx_s))
            return np.column_stack([np.full_like(q_pred, x0),
                                    ref - q_pred / to_m])
    else:
        q = T.rest_relative_q(cx_s) * to_m
        ref = float(np.median(cx_s))

        def to_pixels(q_pred: np.ndarray) -> np.ndarray:
            y0 = float(np.median(cy_s))
            return np.column_stack([q_pred / to_m + ref,
                                    np.full_like(q_pred, y0)])
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
    pivot: tuple[float, float] | None = None
    if args.pivot is not None:
        try:
            px, py = (float(v) for v in str(args.pivot).split(","))
        except ValueError:
            parser.error(f"--pivot must be X,Y in pixels, got {args.pivot!r}")
        pivot = (px, py)
    if args.trim and pivot is not None:
        parser.error("--trim trims toss free-flight; do not combine it with --pivot")
    # Finding 12: pre-flight before any compute (tracking burns API calls).
    import os

    from .config import load_dotenv

    load_dotenv()
    if not os.environ.get("MODEL_API_KEY"):
        parser.error("MODEL_API_KEY is not set (copy .env.example to .env); "
                     "the Spark scientist loop needs it")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path("runs") / f"{stamp}_{clip.stem}"
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    config = {"clip": str(clip), "prompt": args.prompt, "tracker": args.tracker,
              "mode": args.mode,
              "axis": args.axis, "trim": args.trim, "scale": args.scale,
              "pivot": args.pivot, "effort": args.effort}
    (out_dir / "config.json").write_text(json.dumps(config, indent=1))

    obs, used = track_observations(clip, args.prompt, args.tracker)
    T.save_observations(obs, out_dir / "observations.json")
    if args.mode == "projectile":
        return run_projectile(args, clip, out_dir, obs, used)
    return run_1d(args, clip, out_dir, obs, used, pivot)


def run_1d(args, clip: Path, out_dir: Path, obs, used: str,
           pivot: tuple[float, float] | None) -> int:
    t, q, to_pixels, fps = to_coordinate(obs, args.axis, args.trim,
                                         scale=args.scale, pivot=pivot)
    cut = int(len(t) * 0.6)
    t_fit, q_fit, t_held, q_held = t[:cut], q[:cut], t[cut:], q[cut:]
    units = "px" if args.scale is None else "m"
    scale_note = "pixels" if args.scale is None else f"{args.scale} px/m"

    tools = S.ScientistTools(t_fit, q_fit, t_held, q_held, out_dir)
    try:
        result = S.run_loop(tools, S.default_chat_fn(args.effort))
    except RuntimeError as exc:
        print(f"scientist loop failed (transport or API error): {exc}", flush=True)
        print(f"partial run kept in {out_dir} (config, observations).", flush=True)
        return 2
    base, final_entry = tools.baseline, tools.models[result.final_model_id].scored
    if not base.ok:
        print("baseline fit failed; cannot score candidates. "
              f"Partial run kept in {out_dir}.", flush=True)
        return 2
    if not final_entry.ok:
        print("no fitted model survived; nothing to render. "
              f"Partial run kept in {out_dir}.", flush=True)
        return 2
    # Finding 2: the gravity-only control belongs in every report, or the
    # Finding 2: the gravity-only control belongs in every report, or the
    # demo narrative presents drag as the win while gravity wins off-screen.
    gravity = P.score_candidate(["1"], t_fit, q_fit, t_held, q_held)
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
        units=units, title=f"{clip.name} ({used}, {scale_note})",
        control_terms=gravity.terms, control_coeffs=gravity.coeffs,
        control_q0=gravity.q0, control_v0=gravity.v0,
        control_held=gravity.held_nrmse, control_label="gravity-only")
    overlay_path = R.render_overlay(clip, grid, to_pixels(fin_q),
                                    out_dir / "overlay.mp4", fps=fps,
                                    t_span=(float(grid[0]), float(grid[-1])))
    model = {
        "clip": str(clip), "prompt": args.prompt, "tracker": used,
        "axis": args.axis, "trim": args.trim, "units": units,
        "px_per_m": args.scale,
        "baseline": {"terms": list(base.terms),
                     "coeffs": [float(c) for c in base.coeffs],
                     "q0": base.q0, "v0": base.v0,
                     "held_nrmse": base.held_nrmse, "bic": base.bic},
        "gravity_only": {"terms": list(gravity.terms),
                         "coeffs": [float(c) for c in gravity.coeffs],
                         "q0": gravity.q0, "v0": gravity.v0,
                         "held_nrmse": gravity.held_nrmse, "bic": gravity.bic},
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
         {"model": "gravity-only (control)", "terms": list(gravity.terms),
          "held_nrmse": gravity.held_nrmse, "bic": gravity.bic,
          "train_rmse": gravity.train_rmse, "rounds": 0, "submitted": True},
         {"model": "final (B)", "terms": list(final_entry.terms),
          "held_nrmse": final_entry.held_nrmse, "bic": final_entry.bic,
          "train_rmse": final_entry.train_rmse, "rounds": result.rounds,
          "submitted": result.submitted}],
        [eq_path], transcript="\n".join(summary_lines),
        cost_usd=result.cost_usd)
    print(f"equation: {R.render_equation(final_entry.terms, final_entry.coeffs)} [{units}]")
    print(f"held NRMSE: baseline {base.held_nrmse:.4f} / "
          f"gravity-only {gravity.held_nrmse:.4f} -> final {final_entry.held_nrmse:.4f}")
    print(f"submitted={result.submitted} rounds={result.rounds} "
          f"cost=${result.cost_usd:.4f} tracker={used}")
    print(f"wrote {out_dir} (model.json, report.html, overlay.mp4)")
    return 0


def run_projectile(args, clip: Path, out_dir: Path, obs, used: str) -> int:
    """Coupled 2-DOF path: track -> x(t),y(t) -> projectile Spark loop.

    Coordinates are raw smoothed pixels (mirrors the tracked round-trip
    test); trim uses the vertical free-flight window when requested.
    """
    t, cx, cy = _clean_grid(obs)
    xs, ys = T.smooth_position(cx), T.smooth_position(cy)
    if args.trim:
        qv = T.vertical_q(ys)
        start, end = T.trim_free_flight(t, qv)
        t, xs, ys = t[start:end], xs[start:end], ys[start:end]
    cut = int(len(t) * 0.6)
    t_fit, x_fit, y_fit = t[:cut], xs[:cut], ys[:cut]
    t_held, x_held, y_held = t[cut:], xs[cut:], ys[cut:]
    units = "px"

    tools = S.ProjectileTools(t_fit, x_fit, y_fit, t_held, x_held, y_held,
                              out_dir,
                              baseline=P.score_projectile(
                                  P.PROJECTILE_BASELINE_X,
                                  P.PROJECTILE_BASELINE_Y,
                                  t_fit, x_fit, y_fit, t_held, x_held, y_held))
    try:
        result = S.run_loop(tools, S.default_chat_fn(args.effort),
                            system_prompt=S.PROJECTILE_SYSTEM_PROMPT,
                            schemas=S.TOOL_SCHEMAS_2D)
    except RuntimeError as exc:
        print(f"scientist loop failed (transport or API error): {exc}", flush=True)
        print(f"partial run kept in {out_dir} (config, observations).", flush=True)
        return 2
    base = tools.baseline
    final_entry = tools.models[result.final_model_id].scored
    if not base.ok:
        print("baseline fit failed; cannot score candidates. "
              f"Partial run kept in {out_dir}.", flush=True)
        return 2
    if not final_entry.ok:
        print("no fitted model survived; nothing to render. "
              f"Partial run kept in {out_dir}.", flush=True)
        return 2
    (out_dir / "transcript.json").write_text(json.dumps(
        {"result": {"axis_terms": result.axis_terms,
                    "held_nrmse": result.held_nrmse, "bic": result.bic,
                    "rounds": result.rounds, "cost_usd": result.cost_usd,
                    "submitted": result.submitted,
                    "explanation": result.explanation},
         "transcript": result.transcript}, indent=1))

    gravity = P.score_projectile(["1"], ["1"],
                                 t_fit, x_fit, y_fit, t_held, x_held, y_held)
    grid = np.concatenate([t_fit, t_held])

    def _rollout(scored) -> tuple[np.ndarray, np.ndarray]:
        px, py, _, _ = P.simulate_projectile(
            list(scored.terms_x), list(scored.terms_y),
            np.asarray(scored.coeffs_x), np.asarray(scored.coeffs_y),
            grid, tuple(float(v) for v in scored.state0))
        return px, py

    base_x, base_y = _rollout(base)
    grav_x, grav_y = _rollout(gravity)
    fin_x, fin_y = _rollout(final_entry)
    eq_paths = []
    for coord, fit_c, held_c in (("x", x_fit, x_held), ("y", y_fit, y_held)):
        terms = final_entry.terms_x if coord == "x" else final_entry.terms_y
        coeffs = final_entry.coeffs_x if coord == "x" else final_entry.coeffs_y
        bterms = base.terms_x if coord == "x" else base.terms_y
        bcoeffs = base.coeffs_x if coord == "x" else base.coeffs_y
        gcoeffs = (gravity.coeffs_x if coord == "x" else gravity.coeffs_y)
        gterms = ["1"]
        eq_paths.append(R.plot_equation(
            t_fit, fit_c, t_held, held_c,
            bterms, bcoeffs, 0.0, 0.0, base.held_nrmse,
            terms, coeffs, 0.0, 0.0, final_entry.held_nrmse,
            out_dir / "plots" / f"equation_{coord}.png",
            units=units, title=f"{clip.name} {coord}(t) ({used})",
            lhs=f"{coord}''",
            control_terms=gterms, control_coeffs=gcoeffs,
            control_q0=0.0, control_v0=0.0,
            control_held=gravity.held_nrmse, control_label="gravity-only",
            pred_base=base_x if coord == "x" else base_y,
            pred_final=fin_x if coord == "x" else fin_y,
            pred_control=grav_x if coord == "x" else grav_y))
    fpx, fpy, _, _ = P.simulate_projectile(
        list(final_entry.terms_x), list(final_entry.terms_y),
        np.asarray(final_entry.coeffs_x), np.asarray(final_entry.coeffs_y),
        grid, tuple(float(v) for v in final_entry.state0))
    overlay_path = R.render_overlay(
        clip, grid, np.column_stack([fpx, fpy]),
        out_dir / "overlay.mp4",
        fps=float(1.0 / np.median(np.diff(grid))) if len(grid) > 1 else 30.0,
        t_span=(float(grid[0]), float(grid[-1])))
    replay_path = R.render_replay_2d(
        list(final_entry.terms_x), np.asarray(final_entry.coeffs_x),
        list(final_entry.terms_y), np.asarray(final_entry.coeffs_y),
        tuple(float(v) for v in final_entry.state0), grid,
        out_dir / "replay.mp4",
        caption=f"x''={R.render_equation(final_entry.terms_x, final_entry.coeffs_x, lhs='x')[:60]}")
    model = {
        "clip": str(clip), "prompt": args.prompt, "tracker": used,
        "mode": "projectile", "trim": args.trim, "units": units,
        "baseline": {"terms_x": list(base.terms_x),
                     "terms_y": list(base.terms_y),
                     "coeffs_x": [float(c) for c in base.coeffs_x],
                     "coeffs_y": [float(c) for c in base.coeffs_y],
                     "held_nrmse": base.held_nrmse, "bic": base.bic},
        "gravity_only": {"held_nrmse": gravity.held_nrmse, "bic": gravity.bic,
                         "coeffs_x": [float(c) for c in gravity.coeffs_x],
                         "coeffs_y": [float(c) for c in gravity.coeffs_y]},
        "final": {"terms_x": list(final_entry.terms_x),
                  "terms_y": list(final_entry.terms_y),
                  "coeffs_x": [float(c) for c in final_entry.coeffs_x],
                  "coeffs_y": [float(c) for c in final_entry.coeffs_y],
                  "state0": [float(v) for v in final_entry.state0],
                  "held_nrmse": final_entry.held_nrmse, "bic": final_entry.bic,
                  "train_rmse": final_entry.train_rmse},
        "rounds": result.rounds, "cost_usd": result.cost_usd,
        "submitted": result.submitted, "stopped": result.stopped_reason,
        "explanation": result.explanation,
    }
    (out_dir / "model.json").write_text(json.dumps(model, indent=1))
    R.build_report(
        out_dir / "report.html", f"pixelphys run (projectile): {clip.name}",
        R.render_equation(final_entry.terms_x, final_entry.coeffs_x, lhs="x")
        + " ; "
        + R.render_equation(final_entry.terms_y, final_entry.coeffs_y, lhs="y"),
        units,
        [{"model": "baseline (A)",
          "terms": list(base.terms_x) + list(base.terms_y),
          "held_nrmse": base.held_nrmse, "bic": base.bic,
          "train_rmse": base.train_rmse, "rounds": 0, "submitted": True},
         {"model": "gravity-only (control)", "terms": ["1", "1"],
          "held_nrmse": gravity.held_nrmse, "bic": gravity.bic,
          "train_rmse": gravity.train_rmse, "rounds": 0, "submitted": True},
         {"model": "final (B)",
          "terms": list(final_entry.terms_x) + list(final_entry.terms_y),
          "held_nrmse": final_entry.held_nrmse, "bic": final_entry.bic,
          "train_rmse": final_entry.train_rmse, "rounds": result.rounds,
          "submitted": result.submitted}],
        eq_paths, transcript=result.explanation,
        cost_usd=result.cost_usd)
    print(f"equation x: {R.render_equation(final_entry.terms_x, final_entry.coeffs_x, lhs='x')} [{units}]")
    print(f"equation y: {R.render_equation(final_entry.terms_y, final_entry.coeffs_y, lhs='y')} [{units}]")
    print(f"held NRMSE: baseline {base.held_nrmse:.4f} / "
          f"gravity-only {gravity.held_nrmse:.4f} -> final {final_entry.held_nrmse:.4f}")
    print(f"submitted={result.submitted} rounds={result.rounds} "
          f"cost=${result.cost_usd:.4f} tracker={used}")
    print(f"wrote {out_dir} (model.json, report.html, overlay.mp4, replay.mp4)")
    return 0


def _clean_grid(obs):
    gaps = np.diff([o.t for o in obs])
    fps = 1.0 / float(np.median(gaps))
    obs = T.clean_observations(obs)
    obs = max(T.split_by_gaps(obs), key=len)
    t, cx, cy = T.resample_uniform(obs, fps)
    return t, cx, cy


if __name__ == "__main__":
    raise SystemExit(main())
