"""Plot q(t) for a clip (gate G2): track or load cache, clean, smooth, split.

Usage:
    set -a; source .env; set +a
    .venv/bin/python scripts/plot_qt.py --clip data/clips/hero_toss_take1.mp4 \\
        --prompt "green tennis ball" --fps 29.997 --file-id file-... --out runs/qt.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixelphys import trajectory as traj  # noqa: E402
from pixelphys.tracking import track_clip  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot q(t) for a clip.")
    parser.add_argument("--clip", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--fps", type=float, default=None,
                        help="Frame rate; probed from the clip when omitted.")
    parser.add_argument("--file-id", default=None)
    parser.add_argument("--trim", action="store_true",
                        help="Cut to the free-flight window (launch to first impact).")
    parser.add_argument("--axis", choices=["x", "y"], default="y",
                        help="Which image axis is the 1-DOF coordinate (pendulum swings in x).")
    parser.add_argument("--scale", type=float, default=None,
                        help="Pixels per metre (M5): q is reported in metres.")
    parser.add_argument("--pivot", default=None,
                        help="Pendulum pivot as X,Y in pixels; with --axis x, "
                             "q becomes the swing angle in radians (M5).")
    parser.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = build_parser().parse_args(argv)
    clip = Path(args.clip)
    from pixelphys.tracking import clip_fps  # noqa: E402

    fps = args.fps if args.fps is not None else clip_fps(clip)
    cache = traj.cache_path(clip, args.prompt)
    if cache.exists():
        obs = traj.load_observations(cache)
        print(f"loaded {len(obs)} cached observations")
    else:
        obs = track_clip(clip, args.prompt, fps, file_id=args.file_id)
        traj.save_observations(obs, cache)
        print(f"tracked {len(obs)} observations")
    obs = traj.clean_observations(obs)
    runs = traj.split_by_gaps(obs)
    obs = max(runs, key=len)
    print(f"using longest consecutive run: {len(obs)} of {sum(len(r) for r in runs)} obs")
    t, cx, cy = traj.resample_uniform(obs, fps)
    import numpy as np

    from pixelphys import calibrate

    if args.pivot is not None:
        if args.axis != "x":
            raise SystemExit("--pivot applies to the pendulum (--axis x)")
        px, py = (float(v) for v in args.pivot.split(","))
        q = traj.pendulum_angle(traj.smooth_position(cx),
                                traj.smooth_position(cy), (px, py))
        unit = "rad"
    elif args.axis == "x":
        q = traj.rest_relative_q(traj.smooth_position(cx))
        unit = "px"
    else:
        q = traj.vertical_q(traj.smooth_position(cy))
        unit = "px"
    if args.scale is not None and unit == "px":
        q = np.array(calibrate.to_meters(list(q), args.scale))
        unit = "m"
    if args.trim:
        start, end = traj.trim_free_flight(t, q)
        t, q = t[start:end], q[start:end]
        print(f"trimmed to free flight: {len(t)} points, {t[0]:.2f}..{t[-1]:.2f} s")
    t_fit, q_fit, t_held, q_held = traj.split_fit_held(t, q)
    print(traj.trajectory_summary(t, q))

    _, ax = plt.subplots()
    ax.plot(t, q, label="q(t) smoothed")
    ax.axvline(t_held[0], linestyle="--", label="fit/held-out split")
    ax.plot(t_held, q_held, linewidth=2.5, label="held-out")
    ax.set_xlabel("t (s)")
    ax.set_ylabel(f"q ({unit})")
    ax.legend()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=100)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
