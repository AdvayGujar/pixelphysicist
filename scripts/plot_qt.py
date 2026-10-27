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
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--file-id", default=None)
    parser.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = build_parser().parse_args(argv)
    clip = Path(args.clip)
    cache = Path("runs") / f"obs_{clip.stem}.json"
    if cache.exists():
        obs = traj.load_observations(cache)
        print(f"loaded {len(obs)} cached observations")
    else:
        obs = track_clip(clip, args.prompt, args.fps, file_id=args.file_id)
        traj.save_observations(obs, cache)
        print(f"tracked {len(obs)} observations")
    obs = traj.clean_observations(obs)
    t, _, cy = traj.resample_uniform(obs, args.fps)
    q = traj.vertical_q(traj.smooth_position(cy))
    t_fit, q_fit, t_held, q_held = traj.split_fit_held(t, q)
    print(traj.trajectory_summary(t, q))

    _, ax = plt.subplots()
    ax.plot(t, q, label="q(t) smoothed")
    ax.axvline(t_held[0], linestyle="--", label="fit/held-out split")
    ax.plot(t_held, q_held, linewidth=2.5, label="held-out")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("q (px, up positive)")
    ax.legend()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=100)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
