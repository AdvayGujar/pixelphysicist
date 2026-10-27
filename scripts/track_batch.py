"""Track every clip matching a glob, caching Observations to runs/.

Usage:
    set -a; source .env; set +a
    .venv/bin/python scripts/track_batch.py --glob "hero_toss_take[12345]" --prompt "green tennis ball"
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixelphys import trajectory as traj  # noqa: E402
from pixelphys.tracking import track_clip  # noqa: E402

FPS = 30.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-track clips.")
    parser.add_argument("--glob", required=True, help="Stem pattern under data/clips.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--force", action="store_true", help="Re-track even if cached.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    clips = sorted(glob.glob(str(Path("data/clips") / (args.glob + ".mp4"))))
    if not clips:
        print("no clips matched")
        return 1
    for clip in clips:
        stem = Path(clip).stem
        cache = Path("runs") / f"obs_{stem}.json"
        if cache.exists() and not args.force:
            print(f"{stem}: cached ({len(traj.load_observations(cache))} obs)")
            continue
        obs = track_clip(Path(clip), args.prompt, FPS)
        traj.save_observations(obs, cache)
        span = f"{obs[0].frame_idx}->{obs[-1].frame_idx}" if obs else "empty"
        print(f"{stem}: tracked {len(obs)} obs, frames {span}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
