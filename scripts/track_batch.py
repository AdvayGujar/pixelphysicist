"""Track every clip in data/clips, caching Observations to runs/.

Usage:
    set -a; source .env
    .venv/bin/python scripts/track_batch.py --prompt "green tennis ball" [--glob "hero*"]
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixelphys import trajectory as traj  # noqa: E402
from pixelphys.tracking import clip_fps, track_clip  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-track clips.")
    parser.add_argument("--glob", default="*.mp4", help="Filename pattern under data/clips.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--force", action="store_true", help="Re-track even if cached.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    clips = sorted(glob.glob(str(Path("data/clips") / args.glob)))
    if not clips:
        print("no clips matched")
        return 1
    for clip in clips:
        stem = Path(clip).stem
        cache = traj.cache_path(Path(clip), args.prompt)
        if cache.exists() and not args.force:
            print(f"{stem}: cached ({len(traj.load_observations(cache))} obs)")
            continue
        fps = clip_fps(Path(clip))
        obs = track_clip(Path(clip), args.prompt, fps)
        traj.save_observations(obs, cache)
        span = f"{obs[0].frame_idx}->{obs[-1].frame_idx}" if obs else "empty"
        print(f"{stem}: tracked {len(obs)} obs, frames {span}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
