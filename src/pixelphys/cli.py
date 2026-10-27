"""M11: argparse + run orchestration. Skeleton Day 1, polish Day 6."""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pixelphys", description="Video-to-physical-law discovery.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Track, fit, discover, predict, render.")
    run.add_argument("clip", help="Path to input clip.")
    run.add_argument("--prompt", required=True, help="Object description for SAM.")
    run.add_argument("--scale", type=float, default=None, help="Pixels per meter.")
    run.add_argument("--pivot", default=None, help="Pivot as X,Y.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.print_usage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
