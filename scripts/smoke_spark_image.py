"""M2 image-leg smoke: one Spark Chat Completions call with an image.

Reproduces the Day-1 probe (Spark described a photo correctly) as a
committed, re-runnable script. The call goes through meta.chat, so it is
retried and cost-logged like every other Spark call.

Usage:
    .venv/bin/python scripts/smoke_spark_image.py --image-url <url> \
        [--prompt "Describe this image in one sentence."]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixelphys import meta  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spark image smoke test.")
    parser.add_argument("--image-url", required=True,
                        help="Publicly fetchable image URL.")
    parser.add_argument("--prompt",
                        default="Describe this image in one sentence.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": args.prompt},
            {"type": "image_url", "image_url": {"url": args.image_url}},
        ],
    }]
    # Spark is a reasoning model: generous max_tokens or content is None.
    response = meta.chat(messages, max_tokens=1024)
    print("image description:",
          (response.choices[0].message.content or "").strip())
    print("cost log:", meta.default_log_path())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
