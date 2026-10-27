"""SAM 3.1 image smoke test (plan.md Day-1 gate G1, step 1). [PROBE — G1 record]

Sends one Responses request with a noun-phrase prompt + one image URL,
parses the output_text lane with meta-sam-parser, decodes one mask raster.
Characterizes: latency, record counts, raster sizes.

Usage:
    set -a; source .env
    .venv/bin/python scripts/smoke_sam.py --image-url https://example.com/street.png --prompt "yellow school bus"

Local-file and video wiring (input_video / Files API) lands after this
URL-based probe succeeds, so no budget is spent on guessed schemas.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixelphys.config import SAM_MODEL_ID, load_settings  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAM 3.1 image smoke test.")
    parser.add_argument("--image-url", required=True)
    parser.add_argument("--prompt", required=True)
    return parser


async def run(image_url: str, prompt: str) -> int:
    from openai import AsyncOpenAI

    settings = load_settings()
    client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url)
    events = await client.responses.create(
        model=SAM_MODEL_ID,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
        stream=True,
        metadata={"mask_encoding": "one_bit"},
    )
    try:
        from meta_sam_parser import (
            decode_mask_to_raster,
            image_segmentation_format,
            parse_responses_stream,
        )
    except ImportError:
        print("meta-sam-parser not installed; dumping raw output_text deltas:")
        async for event in events:
            if getattr(event, "type", "") == "response.output_text.delta":
                print("DELTA:", str(getattr(event, "delta", ""))[:200])
        return 0

    start = time.time()
    result = await parse_responses_stream(events, image_segmentation_format()).final_result()
    print(f"latency_s: {time.time() - start:.1f}")
    print(f"records: {len(result.records)}")
    for record in result.records:
        print("record:", type(record).__name__, getattr(record, "object_id", "?"))
        mask = getattr(record, "mask", None)
        if mask is not None:
            raster = decode_mask_to_raster(mask)
            print(f"  raster bytes: {len(raster)}, ones: {sum(raster)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(run(args.image_url, args.prompt))


if __name__ == "__main__":
    raise SystemExit(main())
