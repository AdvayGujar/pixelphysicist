"""SAM 3.1 video smoke test (plan.md Day-1 gate G1, step 2).

One Responses request with an input_video part; parses the output_text lane
with video_segmentation_format. Reports latency, frame/record counts, and
the frame indices of the first records (proves cross-frame tracking shape).

Usage:
    set -a; source .env; set +a
    .venv/bin/python scripts/smoke_sam_video.py --video-url <mp4> --prompt "<noun>"
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
    parser = argparse.ArgumentParser(description="SAM 3.1 video smoke test.")
    parser.add_argument("--video-url", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--raw-usage", action="store_true",
                        help="Skip parsing; report line count and usage (billing).")
    return parser


async def run(video_url: str, prompt: str, raw_usage: bool = False) -> int:
    from openai import AsyncOpenAI

    settings = load_settings()
    client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url)
    input_payload = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_video", "video_url": video_url},
            ],
        }
    ]
    if raw_usage:
        stream = await client.responses.create(
            model=SAM_MODEL_ID, input=input_payload, stream=True,
            metadata={"mask_encoding": "one_bit"},
        )
        lines = 0
        async for event in stream:
            etype = getattr(event, "type", "")
            if etype == "response.output_text.delta":
                lines += str(getattr(event, "delta", "")).count("\n")
            elif etype == "response.completed":
                print("usage:", getattr(event, "response", {}).get("usage")
                      if isinstance(getattr(event, "response", None), dict)
                      else getattr(getattr(event, "response", None), "usage", None))
        print(f"output lines: {lines}")
        return 0

    from meta_sam_parser import decode_mask_to_raster, video_segmentation_format, parse_responses_stream

    events = await client.responses.create(
        model=SAM_MODEL_ID,
        input=input_payload,
        stream=True,
        metadata={"mask_encoding": "one_bit"},
    )
    start = time.time()
    result = await parse_responses_stream(events, video_segmentation_format()).final_result()
    print(f"latency_s: {time.time() - start:.1f}")
    print(f"records: {len(result.records)}")
    for record in result.records[:6]:
        kind = type(record).__name__
        frame = getattr(getattr(record, "frame", None), "frame_index", "?")
        print(f"  {kind} id={record.object_id} frame={frame} c={record.confidence}")
        mask = getattr(record, "mask", None)
        if mask is not None:
            print(f"    raster bytes: {len(decode_mask_to_raster(mask))}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(run(args.video_url, args.prompt, args.raw_usage))


if __name__ == "__main__":
    raise SystemExit(main())
