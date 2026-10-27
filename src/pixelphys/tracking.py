"""M3: Observation, SamVideoTracker, ColorTracker. (Smoke test Day 1, finish Day 2.)"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Observation:
    frame_idx: int
    t: float
    cx: float
    cy: float
    bbox: tuple[float, float, float, float]
    mask_area: float
    confidence: float


def raster_centroid(raster: bytes, width: int, height: int) -> tuple[float, float, float] | None:
    """Mean x/y of set pixels plus area. None when the raster is empty."""
    import numpy as np

    arr = np.frombuffer(raster, dtype=np.uint8).reshape(height, width)
    ys, xs = np.nonzero(arr)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean()), float(xs.size)


def observations_from_records(records: Any, fps: float) -> list[Observation]:
    """Pair box+mask records by (frame, object) into Observations."""
    from meta_sam_parser import decode_mask_to_raster

    boxes: dict[tuple[int, str], Any] = {}
    observations: list[Observation] = []
    for record in records:
        kind = type(record).__name__
        if kind == "SegmentationBoxRecord":
            key = (record.frame.frame_index, record.object_id)
            boxes[key] = record
        elif kind == "SegmentationMaskRecord":
            key = (record.frame.frame_index, record.object_id)
            box = boxes.get(key)
            if box is None:
                continue
            raster = decode_mask_to_raster(record.mask)
            center = raster_centroid(raster, record.mask.width, record.mask.height)
            if center is None:
                continue
            local_x, local_y, area = center
            observations.append(Observation(
                frame_idx=key[0],
                t=key[0] / fps,
                cx=box.left + local_x,
                cy=box.top + local_y,
                bbox=(box.left, box.top, box.right, box.bottom),
                mask_area=area,
                confidence=record.confidence if record.confidence is not None else -1.0,
            ))
    observations.sort(key=lambda o: o.frame_idx)
    return observations


def upload_clip(path: Path) -> str:
    """Upload a clip via Files API. Returns the file id."""
    from openai import OpenAI

    from .config import load_settings

    settings = load_settings()
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    with path.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="user_data")
    return str(uploaded.id)


async def track_video(file_id: str, prompt: str, fps: float) -> list[Observation]:
    """One SAM video request -> parsed Observations."""
    from openai import AsyncOpenAI

    from meta_sam_parser import parse_responses_stream, video_segmentation_format

    from .config import SAM_MODEL_ID, load_settings

    settings = load_settings()
    client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url)
    events = await client.responses.create(
        model=SAM_MODEL_ID,
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_video", "file_id": file_id},
        ]}],
        stream=True,
        metadata={"mask_encoding": "one_bit"},
    )
    result = await parse_responses_stream(events, video_segmentation_format()).final_result()
    return observations_from_records(result.records, fps)


def track_clip(path: Path, prompt: str, fps: float, file_id: str | None = None) -> list[Observation]:
    """Upload (unless file_id given) and track. Sync wrapper for the CLI."""
    return asyncio.run(track_video(file_id or upload_clip(path), prompt, fps))
