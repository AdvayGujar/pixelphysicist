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
    """Pair box+mask records by (frame, object) into Observations.

    Two passes: masks are kept even when their box record arrives later.
    When several objects appear, only the most present one (most frames,
    then largest mean area) becomes the trajectory.
    """
    from meta_sam_parser import decode_mask_to_raster

    boxes: dict[tuple[int, str], Any] = {}
    masks: list[Any] = []
    for record in records:
        kind = type(record).__name__
        if kind == "SegmentationBoxRecord":
            boxes[(record.frame.frame_index, record.object_id)] = record
        elif kind == "SegmentationMaskRecord":
            masks.append(record)

    decoded: list[tuple[Any, Any, tuple[float, float, float]]] = []
    for mask_record in masks:
        key = (mask_record.frame.frame_index, mask_record.object_id)
        box = boxes.get(key)
        if box is None:
            continue
        raster = decode_mask_to_raster(mask_record.mask)
        center = raster_centroid(raster, mask_record.mask.width, mask_record.mask.height)
        if center is None:
            continue
        decoded.append((box, mask_record, center))
    if not decoded:
        return []

    by_object: dict[str, list[int]] = {}
    for i, (_, mask_record, _) in enumerate(decoded):
        by_object.setdefault(mask_record.object_id, []).append(i)

    def object_score(object_id: str) -> tuple[int, float]:
        idxs = by_object[object_id]
        frames = len({decoded[i][1].frame.frame_index for i in idxs})
        mean_area = sum(decoded[i][2][2] for i in idxs) / len(idxs)
        return frames, mean_area

    chosen = max(by_object, key=object_score)
    observations: list[Observation] = []
    for i in by_object[chosen]:
        box, mask_record, (local_x, local_y, area) = decoded[i]
        frame_idx = mask_record.frame.frame_index
        observations.append(Observation(
            frame_idx=frame_idx,
            t=frame_idx / fps,
            cx=box.left + local_x,
            cy=box.top + local_y,
            bbox=(box.left, box.top, box.right, box.bottom),
            mask_area=area,
            confidence=mask_record.confidence if mask_record.confidence is not None else -1.0,
        ))
    observations.sort(key=lambda o: o.frame_idx)
    return observations


def clip_fps(path: Path, default: float = 30.0) -> float:
    """Frame rate probed from the file; default when unreadable."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    return fps if fps > 0 else default


def clip_frame_count(path: Path) -> int:
    """Frame count probed from the file; 0 when unreadable."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return max(count, 0)


def upload_clip(path: Path) -> str:
    """Upload a clip via Files API. Retried and cost-logged. Returns file id."""
    import time

    from openai import OpenAI

    from .config import load_settings
    from .meta import ApiCall, _with_retries, log_call

    settings = load_settings()
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    start = time.time()

    def attempt() -> Any:
        with path.open("rb") as handle:
            return client.files.create(file=(path.name, handle), purpose="user_data")

    try:
        uploaded = _with_retries("upload", attempt)
    except RuntimeError as exc:
        log_call(ApiCall("files", "upload", False, time.time() - start,
                         error=str(exc)[:200]))
        raise
    log_call(ApiCall("files", "upload", True, time.time() - start))
    return str(uploaded.id)


async def track_video(file_id: str, prompt: str, fps: float,
                      billed_frames: int = 0) -> list[Observation]:
    """One SAM video request -> parsed Observations. Retried and cost-logged."""
    import time

    from openai import AsyncOpenAI

    from meta_sam_parser import parse_responses_stream, video_segmentation_format

    from .config import SAM_MODEL_ID, load_settings
    from .meta import BACKOFF_S, MAX_ATTEMPTS, ApiCall, log_call

    async def once() -> Any:
        settings = load_settings()
        client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url)
        events = await client.responses.create(
            model=SAM_MODEL_ID,
            input=[{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_video", "file_id": file_id},
            ]}],
            stream=True,
            metadata={"mask_encoding": "one_bit", "include_confidence": "true"},
        )
        return await parse_responses_stream(events, video_segmentation_format()).final_result()

    start = time.time()
    last_error = ""
    result = None
    for attempt_no in range(1, MAX_ATTEMPTS + 1):
        try:
            result = await once()
            break
        except Exception as exc:  # noqa: BLE001 - retried, then recorded
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt_no < MAX_ATTEMPTS:
                await asyncio.sleep(BACKOFF_S * attempt_no)
    if result is None:
        log_call(ApiCall("responses", SAM_MODEL_ID, False, time.time() - start,
                         error=last_error[:200]))
        raise RuntimeError(f"responses failed after {MAX_ATTEMPTS} attempts: {last_error}")
    observed = len({r.frame.frame_index for r in result.records
                    if hasattr(r, "frame") and r.frame is not None})
    log_call(ApiCall("responses", SAM_MODEL_ID, True, time.time() - start,
                     video_frames_billed=billed_frames,
                     video_frames_observed=observed))
    return observations_from_records(result.records, fps)


def track_clip(path: Path, prompt: str, fps: float, file_id: str | None = None) -> list[Observation]:
    """Upload (unless file_id given) and track. Sync wrapper for the CLI."""
    billed = clip_frame_count(path)
    return asyncio.run(track_video(file_id or upload_clip(path), prompt, fps,
                                  billed_frames=billed))


TENNIS_GREEN_LOW = (25, 40, 40)
TENNIS_GREEN_HIGH = (85, 255, 255)
SYNTHETIC_DARK_LOW = (0, 0, 0)
SYNTHETIC_DARK_HIGH = (180, 255, 60)


def track_color_clip(path: Path, low: tuple[int, int, int] = TENNIS_GREEN_LOW,
                     high: tuple[int, int, int] = TENNIS_GREEN_HIGH,
                     fps: float | None = None) -> list[Observation]:
    """HSV fallback tracker: largest contour centroid per frame.

    Deterministic and offline. Confidence is 1.0 (no model involved).
    """
    import cv2
    import numpy as np

    fps = fps if fps is not None else clip_fps(path)
    capture = cv2.VideoCapture(str(path))
    observations: list[Observation] = []
    try:
        idx = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
                               np.array(low), np.array(high))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                biggest = max(contours, key=cv2.contourArea)
                moments = cv2.moments(biggest)
                if moments["m00"] > 0:
                    x, y, w, h = cv2.boundingRect(biggest)
                    observations.append(Observation(
                        frame_idx=idx,
                        t=idx / fps,
                        cx=moments["m10"] / moments["m00"],
                        cy=moments["m01"] / moments["m00"],
                        bbox=(float(x), float(y), float(x + w), float(y + h)),
                        mask_area=float(cv2.contourArea(biggest)),
                        confidence=1.0,
                    ))
            idx += 1
    finally:
        capture.release()
    return observations
