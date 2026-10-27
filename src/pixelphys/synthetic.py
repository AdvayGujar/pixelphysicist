"""M4: render a known ODE + ground truth. (First clip Day 1 evening.)

Day-1 scope: vertical toss with quadratic drag, integrated with fixed-step
RK4 and rendered as a dark dot on a plain background. The Day-3 fitter (M8)
will own the shared simulator; this inline integrator exists so the
synthetic clip and its truth JSON are available before M8 lands.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

FRAME_W = 640
FRAME_H = 480
FPS = 30
DOT_RADIUS_PX = 12
GROUND_Y_PX = 440
PX_PER_M = 400
BACKGROUND = (255, 255, 255)
DOT_COLOR = (20, 20, 20)


@dataclass(frozen=True)
class TossParams:
    v0: float = 4.0
    gravity: float = 9.81
    drag: float = 0.35
    duration_s: float = 3.0


def toss_acceleration(q: float, v: float, params: TossParams) -> float:
    return -params.gravity - params.drag * v * abs(v)


def integrate_toss(params: TossParams, dt: float = 1.0 / 240.0) -> tuple[list[float], list[float], list[float]]:
    times, positions, velocities = [0.0], [0.0], [params.v0]
    while times[-1] < params.duration_s:
        q, v = positions[-1], velocities[-1]
        a1 = toss_acceleration(q, v, params)
        a2 = toss_acceleration(q + 0.5 * dt * v, v + 0.5 * dt * a1, params)
        a3 = toss_acceleration(q + 0.5 * dt * (v + 0.5 * dt * a1), v + 0.5 * dt * a2, params)
        a4 = toss_acceleration(q + dt * (v + 0.5 * dt * a2), v + dt * a3, params)
        new_q = q + dt * (v + dt * (a1 + a2 + a3) / 6.0)
        new_v = v + dt * (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0
        times.append(times[-1] + dt)
        positions.append(new_q)
        velocities.append(new_v)
    return times, positions, velocities


def sample_at_fps(times: list[float], values: list[float], fps: int = FPS) -> list[float]:
    step = times[1] - times[0]
    stride = max(1, round((1.0 / fps) / step))
    return values[::stride]


def ball_center_px(q: float) -> tuple[int, int]:
    return (FRAME_W // 2, int(GROUND_Y_PX - q * PX_PER_M))


def render_clip(positions: list[float], mp4_path: Path, fps: int = FPS) -> None:
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (FRAME_W, FRAME_H))
    try:
        for q in positions:
            frame = np.full((FRAME_H, FRAME_W, 3), BACKGROUND, dtype=np.uint8)
            cv2.circle(frame, ball_center_px(q), DOT_RADIUS_PX, DOT_COLOR, -1)
            writer.write(frame)
    finally:
        writer.release()


def render_toss_drag(mp4_path: Path, json_path: Path, params: TossParams = TossParams()) -> dict[str, Path]:
    times, positions, velocities = integrate_toss(params)
    sampled_q = sample_at_fps(times, positions)
    sampled_v = sample_at_fps(times, velocities)
    sampled_t = sample_at_fps(times, times)
    render_clip(sampled_q, mp4_path)
    truth = {
        "model": "q'' = -g - c*v*|v|",
        "params": {"v0": params.v0, "gravity": params.gravity, "drag": params.drag},
        "fps": FPS,
        "t": sampled_t,
        "q": sampled_q,
        "v": sampled_v,
    }
    json_path.write_text(json.dumps(truth))
    return {"mp4": mp4_path, "json": json_path}
