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
DOT_RADIUS_PX = 20
GROUND_Y_PX = 440
PX_PER_M = 250
BACKGROUND = (255, 255, 255)
DOT_COLOR = (20, 20, 20)


@dataclass(frozen=True)
class TossParams:
    v0: float = 6.0
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
        if times[-1] > 0.1 and new_q < 0.0 and new_v < 0.0:
            break  # caught at release height: mirrors the real cut-at-impact
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


PX_PER_M_2D = 30
PROJECTILE_X0_M = 1.0


@dataclass(frozen=True)
class ProjectileParams:
    v0: float = 11.0
    angle_deg: float = 52.0
    gravity: float = 9.81
    drag_k: float = 0.06
    linear_c: float = 0.0


def projectile_accel(vx: float, vy: float,
                     params: ProjectileParams) -> tuple[float, float]:
    speed = (vx * vx + vy * vy) ** 0.5
    ax = -params.drag_k * speed * vx - params.linear_c * vx
    ay = -params.gravity - params.drag_k * speed * vy - params.linear_c * vy
    return ax, ay


def integrate_projectile(params: ProjectileParams,
                         dt: float = 1.0 / 240.0,
                         ) -> tuple[list[float], list[float], list[float],
                                    list[float], list[float]]:
    """RK4 the coupled drag projectile until it returns to launch height."""
    import math

    angle = math.radians(params.angle_deg)
    times, xs, ys = [0.0], [0.0], [0.0]
    vxs, vys = [params.v0 * math.cos(angle)], [params.v0 * math.sin(angle)]

    def deriv(vx: float, vy: float) -> tuple[float, float]:
        return projectile_accel(vx, vy, params)

    while True:
        vx, vy = vxs[-1], vys[-1]
        a1 = deriv(vx, vy)
        a2 = deriv(vx + 0.5 * dt * a1[0], vy + 0.5 * dt * a1[1])
        a3 = deriv(vx + 0.5 * dt * a2[0], vy + 0.5 * dt * a2[1])
        a4 = deriv(vx + dt * a3[0], vy + dt * a3[1])
        new_vx = vx + dt * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0]) / 6.0
        new_vy = vy + dt * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1]) / 6.0
        new_x = xs[-1] + dt * (vx + new_vx) / 2.0
        new_y = ys[-1] + dt * (vy + new_vy) / 2.0
        times.append(times[-1] + dt)
        xs.append(new_x)
        ys.append(new_y)
        vxs.append(new_vx)
        vys.append(new_vy)
        if times[-1] > 0.1 and new_y < 0.0 and new_vy < 0.0:
            break
        if times[-1] > 20.0:
            break
    return times, xs, ys, vxs, vys


def projectile_center_px(x_m: float, y_m: float) -> tuple[int, int]:
    return (int((PROJECTILE_X0_M + x_m) * PX_PER_M_2D),
            int(GROUND_Y_PX - y_m * PX_PER_M_2D))


def render_projectile_2d(mp4_path: Path, json_path: Path,
                         params: ProjectileParams = ProjectileParams(),
                         fps: int = FPS) -> dict[str, Path]:
    """Render the coupled-drag projectile; truth JSON carries (x, y, vx, vy)."""
    import cv2
    import numpy as np

    times, xs, ys, vxs, vys = integrate_projectile(params)
    sampled = [sample_at_fps(times, values, fps)
               for values in (times, xs, ys, vxs, vys)]
    s_t, s_x, s_y, s_vx, s_vy = sampled
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    truth = {
        "model": "x''=-k*|v|*vx-c*vx, y''=-g-k*|v|*vy-c*vy",
        "params": {"v0": params.v0, "angle_deg": params.angle_deg,
                   "gravity": params.gravity, "drag_k": params.drag_k,
                   "linear_c": params.linear_c},
        "fps": fps, "px_per_m": PX_PER_M_2D, "x0_m": PROJECTILE_X0_M,
        "ground_y_px": GROUND_Y_PX,
        "t": s_t, "x": s_x, "y": s_y, "vx": s_vx, "vy": s_vy,
    }
    json_path.write_text(json.dumps(truth))
    writer = cv2.VideoWriter(str(mp4_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (FRAME_W, FRAME_H))
    try:
        for x_m, y_m in zip(s_x, s_y):
            frame = np.full((FRAME_H, FRAME_W, 3), BACKGROUND, dtype=np.uint8)
            cv2.circle(frame, projectile_center_px(x_m, y_m),
                       DOT_RADIUS_PX, DOT_COLOR, -1)
            writer.write(frame)
    finally:
        writer.release()
    return {"mp4": mp4_path, "json": json_path}


SPRING_CENTRE_Y_PX = FRAME_H // 2

SLOPE_START_PX = (40, GROUND_Y_PX)
SLOPE_END_PX = (FRAME_W - 40, GROUND_Y_PX - 180)


@dataclass(frozen=True)
class SpringParams:
    q0: float = 0.55
    stiffness: float = 22.0
    damping: float = 0.25
    cubic: float = 6.0
    duration_s: float = 4.0


def spring_acceleration(q: float, v: float, params: SpringParams) -> float:
    return -params.stiffness * q - params.damping * v - params.cubic * q**3


def integrate_spring(params: SpringParams, dt: float = 1.0 / 240.0) -> tuple[list[float], list[float], list[float]]:
    times, positions, velocities = [0.0], [params.q0], [0.0]
    while times[-1] < params.duration_s:
        q, v = positions[-1], velocities[-1]
        a1 = spring_acceleration(q, v, params)
        a2 = spring_acceleration(q + 0.5 * dt * v, v + 0.5 * dt * a1, params)
        a3 = spring_acceleration(q + 0.5 * dt * (v + 0.5 * dt * a1), v + 0.5 * dt * a2, params)
        a4 = spring_acceleration(q + dt * (v + 0.5 * dt * a2), v + dt * a3, params)
        new_q = q + dt * (v + dt * (a1 + a2 + a3) / 6.0)
        new_v = v + dt * (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0
        times.append(times[-1] + dt)
        positions.append(new_q)
        velocities.append(new_v)
    return times, positions, velocities


def spring_center_px(q: float) -> tuple[int, int]:
    return (FRAME_W // 2, int(SPRING_CENTRE_Y_PX - q * PX_PER_M))


def render_spring(mp4_path: Path, json_path: Path, params: SpringParams = SpringParams(),
                  fps: int = FPS) -> dict[str, Path]:
    import cv2
    import numpy as np

    times, positions, velocities = integrate_spring(params)
    sampled_t = sample_at_fps(times, times, fps)
    sampled_q = sample_at_fps(times, positions, fps)
    sampled_v = sample_at_fps(times, velocities, fps)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    truth = {
        "model": "q''=-k*q-c*v-beta*q**3",
        "params": {"q0": params.q0, "stiffness": params.stiffness,
                   "damping": params.damping, "cubic": params.cubic,
                   "duration_s": params.duration_s},
        "fps": fps,
        "px_per_m": PX_PER_M,
        "centre_y_px": SPRING_CENTRE_Y_PX,
        "t": sampled_t,
        "q": sampled_q,
        "v": sampled_v,
    }
    json_path.write_text(json.dumps(truth))
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (FRAME_W, FRAME_H))
    try:
        for q in sampled_q:
            frame = np.full((FRAME_H, FRAME_W, 3), BACKGROUND, dtype=np.uint8)
            cv2.circle(frame, spring_center_px(q), DOT_RADIUS_PX, DOT_COLOR, -1)
            writer.write(frame)
    finally:
        writer.release()
    return {"mp4": mp4_path, "json": json_path}


@dataclass(frozen=True)
class RollingParams:
    v0: float = 3.4
    slope_accel: float = 1.6
    coulomb: float = 0.55
    viscous: float = 0.05
    duration_s: float = 6.0


def _sign(v: float) -> float:
    if v > 0.0:
        return 1.0
    if v < 0.0:
        return -1.0
    return 0.0


def rolling_acceleration(v: float, params: RollingParams) -> float:
    return -params.slope_accel - params.coulomb * _sign(v) - params.viscous * v


def integrate_rolling(params: RollingParams, dt: float = 1.0 / 240.0) -> tuple[list[float], list[float], list[float]]:
    times, positions, velocities = [0.0], [0.0], [params.v0]
    while times[-1] < params.duration_s:
        q, v = positions[-1], velocities[-1]
        a1 = rolling_acceleration(v, params)
        a2 = rolling_acceleration(v + 0.5 * dt * a1, params)
        a3 = rolling_acceleration(v + 0.5 * dt * a2, params)
        a4 = rolling_acceleration(v + dt * a3, params)
        new_q = q + dt * (v + dt * (a1 + a2 + a3) / 6.0)
        new_v = v + dt * (a1 + 2.0 * a2 + 2.0 * a3 + a4) / 6.0
        times.append(times[-1] + dt)
        positions.append(new_q)
        velocities.append(new_v)
    return times, positions, velocities


def _slope_unit_vector() -> tuple[float, float]:
    import math

    dx = SLOPE_END_PX[0] - SLOPE_START_PX[0]
    dy = SLOPE_END_PX[1] - SLOPE_START_PX[1]
    length = math.hypot(dx, dy)
    return dx / length, dy / length


def rolling_center_px(q: float) -> tuple[int, int]:
    ux, uy = _slope_unit_vector()
    return (int(SLOPE_START_PX[0] + q * PX_PER_M * ux),
            int(SLOPE_START_PX[1] + q * PX_PER_M * uy))


def render_rolling(mp4_path: Path, json_path: Path, params: RollingParams = RollingParams(),
                   fps: int = FPS) -> dict[str, Path]:
    import cv2
    import numpy as np

    times, positions, velocities = integrate_rolling(params)
    sampled_t = sample_at_fps(times, times, fps)
    sampled_q = sample_at_fps(times, positions, fps)
    sampled_v = sample_at_fps(times, velocities, fps)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    truth = {
        "model": "q''=-a-mu*sign(v)-c*v",
        "params": {"v0": params.v0, "slope_accel": params.slope_accel,
                   "coulomb": params.coulomb, "viscous": params.viscous,
                   "duration_s": params.duration_s},
        "fps": fps,
        "px_per_m": PX_PER_M,
        "t": sampled_t,
        "q": sampled_q,
        "v": sampled_v,
    }
    json_path.write_text(json.dumps(truth))
    writer = cv2.VideoWriter(str(mp4_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (FRAME_W, FRAME_H))
    try:
        for q in sampled_q:
            frame = np.full((FRAME_H, FRAME_W, 3), BACKGROUND, dtype=np.uint8)
            cv2.line(frame, SLOPE_START_PX, SLOPE_END_PX, (150, 150, 150), 3)
            cv2.circle(frame, rolling_center_px(q), DOT_RADIUS_PX, DOT_COLOR, -1)
            writer.write(frame)
    finally:
        writer.release()
    return {"mp4": mp4_path, "json": json_path}


def render_toss_drag(mp4_path: Path, json_path: Path, params: TossParams = TossParams()) -> dict[str, Path]:
    times, positions, velocities = integrate_toss(params)
    sampled_q = sample_at_fps(times, positions)
    sampled_v = sample_at_fps(times, velocities)
    sampled_t = sample_at_fps(times, times)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    truth = {
        "model": "q'' = -g - c*v*|v|",
        "params": {"v0": params.v0, "gravity": params.gravity, "drag": params.drag},
        "fps": FPS,
        "t": sampled_t,
        "q": sampled_q,
        "v": sampled_v,
    }
    json_path.write_text(json.dumps(truth))
    render_clip(sampled_q, mp4_path)
    return {"mp4": mp4_path, "json": json_path}
