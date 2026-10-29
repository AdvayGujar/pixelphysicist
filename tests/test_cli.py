"""M13: CLI coordinate/scale tests (all offline, no API key)."""

from pathlib import Path

import numpy as np
import pytest

from pixelphys import physics as P
from pixelphys import render as R
from pixelphys import tracking as tr
from pixelphys.cli import to_coordinate

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def tracked_synthetic_obs():
    obs = tr.track_color_clip(FIXTURES / "toss_drag.mp4",
                              low=tr.SYNTHETIC_DARK_LOW,
                              high=tr.SYNTHETIC_DARK_HIGH)
    assert len(obs) >= 20
    return obs


def test_scale_converts_pixels_to_metres() -> None:
    """Finding 3: --scale 250 on the synthetic toss must recover SI physics.

    A lone ["1"] cannot match gravity on drag-affected data (drag steepens
    the mean deceleration), so the check fits the true drag family and
    asserts g within 5% and k within 25% of the known SI values.
    """
    import json

    from pixelphys import synthetic as syn

    truth = json.loads((FIXTURES / "toss_drag.json").read_text())
    assert truth["params"]["gravity"] == 9.81
    assert truth["params"]["drag"] == 0.35
    obs = tracked_synthetic_obs()
    t, q, _, _ = to_coordinate(obs, axis="y", trim=True, scale=syn.PX_PER_M)
    assert abs(q.max() - q.min() - 1.18) < 0.1  # metre-scale amplitude
    cut = int(len(t) * 0.6)
    scored = P.score_candidate(["1", "abs(v)*v"], t[:cut], q[:cut],
                               t[cut:], q[cut:])
    assert scored.ok
    assert abs(-scored.coeffs[0] - 9.81) / 9.81 < 0.05
    assert abs(-scored.coeffs[1] - 0.35) / 0.35 < 0.25


def test_track_plausibility_rejects_static_background() -> None:
    """Morpheus failure mode: a static-contour track must not pass."""
    from pixelphys.cli import _track_plausible
    from pixelphys.tracking import Observation

    def obs(i: int, cx: float, cy: float) -> Observation:
        return Observation(i, i / 30.0, cx, cy, (0.0, 0.0, 3.0, 3.0), 10.0, 1.0)

    static = [obs(i, 900.0 + (i % 2), 800.0) for i in range(40)]
    assert not _track_plausible(static)
    moving = [obs(i, 100.0 + 18.0 * i, 500.0 - 5.0 * i) for i in range(40)]
    assert _track_plausible(moving)


def test_motion_tracker_recovers_synthetic() -> None:
    """Frame-difference fallback: same recovery as the color tracker."""
    import json
    import statistics

    from pixelphys.synthetic import FRAME_W, GROUND_Y_PX, PX_PER_M
    from pixelphys.tracking import track_motion_clip

    truth = json.loads((FIXTURES / "toss_drag.json").read_text())
    obs = track_motion_clip(FIXTURES / "toss_drag.mp4")
    expected = [(FRAME_W // 2, GROUND_Y_PX - q * PX_PER_M) for q in truth["q"]]
    assert len(obs) == len(expected)
    errs = [abs(o.cx - ex) + abs(o.cy - ey)
            for o, (ex, ey) in zip(obs, expected)]
    assert statistics.mean(errs) <= 2.0


def test_projectile_coordinates_gravity_points_down_in_y_up() -> None:
    """Morpheus fix: converted coordinates make gravity negative on y."""
    from pixelphys import trajectory as T
    from pixelphys.cli import projectile_coordinates

    obs = T.load_observations(FIXTURES / "obs_projectile_2d.json")
    obs = T.clean_observations(obs)
    obs = max(T.split_by_gaps(obs), key=len)
    gaps = np.diff([o.t for o in obs])
    fps = 1.0 / float(np.median(gaps))
    t, cx, cy = T.resample_uniform(obs, fps)
    x, y, factor = projectile_coordinates(T.smooth_position(cx),
                                          T.smooth_position(cy))
    assert factor == 1.0
    assert y.max() > 0  # arc rises in y-up coordinates
    cut = int(len(t) * 0.6)
    gravity = P.score_projectile(["0"], ["1"], t[:cut], x[:cut], y[:cut],
                                 t[cut:], x[cut:], y[cut:])
    assert gravity.ok and gravity.coeffs_y[0] < 0  # gravity points down
    assert gravity.coeffs_x[0] == pytest.approx(0.0, abs=1e-9)
    # The drag family in y-up coordinates recovers the signed truth:
    # g = 9.81 * 30 px/m downward, k = 0.06 / 30 per px.
    drag = P.score_projectile([P.DRAG_TERM_X], ["1", P.DRAG_TERM_Y],
                              t[:cut], x[:cut], y[:cut],
                              t[cut:], x[cut:], y[cut:])
    assert drag.ok
    assert abs(drag.coeffs_y[0] + 294.3) / 294.3 < 0.05
    assert abs(-drag.coeffs_x[0] - 0.002) / 0.002 < 0.25


def test_replay_pixels_ball_stays_in_frame(tmp_path) -> None:
    """Pixel replay: the ball is drawn inside the frame, not off-canvas."""
    import cv2

    out = R.render_replay_pixels(
        ["0"], np.array([0.0]), ["1"], np.array([-300.0]),
        (0.0, 0.0, 120.0, 160.0), np.linspace(0.0, 0.5, 16),
        tmp_path / "replay.mp4", origin_px=(60.0, 300.0), factor=1.0,
        frame_size=(640, 480))
    cap = cv2.VideoCapture(str(out))
    try:
        ok, frame = cap.read()
        assert ok
        dark = (frame[:, :, 0] < 60) & (frame[:, :, 1] < 60)
        assert dark.any(), "replay ball missing from the frame"
    finally:
        cap.release()


def test_pixel_units_without_scale() -> None:
    """Without --scale the coordinate stays in pixels (never mislabelled)."""
    obs = tracked_synthetic_obs()
    t, q, _, _ = to_coordinate(obs, axis="y", trim=True, scale=None)
    assert abs(q.max() - q.min()) > 100.0  # pixel-scale amplitude


def test_overlay_blanks_ghost_outside_span(tmp_path) -> None:
    """Finding 13: no frozen ghost before the fitted window starts."""
    import cv2

    truth = __import__("json").loads((FIXTURES / "toss_drag.json").read_text())
    t = np.array(truth["t"])
    xy = np.column_stack([np.full_like(t, 320.0), 440.0 - t * 0.0 - 240.0])
    out = R.render_overlay(FIXTURES / "toss_drag.mp4", t, xy,
                           tmp_path / "overlay.mp4", t_span=(0.4, 0.6))
    assert out.exists()
    cap = cv2.VideoCapture(str(out))
    try:
        ok0, frame0 = cap.read()
        assert ok0
        red0 = (frame0[:, :, 2] > 200) & (frame0[:, :, 1] < 100) & (frame0[:, :, 0] < 100)
        assert not red0.any(), "ghost drawn before the fitted window"
        mid = int(0.5 * 30)
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok1, frame1 = cap.read()
        assert ok1
        red1 = (frame1[:, :, 2] > 200) & (frame1[:, :, 1] < 100) & (frame1[:, :, 0] < 100)
        assert red1.any(), "ghost missing inside the fitted window"
    finally:
        cap.release()
