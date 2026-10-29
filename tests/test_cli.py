"""M13: CLI coordinate/scale tests (all offline, no API key)."""

from pathlib import Path

import numpy as np

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
