"""M13: smoke test - imports and CLI skeleton work without an API key."""


def test_imports() -> None:
    import pixelphys  # noqa: F401
    from pixelphys import calibrate, cli, config, expr, meta, physics, render, scientist, synthetic, tracking, trajectory  # noqa: F401


def test_cli_help() -> None:
    from pixelphys.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["run", "clip.mp4", "--prompt", "ball"])
    assert args.clip == "clip.mp4"
    assert args.prompt == "ball"


def test_config_fails_fast_without_key(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from pixelphys import config

    monkeypatch.setenv("PIXELPHYS_NO_DOTENV", "1")
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    try:
        config.load_settings()
    except RuntimeError:
        return
    raise AssertionError("load_settings should fail fast without MODEL_API_KEY")


def test_config_defaults_to_meta_base_url(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from pixelphys import config

    monkeypatch.setenv("PIXELPHYS_NO_DOTENV", "1")
    monkeypatch.setenv("MODEL_API_KEY", "dummy")
    monkeypatch.delenv("MODEL_BASE_URL", raising=False)
    assert config.load_settings().base_url == "https://api.meta.ai/v1"


def test_observations_skip_text_records() -> None:
    from meta_sam_parser import SegmentationTextRecord

    from pixelphys.tracking import observations_from_records

    text = SegmentationTextRecord(order=0, text="hello")
    assert observations_from_records([text], fps=30.0) == []


def test_observation_cache_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from pixelphys import trajectory as traj
    from pixelphys.tracking import Observation

    obs = [Observation(0, 0.0, 1.0, 2.0, (0.0, 0.0, 3.0, 3.0), 10.0, 0.9)]
    loaded = traj.load_observations(traj.save_observations(obs, tmp_path / "obs.json"))
    assert loaded == obs
    assert isinstance(loaded[0].bbox, tuple)


def test_trim_free_flight_cuts_bounces() -> None:
    import numpy as np

    from pixelphys import trajectory as traj

    t = np.arange(0.0, 2.0, 1.0 / 30.0)
    q = np.where(t <= 1.0, 500.0 * t,
                 np.where(t <= 1.5, 500.0 - 800.0 * (t - 1.0), 100.0 + 600.0 * (t - 1.5)))
    start, end = traj.trim_free_flight(t, q)
    assert abs(t[start] - 0.05) < 0.05
    assert abs(t[end - 1] - 1.5) < 0.05
    assert end - start >= 40


def test_dotenv_loader_respects_no_override(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import os

    from pixelphys import config

    dotenv = tmp_path / ".env"
    dotenv.write_text("MODEL_API_KEY=from-file\n")
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.delenv("PIXELPHYS_NO_DOTENV", raising=False)
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.setenv("FROM_TEST", "x")
    config.load_dotenv()
    assert os.environ["MODEL_API_KEY"] == "from-file"
    assert "FROM_TEST" in os.environ


def test_resample_refuses_long_gaps() -> None:
    import numpy as np
    import pytest

    from pixelphys import trajectory as traj
    from pixelphys.tracking import Observation

    obs = [Observation(i if i < 10 else i + 7, i / 30.0, 1.0, 2.0,
                       (0.0, 0.0, 3.0, 3.0), 10.0, 1.0) for i in range(20)]
    with pytest.raises(ValueError, match="tracking gap"):
        traj.resample_uniform(obs, 30.0)


def test_split_by_gaps_keeps_longest_run() -> None:
    from pixelphys import trajectory as traj
    from pixelphys.tracking import Observation

    def obs_at(i: int) -> Observation:
        return Observation(i, i / 30.0, 1.0, 2.0, (0.0, 0.0, 3.0, 3.0), 10.0, 1.0)

    obs = [obs_at(i) for i in list(range(10)) + list(range(20, 60))]
    runs = traj.split_by_gaps(obs)
    assert [len(r) for r in runs] == [10, 40]


def test_clean_keeps_near_constant_areas() -> None:
    """The committed synthetic track (mask areas constant to <1%) must
    survive cleaning at full length: 3σ of discretization noise is not
    an outlier. Guards the Day-5 ablation's end-to-end S condition."""
    from pathlib import Path as _Path

    from pixelphys import trajectory as traj

    root = _Path(__file__).resolve().parent / "fixtures"
    obs = traj.load_observations(root / "obs_synthetic_toss_drag.json")
    assert len(traj.clean_observations(obs)) == len(obs) == 30


def test_observations_select_single_object(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import types

    from pixelphys.tracking import observations_from_records

    box_cls = type("SegmentationBoxRecord", (), {})
    mask_cls = type("SegmentationMaskRecord", (), {})

    def frame(i: int) -> types.SimpleNamespace:
        return types.SimpleNamespace(frame_index=i)

    def box(i: int, oid: str) -> object:
        r = box_cls()
        r.frame, r.object_id = frame(i), oid
        r.left, r.top, r.right, r.bottom = 10.0, 20.0, 30.0, 40.0
        return r

    def mask(i: int, oid: str) -> object:
        r = mask_cls()
        r.frame, r.object_id = frame(i), oid
        r.mask = types.SimpleNamespace(width=2, height=2)
        r.confidence = 0.9
        return r

    monkeypatch.setattr("meta_sam_parser.decode_mask_to_raster", lambda m: b"\x01\x00\x00\x01")
    records: list[object] = []
    for i in range(5):
        records.append(mask(i, "B"))  # mask before box: must still pair
        records.append(box(i, "B"))
    for i in range(3):
        records.append(box(i, "A"))
        records.append(mask(i, "A"))
    out = observations_from_records(records, fps=30.0)
    assert len(out) == 5  # only object B (most frames), no duplicates
    assert sorted(o.frame_idx for o in out) == [0, 1, 2, 3, 4]


def test_compute_cost_usd() -> None:
    from pixelphys.meta import compute_cost_usd

    assert compute_cost_usd(1_000_000, 1_000_000) == 5.5
    assert compute_cost_usd(0, 0) == 0.0


def test_validate_observations_rejects_bad_streams() -> None:
    import pytest

    from pixelphys.tracking import Observation, validate_observations

    def obs(i: int, cx: float = 1.0) -> Observation:
        return Observation(i, i / 30.0, cx, 2.0, (0.0, 0.0, 3.0, 3.0), 10.0, 1.0)

    assert len(validate_observations([obs(0), obs(1)])) == 2
    with pytest.raises(ValueError, match="no observations"):
        validate_observations([])
    with pytest.raises(ValueError, match="non-finite"):
        validate_observations([obs(0, float("nan"))])
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_observations([obs(1), obs(1)])
    with pytest.raises(ValueError, match="negative mask area"):
        validate_observations([Observation(0, 0.0, 1.0, 2.0,
                                           (0.0, 0.0, 3.0, 3.0), -1.0, 1.0)])


def test_calibrate_scale_and_pendulum_angle() -> None:
    import numpy as np

    from pixelphys import calibrate, trajectory as traj

    assert calibrate.to_meters([100.0, 250.0], 250.0) == [0.4, 1.0]
    assert calibrate.to_meters([100.0], None) == [100.0]
    # Bob directly below the pivot is angle 0; displaced right is positive.
    angles = traj.pendulum_angle(np.array([10.0, 20.0]),
                                 np.array([20.0, 20.0]), (10.0, 10.0))
    assert angles[0] == 0.0
    assert angles[1] > 0.7


def test_retries_only_transient_errors() -> None:
    import pytest

    from pixelphys import meta

    attempts = {"n": 0}

    def bad_request() -> None:
        attempts["n"] += 1
        raise ValueError("programming bug, not a transient failure")

    with pytest.raises(RuntimeError, match="after 1 attempt"):
        meta._with_retries("probe", bad_request)
    assert attempts["n"] == 1

    class RateLimited(Exception):
        status_code = 429

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RateLimited("slow down")
        return "ok"

    attempts["n"] = 0
    assert meta._with_retries("probe", flaky) == "ok"
    assert attempts["n"] == 3


def test_color_tracker_recovers_synthetic_fixture() -> None:
    """Offline fallback evidence: HSV tracker on the committed synthetic clip."""
    import json
    import statistics
    from pathlib import Path as _Path

    from pixelphys.synthetic import FRAME_W, GROUND_Y_PX, PX_PER_M
    from pixelphys.tracking import (
        SYNTHETIC_DARK_HIGH,
        SYNTHETIC_DARK_LOW,
        track_color_clip,
    )

    root = _Path(__file__).resolve().parent / "fixtures"
    truth = json.loads((root / "toss_drag.json").read_text())
    obs = track_color_clip(root / "toss_drag.mp4",
                           SYNTHETIC_DARK_LOW, SYNTHETIC_DARK_HIGH)
    expected = [(FRAME_W // 2, GROUND_Y_PX - q * PX_PER_M) for q in truth["q"]]
    assert len(obs) == len(expected)
    errs = [abs(o.cx - ex) + abs(o.cy - ey)
            for o, (ex, ey) in zip(obs, expected)]
    assert statistics.mean(errs) <= 2.0, errs
    assert max(errs) <= 4.0, max(errs)


def test_synthetic_tracker_accuracy_from_fixtures() -> None:
    """Committed evidence: tracked centroids vs ground truth, no API needed."""
    import json
    import statistics
    from pathlib import Path as _Path

    from pixelphys import trajectory as traj
    from pixelphys.synthetic import PX_PER_M, GROUND_Y_PX, FRAME_W

    root = _Path(__file__).resolve().parent / "fixtures"
    truth = json.loads((root / "toss_drag.json").read_text())
    obs = traj.load_observations(root / "obs_synthetic_toss_drag.json")
    expected = [(FRAME_W // 2, GROUND_Y_PX - q * PX_PER_M) for q in truth["q"]]
    assert len(obs) == len(expected) == len(truth["q"])
    errs = [abs(o.cx - ex) + abs(o.cy - ey)
            for o, (ex, ey) in zip(obs, expected)]
    assert statistics.mean(errs) <= 1.0, errs
    assert max(errs) <= 2.0, max(errs)
