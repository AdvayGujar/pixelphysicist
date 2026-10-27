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
