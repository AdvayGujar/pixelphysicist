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

    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    try:
        config.load_settings()
    except RuntimeError:
        return
    raise AssertionError("load_settings should fail fast without MODEL_API_KEY")


def test_config_defaults_to_meta_base_url(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from pixelphys import config

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
