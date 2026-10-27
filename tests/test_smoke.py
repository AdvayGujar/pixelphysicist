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
