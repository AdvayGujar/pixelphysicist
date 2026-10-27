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
    monkeypatch.delenv("MODEL_BASE_URL", raising=False)
    try:
        config.load_settings()
    except RuntimeError:
        return
    raise AssertionError("load_settings should fail fast without MODEL_API_KEY")
