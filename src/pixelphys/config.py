"""M1: settings, paths, model ids. Fails fast on missing key."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

SAM_MODEL_ID = "sam-3.1"
SPARK_MODEL_ID = "muse-spark-1.3"

# Single API (SAM docs): images, video, and LLMs share one base URL + Bearer key.
DEFAULT_BASE_URL = "https://api.meta.ai/v1"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CLIPS = PROJECT_ROOT / "data" / "clips"
DATA_SYNTHETIC = PROJECT_ROOT / "data" / "synthetic"
RESULTS_DIR = PROJECT_ROOT / "results"
RUNS_DIR = PROJECT_ROOT / "runs"


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str


def load_settings() -> Settings:
    api_key = os.environ.get("MODEL_API_KEY", "")
    base_url = os.environ.get("MODEL_BASE_URL", "") or DEFAULT_BASE_URL
    if not api_key:
        raise RuntimeError("MODEL_API_KEY is not set. Copy .env.example to .env and fill it in.")
    return Settings(api_key=api_key, base_url=base_url)
