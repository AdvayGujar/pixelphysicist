"""M2: Meta API client (OpenAI-compatible) + retries + JSONL cost log.

One model family, two endpoints (plan.md Section 3):
- Muse Spark (scientist): Chat Completions, Bearer key on DEFAULT_BASE_URL.
- SAM 3.1 (sensor): POST /v1/responses on the same base URL + key.

Every API call is appended to a JSONL cost log. No key is ever logged.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_BASE_URL, PROJECT_ROOT, SPARK_MODEL_ID, load_settings

MAX_ATTEMPTS = 3
BACKOFF_S = 2.0
DEFAULT_TIMEOUT_S = 120


@dataclass(frozen=True)
class ApiCall:
    endpoint: str
    model: str
    ok: bool
    latency_s: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""


def default_log_path() -> Path:
    return PROJECT_ROOT / "runs" / "cost.jsonl"


def log_call(call: ApiCall, log_path: Path | None = None) -> Path:
    path = log_path or default_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": time.time(),
        "endpoint": call.endpoint,
        "model": call.model,
        "ok": call.ok,
        "latency_s": round(call.latency_s, 3),
        "prompt_tokens": call.prompt_tokens,
        "completion_tokens": call.completion_tokens,
        "error": call.error,
    }
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    return path


def _usage_of(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0
    return int(getattr(usage, "prompt_tokens", 0) or 0), int(
        getattr(usage, "completion_tokens", 0) or 0
    )


def _with_retries(label: str, func: Any, *args: Any, **kwargs: Any) -> Any:
    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - retried, then recorded
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_S * attempt)
    raise RuntimeError(f"{label} failed after {MAX_ATTEMPTS} attempts: {last_error}")


def get_client() -> Any:
    from openai import OpenAI

    settings = load_settings()
    return OpenAI(api_key=settings.api_key, base_url=settings.base_url or DEFAULT_BASE_URL,
                  timeout=DEFAULT_TIMEOUT_S)


def chat(messages: list[dict[str, Any]], model: str = SPARK_MODEL_ID,
         log_path: Path | None = None, **kwargs: Any) -> Any:
    """One Chat Completions call (Spark), retried and cost-logged."""
    client = get_client()
    start = time.time()
    try:
        response = _with_retries("chat", client.chat.completions.create,
                                 model=model, messages=messages, **kwargs)
    except RuntimeError as exc:
        log_call(ApiCall("chat.completions", model, False, time.time() - start,
                         error=str(exc)[:200]), log_path)
        raise
    prompt_tokens, completion_tokens = _usage_of(response)
    log_call(ApiCall("chat.completions", model, True, time.time() - start,
                     prompt_tokens, completion_tokens), log_path)
    return response


def respond(request: dict[str, Any], log_path: Path | None = None) -> Any:
    """One Responses API call (SAM 3.1), retried and cost-logged."""
    client = get_client()
    model = str(request.get("model", ""))
    start = time.time()
    try:
        response = _with_retries("responses", client.responses.create, **request)
    except RuntimeError as exc:
        log_call(ApiCall("responses", model, False, time.time() - start,
                         error=str(exc)[:200]), log_path)
        raise
    prompt_tokens, completion_tokens = _usage_of(response)
    log_call(ApiCall("responses", model, True, time.time() - start,
                     prompt_tokens, completion_tokens), log_path)
    return response
