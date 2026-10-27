"""M2 smoke test (plan.md Day 1 morning): one text call, one tool call.

Usage:
    cp .env.example .env   # then fill in MODEL_API_KEY
    set -a; source .env; set +a
    PYTHONPATH=src python3 scripts/smoke_meta.py

Writes to runs/cost.jsonl. Per-run cap for later Spark loops is $0.50;
this smoke test is two tiny calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixelphys import meta  # noqa: E402

TEXT_MESSAGES = [{"role": "user", "content": 'Reply with exactly: smoke-ok'}]
TOOL_MESSAGES = [{"role": "user", "content": "What is 6 times 7? Use the multiply tool."}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two integers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        },
    }
]


def main() -> int:
    text = meta.chat(TEXT_MESSAGES, max_tokens=16)
    print("text:", text.choices[0].message.content.strip())
    tool = meta.chat(TOOL_MESSAGES, tools=TOOLS, tool_choice="auto", max_tokens=64)
    calls = tool.choices[0].message.tool_calls or []
    print("tool_calls:", [(c.function.name, c.function.arguments) for c in calls])
    print("cost log:", meta.default_log_path())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
