# Approach log: what failed, why, and what to do instead.
# Add an entry after two failed attempts at the same thing (per AGENTS.md).

No entries yet.

## 2026-10-27 - `uv pip install` timeouts on Day-1 scaffold
- Tried: `uv venv` (ok) then `uv pip install -e ".[dev]"` and `uv pip install pytest` into `.venv`.
- Happened: both timed out at 120-180s (likely slow network / heavy wheels: scipy, opencv).
- Instead: verified with stdlib-only checks via `PYTHONPATH=src python3` (imports, CLI parser, config fail-fast). Retry the full install on a better connection before Day 2; do not loop it repeatedly.
