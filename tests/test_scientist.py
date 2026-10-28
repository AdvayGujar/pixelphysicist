"""M13: scientist tool layer + loop mechanics, offline with a scripted Spark."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pixelphys import scientist as S

# Every test builds a ScientistTools, which fits the baseline by shooting;
# these are integration-speed tests, not inner-loop unit tests.
pytestmark = pytest.mark.slow

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def synthetic_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    truth = json.loads((FIXTURES / "toss_drag.json").read_text())
    t = np.array(truth["t"])
    q = np.array(truth["q"])
    cut = int(len(t) * 0.6)
    return t[:cut], q[:cut], t[cut:], q[cut:]


def make_tools(tmp_path: Path) -> S.ScientistTools:
    t_fit, q_fit, t_held, q_held = synthetic_split()
    return S.ScientistTools(t_fit, q_fit, t_held, q_held, tmp_path)


def call(name: str, arguments: dict, call_id: str = "c1") -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def response(*calls: SimpleNamespace,
             prompt_tokens: int = 1000,
             completion_tokens: int = 1000) -> SimpleNamespace:
    message = SimpleNamespace(content=None, tool_calls=list(calls) or None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens,
                              completion_tokens=completion_tokens))


def test_tools_fit_drag_and_submit_guard_accepts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    tools = make_tools(tmp_path)
    assert tools.baseline.ok
    fitted = tools.fit_model(["1", "abs(v)*v"], "quadratic drag")
    assert fitted["ok"] and fitted["held_nrmse"] < 0.01
    rejected = tools.submit_model(tools.baseline_id, "baseline is enough")
    assert rejected["accepted"]  # submitting the baseline itself is allowed
    accepted = tools.submit_model(fitted["model_id"], "drag fits held-out")
    assert accepted["accepted"]
    assert tools.submitted_id == fitted["model_id"]


def test_submit_guard_rejects_worse_than_baseline(tmp_path) -> None:  # type: ignore[no-untyped-def]
    tools = make_tools(tmp_path)
    worse = tools.fit_model(["v**2"], "unsigned square only")
    assert worse["ok"]
    rejected = tools.submit_model(worse["model_id"], "trust me")
    assert not rejected["accepted"]
    assert "baseline" in rejected["error"]


def test_fit_model_returns_failure_as_data(tmp_path) -> None:  # type: ignore[no-untyped-def]
    tools = make_tools(tmp_path)
    bad = tools.fit_model(["q.__class__"], "injection attempt")
    assert not bad["ok"] and bad["error"]
    unknown = tools.plot_residuals("model-999", "t")
    assert "unknown" in unknown["error"]


def test_plot_residuals_writes_png(tmp_path) -> None:  # type: ignore[no-untyped-def]
    tools = make_tools(tmp_path)
    result = tools.plot_residuals(tools.baseline_id, "v")
    assert Path(result["path"]).exists()
    assert result["versus"] == "v"


def test_loop_scripted_spark_finds_drag(tmp_path) -> None:  # type: ignore[no-untyped-def]
    tools = make_tools(tmp_path)
    rounds = [
        response(call("fit_model",
                      {"terms": ["1", "abs(v)*v"],
                       "rationale": "residual grows with speed squared"})),
        response(call("submit_model",
                      {"model_id": "model-1",
                       "explanation": "quadratic drag recovers held-out"})),
    ]
    result = S.run_loop(tools, lambda messages, tools_: rounds.pop(0))
    assert result.submitted
    assert result.terms == ["1", "abs(v)*v"]
    assert result.final_model_id == "model-1"
    assert result.cost_usd > 0
    assert result.stopped_reason == "submitted"
    assert result.held_nrmse is not None and result.held_nrmse < 0.01
    assert result.bic is not None and result.train_rmse is not None
    assert result.transcript[0].get("baseline_plots"), \
        "opening message must attach baseline residual evidence"


def test_loop_fallback_submits_best_when_never_submitted(tmp_path) -> None:  # type: ignore[no-untyped-def]
    tools = make_tools(tmp_path)
    always_fit = lambda messages, tools_: response(  # noqa: E731
        call("fit_model", {"terms": ["v**2"], "rationale": "same again"}))
    result = S.run_loop(tools, always_fit, max_rounds=3)
    assert not result.submitted
    assert result.final_model_id == tools.baseline_id  # baseline stays best
    assert "fallback" in result.stopped_reason


def test_loop_cost_cap_stops_rounds(tmp_path) -> None:  # type: ignore[no-untyped-def]
    tools = make_tools(tmp_path)
    pricey = lambda messages, tools_: response(  # noqa: E731
        call("fit_model", {"terms": ["v**2"], "rationale": "x"}),
        prompt_tokens=1_000_000, completion_tokens=1_000_000)
    result = S.run_loop(tools, pricey, max_rounds=6, cost_cap_usd=0.50)
    assert result.rounds < 6
    assert "cost cap" in result.stopped_reason


def test_system_prompt_guides_sparse_replacement() -> None:
    assert "sparser version" in S.SYSTEM_PROMPT  # Day-5 H pilot: additive drag fails to integrate
    assert "submit_model promptly" in S.SYSTEM_PROMPT  # Day-5 H2: submit before more plots


def test_loop_plot_rounds_do_not_burn_stagnation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Plot-only rounds are diagnosis, not failed improvement (Day-5 H2).

    Script: worse fit, plot, same worse fit, drag fit, submit. Old logic
    stopped at the end of round 3 (stagnant==2) and never submitted; the
    fix skips plot rounds so the drag fit in round 4 still runs.
    """
    tools = make_tools(tmp_path)
    rounds = [
        response(call("fit_model",
                      {"terms": ["v**2"], "rationale": "unsigned square"})),
        response(call("plot_residuals",
                      {"model_id": "model-0", "versus": "t"})),
        response(call("fit_model",
                      {"terms": ["v**2"], "rationale": "same again"})),
        response(call("fit_model",
                      {"terms": ["1", "abs(v)*v"],
                       "rationale": "sparse quadratic drag"})),
        response(call("submit_model",
                      {"model_id": "model-3",
                       "explanation": "drag beats baseline on held-out"})),
    ]
    result = S.run_loop(tools, lambda messages, tools_: rounds.pop(0))
    assert result.submitted
    assert result.terms == ["1", "abs(v)*v"]
    assert result.stopped_reason == "submitted"
