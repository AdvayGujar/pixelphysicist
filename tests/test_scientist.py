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


def test_submit_guard_falls_back_to_gravity_when_baseline_explodes(
        tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Benchmark loophole: a failed baseline must not accept anything.

    With the baseline entry replaced by a failure, the guard compares
    against gravity-only: a model that loses to gravity is rejected,
    the drag model (which beats it) is accepted.
    """
    import numpy as np

    from pixelphys import physics as P
    from pixelphys import scientist as S

    tools = make_tools(tmp_path)
    failed = P.ScoredFit(ok=False, terms=list(P.BASELINE_TERMS),
                         coeffs=np.full(4, np.nan), q0=float("nan"),
                         v0=float("nan"), train_rmse=float("inf"),
                         held_nrmse=float("inf"), bic=float("inf"),
                         n_scored=0, cost=float("inf"), start_index=-1,
                         error="rollout exploded")
    tools.models[tools.baseline_id] = S._Entry(failed, "exploded", tight=True)
    worse = tools.fit_model(["v**2"], "unsigned square only")
    rejected = tools.submit_model(worse["model_id"], "trust me")
    assert not rejected["accepted"]
    assert rejected["reference"] == "gravity-only (baseline failed)"
    drag = tools.fit_model(["1", "abs(v)*v"], "quadratic drag")
    accepted = tools.submit_model(drag["model_id"], "drag beats gravity")
    assert accepted["accepted"]


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


def test_fit_model_memoizes_identical_terms(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Audit A1: a repeated fit returns the identical score in <0.1 s."""
    import time

    from pixelphys import physics as P

    tools = make_tools(tmp_path)
    first = tools.fit_model(["1", "abs(v)*v"], "quadratic drag")
    assert first["ok"] and first["held_nrmse"] < 0.05
    start = time.monotonic()
    second = tools.fit_model(["1", "abs(v)*v"], "same again")
    assert time.monotonic() - start < 0.1
    assert second["held_nrmse"] == first["held_nrmse"]
    assert second["bic"] == first["bic"]
    assert P.ScoredFit is not None  # import check for guard tests below


def test_precomputed_baseline_avoids_refit(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Audit A2: a passed-in baseline is registered, never refit."""
    from pixelphys import physics as P

    t_fit, q_fit, t_held, q_held = synthetic_split()
    base = P.score_candidate(list(P.BASELINE_TERMS), t_fit, q_fit, t_held, q_held)
    tools = S.ScientistTools(t_fit, q_fit, t_held, q_held, tmp_path, baseline=base)
    assert tools.baseline is base
    assert tools.baseline_id == "model-0"


def test_two_stage_screen_then_tight(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Audit A4: fit_model reports the screen; submit tightens it."""
    tools = make_tools(tmp_path)
    fitted = tools.fit_model(["1", "abs(v)*v"], "quadratic drag")
    assert fitted["ok"] and fitted["screening"] is True
    assert fitted["best_bic_so_far"] is not None
    accepted = tools.submit_model(fitted["model_id"], "drag fits held-out")
    assert accepted["accepted"] and accepted["screening"] is False


def test_submit_guard_v2_requires_all_three(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Audit B6: v2 rejects an NRMSE-only win; v1 still accepts it."""
    from pixelphys import physics as P

    def tools_with_fake(guard: str):
        t_fit, q_fit, t_held, q_held = synthetic_split()
        tools = S.ScientistTools(t_fit, q_fit, t_held, q_held, tmp_path,
                                 guard=guard)
        base = tools.baseline
        assert base.ok
        mid = tools._register(P.ScoredFit(
            ok=True, terms=["1", "v"], coeffs=np.array([0.0, -0.1]),
            q0=0.0, v0=0.0, train_rmse=base.train_rmse * 1.01,
            held_nrmse=base.held_nrmse * 0.5, bic=base.bic + 10.0,
            n_scored=1, cost=0.0, start_index=0, error=None), "nrmse win only")
        return tools, mid

    tools_v2, mid_v2 = tools_with_fake("v2")
    assert not tools_v2.submit_model(mid_v2, "x")["accepted"]
    tools_v1, mid_v1 = tools_with_fake("v1")
    assert tools_v1.submit_model(mid_v1, "x")["accepted"]


def test_submit_guard_v3_drops_train_clause(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Guard v3: held AND BIC decide; train RMSE is not consulted.

    Accepts a model beating held+BIC while losing train by >5%, and
    rejects one winning held but losing BIC.
    """
    from pixelphys import physics as P

    t_fit, q_fit, t_held, q_held = synthetic_split()
    tools = S.ScientistTools(t_fit, q_fit, t_held, q_held, tmp_path,
                             guard="v3")
    base = tools.baseline
    assert base.ok
    # Beats held (>=2%) and BIC, but loses train by 50%: v2 rejects, v3 accepts.
    good = tools._register(P.ScoredFit(
        ok=True, terms=["1", "abs(v)*v"],
        coeffs=np.array([-9.81, -0.35]), q0=0.0, v0=6.0,
        train_rmse=base.train_rmse * 1.5,
        held_nrmse=base.held_nrmse * 0.5, bic=base.bic - 10.0,
        n_scored=1, cost=0.0, start_index=0, error=None), "held+bic win")
    assert tools.submit_model(good, "drag beats held and BIC")["accepted"]
    # Wins held but loses BIC: v1 accepts (OR), v3 rejects (AND).
    luck = tools._register(P.ScoredFit(
        ok=True, terms=["1", "q", "v"],
        coeffs=np.array([0.0, 1.0, 0.1]), q0=0.0, v0=0.0,
        train_rmse=base.train_rmse,
        held_nrmse=base.held_nrmse * 0.5, bic=base.bic + 10.0,
        n_scored=1, cost=0.0, start_index=0, error=None), "held luck")
    rejected = tools.submit_model(luck, "held luck")
    assert not rejected["accepted"]
    assert "BIC" in rejected["error"]


def test_v3_prompt_matches_guard() -> None:
    assert S._V3_RULE in S.V3_SYSTEM_PROMPT
    assert S._V1_RULE not in S.V3_SYSTEM_PROMPT
    assert "Train RMSE is not part of this decision" in S.V3_SYSTEM_PROMPT
    assert "not part of the decision" in S.v3_schemas()[2]["function"]["description"]


def test_loop_fit_call_cap(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Audit A5: fit calls beyond the cap fail fast and are surfaced."""
    tools = make_tools(tmp_path)
    always_fit = lambda messages, tools_: response(  # noqa: E731
        call("fit_model", {"terms": ["v**2"], "rationale": "same again"}))
    result = S.run_loop(tools, always_fit, max_rounds=6, max_fit_calls=2)
    assert any(ev.get("event") == "fit_cap" for ev in result.transcript)
    assert "fit-call cap reached" in result.stopped_reason
