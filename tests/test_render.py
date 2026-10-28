"""M13: render tests -- equation graph, overlay, report (all offline)."""

import json
from pathlib import Path

import numpy as np

from pixelphys import physics as P
from pixelphys import render as R
from pixelphys import synthetic as syn

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def synthetic_split():
    truth = json.loads((FIXTURES / "toss_drag.json").read_text())
    t = np.array(truth["t"])
    q = np.array(truth["q"])
    cut = int(len(t) * 0.6)
    return t[:cut], q[:cut], t[cut:], q[cut:]


def test_render_equation_lists_terms_and_scores(tmp_path):
    t_fit, q_fit, t_held, q_held = synthetic_split()
    base = P.score_candidate(list(P.BASELINE_TERMS), t_fit, q_fit, t_held, q_held)
    final = P.score_candidate(["1", "abs(v)*v"], t_fit, q_fit, t_held, q_held)
    assert base.ok and final.ok
    out = R.plot_equation(t_fit, q_fit, t_held, q_held,
                          base.terms, base.coeffs, base.q0, base.v0,
                          base.held_nrmse,
                          final.terms, final.coeffs, final.q0, final.v0,
                          final.held_nrmse,
                          tmp_path / "equation.png", units="m",
                          title="synthetic check")
    assert out.exists() and out.stat().st_size > 0


def test_render_overlay_draws_ghost(tmp_path):
    truth = json.loads((FIXTURES / "toss_drag.json").read_text())
    t = np.array(truth["t"])
    xy = np.column_stack([np.full_like(t, syn.FRAME_W / 2.0),
                          syn.GROUND_Y_PX - np.array(truth["q"]) * syn.PX_PER_M])
    out = R.render_overlay(FIXTURES / "toss_drag.mp4", t, xy,
                           tmp_path / "overlay.mp4")
    assert out.exists() and out.stat().st_size > 0


def test_build_report_is_self_contained(tmp_path):
    fig = tmp_path / "fig.png"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot([0, 1], [0, 1])
    plt.savefig(fig)
    plt.close()
    out = R.build_report(tmp_path / "report.html", "demo run",
                         "q'' = -9.81*1 + -0.35*abs(v)*v", "m",
                         [{"model": "B", "terms": ["1", "abs(v)*v"],
                           "held_nrmse": 1e-9, "bic": -788.0,
                           "train_rmse": 2e-10, "rounds": 4,
                           "submitted": True}],
                         [fig], transcript="submitted", cost_usd=0.01)
    text = out.read_text()
    assert "abs(v)*v" in text and "data:image/png;base64" in text
