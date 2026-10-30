"""Joint two-toss fit: shared coeffs, per-clip ICs (Task C).

Catches the Day-5 q*v aliasing: one trajectory lets a distractor win,
two trajectories with different v0 pin the shared physics.
"""
from pathlib import Path

import numpy as np
import pytest

from pixelphys import physics as P
from pixelphys import synthetic as syn
from pixelphys import trajectory as T

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _clean_toss(v0: float) -> tuple[np.ndarray, np.ndarray]:
    times, pos, _ = syn.integrate_toss(syn.TossParams(v0=v0))
    t = np.array(syn.sample_at_fps(times, times))
    q = np.array(syn.sample_at_fps(times, pos))
    return t, q


def _tracked_pair() -> tuple[tuple[np.ndarray, np.ndarray],
                             tuple[np.ndarray, np.ndarray]]:
    out = []
    for name in ("obs_synthetic_toss_drag.json",
                 "obs_synthetic_toss_drag_v0b.json"):
        obs = T.load_observations(FIXTURES / name)
        gaps = np.diff([o.t for o in obs])
        fps = 1.0 / float(np.median(gaps))
        obs = T.clean_observations(obs)
        obs = max(T.split_by_gaps(obs), key=len)
        t, cx, cy = T.resample_uniform(obs, fps)
        q = T.vertical_q(T.smooth_position(cy))
        out.append((t, q))
    return out[0], out[1]


def test_joint_exact_truth_identity() -> None:
    """Exact coeffs + true ICs must reproduce both clips (~0 residual).

    Regression guard for the rollout_score lesson: a refit-then-score
    path that starts the integration at the wrong state scores ~0.23
    for truth instead of ~0.
    """
    for v0 in (6.0, 7.5):
        t, q = _clean_toss(v0)
        pred, _ = P.simulate(["1", "abs(v)*v"],
                             np.array([-9.81, -0.35]), t, (0.0, v0))
        assert P.nrmse(pred, q) < 1e-6


@pytest.mark.slow
def test_joint_recovers_shared_drag() -> None:
    t1, q1 = _clean_toss(6.0)
    t2, q2 = _clean_toss(7.5)
    joint = P.fit_terms_joint(["1", "abs(v)*v"], [(t1, q1), (t2, q2)])
    assert joint.coeffs[0] == pytest.approx(-9.81, rel=0.1)
    assert joint.coeffs[1] == pytest.approx(-0.35, rel=0.2)
    assert len(joint.ics) == 2


@pytest.mark.slow
def test_joint_prefers_drag_over_qv_distractor() -> None:
    """Two tracked tosses: drag wins joint BIC though q*v wins one clip."""
    (t1, q1), (t2, q2) = _tracked_pair()
    # Single-clip aliasing: on the first toss the distractor wins held.
    cut = int(len(t1) * 0.6)
    s_drag = P.score_candidate(["1", "abs(v)*v"],
                               t1[:cut], q1[:cut], t1[cut:], q1[cut:])
    s_qv = P.score_candidate(["1", "q", "v", "q*v"],
                             t1[:cut], q1[:cut], t1[cut:], q1[cut:])
    assert s_drag.ok and s_qv.ok
    assert s_qv.held_nrmse < s_drag.held_nrmse
    # Joint: shared coeffs, per-clip ICs, BIC on the concatenated residual.
    j_drag = P.fit_terms_joint(["1", "abs(v)*v"], [(t1, q1), (t2, q2)])
    j_qv = P.fit_terms_joint(["1", "q", "v", "q*v"], [(t1, q1), (t2, q2)])

    def joint_bic(terms, joint) -> float:
        rss, n = 0.0, 0
        for (t, q), (q0, v0) in zip([(t1, q1), (t2, q2)], joint.ics):
            pred, _ = P.simulate(terms, joint.coeffs, t, (q0, v0))
            rss += float(np.sum((pred - q) ** 2))
            n += len(t)
        return P.bic(n, rss, len(terms) + 2 * 2)

    assert joint_bic(["1", "abs(v)*v"], j_drag) < \
        joint_bic(["1", "q", "v", "q*v"], j_qv)


def test_joint_rejects_single_clip() -> None:
    t, q = _clean_toss(6.0)
    with pytest.raises(ValueError, match=">= 2 clips"):
        P.fit_terms_joint(["1", "abs(v)*v"], [(t, q)])
