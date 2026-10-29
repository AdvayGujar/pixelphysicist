"""M10: equation graph + ghost overlay + report.html. (Day 6.)

The equation graph (plan Section 4.6) is the primary visual: measured
points, baseline curve, final-model curve (solid on fit, dashed on held),
the discovered equation with coefficients, and both held-out NRMSEs --
on the same axes. The overlay draws the fitted prediction as a ghost over
the real footage; the report bundles everything into one self-contained
HTML file. All figures use the Agg backend (headless-safe).
"""
from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def render_equation(terms: Sequence[str], coeffs: Sequence[float],
                    lhs: str = "q''") -> str:
    """One-line equation ``LHS = c0*t0 + ...`` with 4-significant digits."""
    parts = [f"{c:.4g}*{t}" for c, t in zip(coeffs, terms)]
    return f"{lhs} = " + " + ".join(parts)


def plot_equation(t_fit: np.ndarray, q_fit: np.ndarray,
                  t_held: np.ndarray, q_held: np.ndarray,
                  baseline_terms: Sequence[str], baseline_coeffs: Sequence[float],
                  baseline_q0: float, baseline_v0: float, baseline_held: float,
                  final_terms: Sequence[str], final_coeffs: Sequence[float],
                  final_q0: float, final_v0: float, final_held: float,
                  out_path: Path, units: str = "px",
                  title: str = "held-out validation",
                  lhs: str = "q''",
                  control_terms: Sequence[str] | None = None,
                  control_coeffs: Sequence[float] | None = None,
                  control_q0: float | None = None, control_v0: float | None = None,
                  control_held: float | None = None,
                  control_label: str = "control",
                  pred_base: np.ndarray | None = None,
                  pred_final: np.ndarray | None = None,
                  pred_control: np.ndarray | None = None) -> Path:
    """Draw the Section 4.6 equation graph; return the written PNG path.

    An optional control model (e.g. gravity-only) is drawn as a third
    curve so the demo narrative can never present the final model as the
    win while a simpler control wins off-screen (audit Finding 2).

    ``pred_*`` are precomputed full-grid curves (used by the 2-DOF CLI,
    whose models the 1-DOF simulator below cannot integrate); when given,
    the matching q0/v0/terms are labels only and no simulation runs.
    """
    from . import physics as P

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t_fit_a = np.asarray(t_fit, dtype=float)
    q_fit_a = np.asarray(q_fit, dtype=float)
    t_held_a = np.asarray(t_held, dtype=float)
    q_held_a = np.asarray(q_held, dtype=float)
    grid = np.concatenate([t_fit_a, t_held_a])
    if pred_base is None:
        base_q, _ = P.simulate(list(baseline_terms), np.asarray(baseline_coeffs),
                               grid, (float(baseline_q0), float(baseline_v0)))
    else:
        base_q = np.asarray(pred_base, dtype=float)
        assert base_q.shape == grid.shape, "pred_base must span the full grid"
    if pred_final is None:
        fin_q, _ = P.simulate(list(final_terms), np.asarray(final_coeffs),
                              grid, (float(final_q0), float(final_v0)))
    else:
        fin_q = np.asarray(pred_final, dtype=float)
        assert fin_q.shape == grid.shape, "pred_final must span the full grid"
    n_fit = len(t_fit_a)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.scatter(t_fit_a, q_fit_a, s=14, c="black", label="measured (fit window)")
    ax.scatter(t_held_a, q_held_a, s=14, c="black", marker="x",
               label="measured (held out)")
    ax.plot(grid, base_q, lw=1.2, ls="-",
            label=f"baseline [{', '.join(baseline_terms)}] held NRMSE {baseline_held:.3f}")
    if control_terms is not None:
        assert (control_coeffs is not None and control_q0 is not None
                and control_v0 is not None and control_held is not None)
        if pred_control is None:
            ctrl_q, _ = P.simulate(list(control_terms), np.asarray(control_coeffs),
                                   grid, (float(control_q0), float(control_v0)))
        else:
            ctrl_q = np.asarray(pred_control, dtype=float)
            assert ctrl_q.shape == grid.shape, "pred_control must span the full grid"
        ax.plot(grid, ctrl_q, lw=1.2, ls=":",
                label=f"{control_label} [{', '.join(control_terms)}] "
                      f"held NRMSE {control_held:.3f}")
    ax.plot(grid[:n_fit], fin_q[:n_fit], lw=1.6,
            label=f"final [{', '.join(final_terms)}] fit")
    ax.plot(grid[n_fit - 1:], fin_q[n_fit - 1:], lw=1.6, ls="--",
            label=f"final held (dashed) NRMSE {final_held:.3f}")
    ax.set_xlabel("t (s)")
    ax.set_ylabel(f"q ({units})")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    equation = render_equation(final_terms, final_coeffs, lhs=lhs)
    ax.text(0.02, 0.98, f"{equation}\nunits: {units}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            family="monospace",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "0.7"})
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def render_overlay(video_path: Path, t_pred: np.ndarray,
                   xy_pred: np.ndarray, out_path: Path,
                   fps: float | None = None, radius: int = 10,
                   t_span: tuple[float, float] | None = None) -> Path:
    """Draw predicted (x, y) pixel positions as a ghost over the footage.

    ``t_pred``/``xy_pred`` come from the caller (the CLI maps the fitted
    1-DOF/2-DOF trajectory back to pixels); frames sample the prediction
    at each frame time by linear interpolation. With ``t_span`` the ghost
    is blanked outside the fitted window instead of freezing clamped at
    the edges (audit Finding 13).
    """
    import cv2

    t_pred_a = np.asarray(t_pred, dtype=float)
    xy_pred_a = np.asarray(xy_pred, dtype=float)
    if t_pred_a.ndim != 1 or xy_pred_a.ndim != 2 or xy_pred_a.shape != (len(t_pred_a), 2):
        raise ValueError("t_pred must be 1-D and xy_pred (n, 2) with matching n")
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video {video_path}")
        src_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
        fps = fps or src_fps
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        try:
            idx = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                t = idx / fps
                in_span = (t_span is None
                           or (t_span[0] <= t <= t_span[1]))
                if in_span:
                    x = float(np.interp(t, t_pred_a, xy_pred_a[:, 0]))
                    y = float(np.interp(t, t_pred_a, xy_pred_a[:, 1]))
                    cv2.circle(frame, (int(round(x)), int(round(y))), radius,
                               (0, 0, 255), 2)
                writer.write(frame)
                idx += 1
        finally:
            writer.release()
        if idx == 0:
            raise ValueError(f"no frames read from {video_path}")
        return out
    finally:
        capture.release()


def _embed_image(path: Path) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" style="max-width:100%"/>'


def render_replay_2d(terms_x: Sequence[str], coeffs_x: Sequence[float],
                     terms_y: Sequence[str], coeffs_y: Sequence[float],
                     state0: tuple[float, float, float, float],
                     t_grid: np.ndarray, out_path: Path, fps: float = 30.0,
                     caption: str = "") -> Path:
    """MVP-2 Phase C: re-fly fitted projectile equations as an MP4 replay.

    The ball is driven by integrating the *exported equations* (never the
    tracked points), drawn on the synthetic renderer geometry (throw plane
    only -- a monocular clip determines no out-of-plane motion). The full
    fitted path is drawn faintly behind the ball; the caption names the
    fitted model. Offline OpenCV render, no network.
    """
    import cv2

    from . import physics as P
    from . import synthetic as syn

    cx = np.asarray(coeffs_x, dtype=float)
    cy = np.asarray(coeffs_y, dtype=float)
    t = np.asarray(t_grid, dtype=float)
    px, py, _, _ = P.simulate_projectile(list(terms_x), list(terms_y),
                                         cx, cy, t, tuple(state0))
    pts = [syn.projectile_center_px(float(x), float(y)) for x, y in zip(px, py)]
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (syn.FRAME_W, syn.FRAME_H))
    try:
        n = len(pts)
        for i, (bx, by) in enumerate(pts):
            frame = np.full((syn.FRAME_H, syn.FRAME_W, 3), syn.BACKGROUND,
                            dtype=np.uint8)
            cv2.line(frame, (0, syn.GROUND_Y_PX),
                     (syn.FRAME_W, syn.GROUND_Y_PX), (200, 200, 200), 1)
            for j in range(1, n):
                cv2.line(frame, pts[j - 1], pts[j], (180, 180, 180), 1)
            cv2.circle(frame, (bx, by), syn.DOT_RADIUS_PX, syn.DOT_COLOR, -1)
            if caption:
                cv2.putText(frame, caption[:90], (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
            cv2.putText(frame, "fitted model (throw plane only)", (10, 464),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
            writer.write(frame)
    finally:
        writer.release()
    return out


def build_report(out_path: Path, title: str, equation: str, units: str,
                 rows: Sequence[dict[str, Any]],
                 figure_paths: Sequence[Path],
                 transcript: str = "", cost_usd: float | None = None) -> Path:
    """Write a self-contained report.html (images inlined as base64)."""
    safe_title = html.escape(title)
    body = [f"<h1>{safe_title}</h1>",
            f"<p><code>{html.escape(equation)}</code> &nbsp;units: {html.escape(units)}</p>",
            "<h2>Models</h2>",
            "<table border='1' cellpadding='4'><tr>"
            "<th>model</th><th>terms</th><th>held NRMSE</th><th>BIC</th>"
            "<th>train RMSE</th><th>rounds</th><th>submitted</th></tr>"]
    for row in rows:
        terms = ", ".join(str(t) for t in row.get("terms", []))
        body.append(
            "<tr><td>{}</td><td><code>{}</code></td><td>{}</td><td>{}</td>"
            "<td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(str(row.get("model", ""))),
                html.escape(terms),
                html.escape(str(row.get("held_nrmse", ""))),
                html.escape(str(row.get("bic", ""))),
                html.escape(str(row.get("train_rmse", ""))),
                html.escape(str(row.get("rounds", ""))),
                html.escape(str(row.get("submitted", "")))))
    body.append("</table><h2>Figures</h2>")
    for path in figure_paths:
        body.append(_embed_image(path))
    if transcript:
        body.append("<h2>Transcript</h2><pre>{}</pre>".format(html.escape(transcript)))
    if cost_usd is not None:
        body.append(f"<p>API cost: ${cost_usd:.4f}</p>")
    doc = ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
           f"<title>{safe_title}</title></head><body>"
           + "\n".join(body) + "</body></html>")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    return out
