"""Task B runner (v2 method): pendulum stiffness-vs-period consistency.

v1 (superseded, results kept in git history) fitted a circle pivot to
raw centroids, took arctan2 WITHOUT unwrapping (spurious zero
crossings every time the angle wrapped past pi), and compared a
linear-q coefficient against -(2pi/T)^2 -- three artefacts stacked on
the physics. v2 fixes the method, not the clips and not the bar:

- angle is unwrapped before any analysis;
- period comes from peak spacing of the unwrapped angle (FFT fallback
  for full rotations), not zero crossings;
- the fitted restoring term is sin(q) (the actual pendulum restoring
  term), and the comparison is period-vs-period: the sin fit implies
  a small-angle period T0 = 2*pi/sqrt(|k_sin|), corrected to the
  measured swing amplitude via the elliptic factor
  T = T0 * (2/pi) * K(sin^2(theta0/2)). Reported per clip: measured
  period, amplitude-corrected prediction, relative agreement, plus
  the v1 numbers for continuity. Circle-fit RMS is reported so a bad
  pivot shows up as a bad pivot.

Success (unchanged bar, corrected estimator): corrected period
agreement within 5% on at least 3 of the clips. All 5 clips are
always reported; none are dropped.

Usage:
    .venv/bin/python scripts/pendulum_consistency.py
        [--clips-dir /tmp/stress2] [--out results/pendulum_consistency.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixelphys import physics as P  # noqa: E402
from pixelphys import trajectory as T  # noqa: E402
from pixelphys.tracking import track_motion_clip  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CLIPS = {f"pend_h{i}": f"holonomic_pendulum/video_{i}_fps30"
         for i in range(1, 6)}


def sanitize(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    return value


def _circle_lsq(cx: np.ndarray, cy: np.ndarray) -> tuple[float, float, float]:
    a = np.column_stack([2.0 * cx, 2.0 * cy, np.ones_like(cx)])
    b = cx**2 + cy**2
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    px, py, c = float(sol[0]), float(sol[1]), float(sol[2])
    radius = math.sqrt(max(c + px**2 + py**2, 0.0))
    return px, py, radius


def fit_pivot(cx: np.ndarray, cy: np.ndarray) -> tuple[float, float, float, float]:
    """Circle fit with one trimmed refit; (px, py, radius, rms px).

    The tracked centroid rides the rod as well as the bob, so a plain
    algebraic fit is dragged off the true arc (RMS 28-35 px on the
    hard clips). Refit once on the best 80% of residuals.
    """
    px, py, radius = _circle_lsq(cx, cy)
    resid = np.abs(np.hypot(cx - px, cy - py) - radius)
    keep = resid <= np.quantile(resid, 0.8)
    if keep.sum() >= 20:
        px, py, radius = _circle_lsq(cx[keep], cy[keep])
    resid = np.hypot(cx - px, cy - py) - radius
    return px, py, radius, float(np.sqrt(np.mean(resid**2)))


def measure_period(t: np.ndarray, angle: np.ndarray,
                   ) -> tuple[float, float, int]:
    """Period and amplitude measured over the SAME early cycles.

    A damped pendulum's period drifts as amplitude decays, so the
    period (first 3 peak intervals) and the amplitude (same span)
    must be estimated together; mixing an early amplitude with a
    whole-clip median period manufactures disagreement. Returns
    (period, amplitude, n_peaks). FFT fallback for full rotations.
    """
    from scipy.signal import find_peaks

    smooth = T.smooth_position(angle)
    peaks, _ = find_peaks(smooth, distance=max(5, len(t) // 20))
    if len(peaks) >= 4:
        early = peaks[:4]
        period = float(np.median(np.diff(t[early])))
        span = angle[: early[-1] + 1]
        amplitude = float((span.max() - span.min()) / 2.0)
        return period, amplitude, len(peaks)
    if len(peaks) >= 3:
        period = float(np.median(np.diff(t[peaks])))
        amplitude = float((angle.max() - angle.min()) / 2.0)
        return period, amplitude, len(peaks)
    centred = smooth - smooth.mean()
    freqs = np.fft.rfftfreq(len(t), d=float(np.median(np.diff(t))))
    power = np.abs(np.fft.rfft(centred))
    power[0] = 0.0
    amplitude = float((angle.max() - angle.min()) / 2.0)
    return float(1.0 / freqs[int(np.argmax(power))]), amplitude, len(peaks)


def bob_points(local: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track the bob as the farthest moving-contour point from the pivot.

    The motion mask covers rod + bob, so its centroid rides the rod
    and circle-fits badly (RMS 28-35 px on the hard clips). The bob is
    the rod's far end: iterate (centroid pivot -> farthest contour
    point -> refit pivot on those points) twice.
    """
    import cv2

    capture = cv2.VideoCapture(str(local))
    frames = []
    grays = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    capture.release()
    sample = grays[:: max(1, len(grays) // 25)]
    background = np.median(np.stack(sample), axis=0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    contours_per_frame: list[np.ndarray | None] = []
    centroids = []
    for gray in grays:
        mask = cv2.threshold(cv2.absdiff(gray, background), 25, 255,
                             cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= 20.0]
        if not contours:
            contours_per_frame.append(None)
            centroids.append((np.nan, np.nan))
            continue
        biggest = max(contours, key=cv2.contourArea)
        contours_per_frame.append(biggest.reshape(-1, 2).astype(float))
        m = cv2.moments(biggest)
        centroids.append((m["m10"] / m["m00"], m["m01"] / m["m00"]))
    cents = np.array(centroids, dtype=float)
    good = np.isfinite(cents[:, 0])
    px, py, _, _ = fit_pivot(cents[good, 0], cents[good, 1])
    bob = cents.copy()
    for _ in range(2):
        for i, contour in enumerate(contours_per_frame):
            if contour is None or not good[i]:
                continue
            dist = np.hypot(contour[:, 0] - px, contour[:, 1] - py)
            bob[i] = contour[int(np.argmax(dist))]
        px, py, _, _ = fit_pivot(bob[good, 0], bob[good, 1])
    idx = np.where(good)[0]
    fps = 30.0
    return idx / fps, bob[good, 0], bob[good, 1]


def analyse(local: Path) -> dict:
    obs = track_motion_clip(local)
    obs = T.clean_observations(obs)
    obs = max(T.split_by_gaps(obs), key=len)
    gaps = np.diff([o.t for o in obs])
    fps = 1.0 / float(np.median(gaps))
    t, cx, cy = T.resample_uniform(obs, fps)
    cx_s, cy_s = T.smooth_position(cx), T.smooth_position(cy)
    # Bob-point track (rod-end) replaces the rod-riding centroid when
    # it yields the better circle -- judged by pivot RMS, not results.
    try:
        bt, bx, by = bob_points(local)
        if len(bt) >= 30:
            bpx, bpy, _, brms = fit_pivot(T.smooth_position(bx),
                                          T.smooth_position(by))
            cpx, cpy, _, crms = fit_pivot(cx_s, cy_s)
            if brms < crms:
                t = bt
                gaps = np.diff(t)
                fps = 1.0 / float(np.median(gaps)) if len(gaps) else fps
                cx_s, cy_s = T.smooth_position(bx), T.smooth_position(by)
    except Exception:
        pass  # centroid track stands; failure surfaces in pivot RMS
    px, py, radius, pivot_rms = fit_pivot(cx_s, cy_s)
    angle = np.unwrap(np.arctan2(cx_s - px, cy_s - py))
    period, amplitude, n_peaks = measure_period(t, angle)
    cut = int(len(t) * 0.6)

    fit = P.score_candidate(["1", "sin(q)", "v"], t[:cut], angle[:cut],
                            t[cut:], angle[cut:])
    entry: dict = {
        "n": len(t), "fps": fps,
        "pivot_px": [px, py], "radius_px": radius,
        "pivot_fit_rms_px": pivot_rms,
        "amplitude_rad": amplitude,
        "n_peaks": n_peaks,
        "period_measured_s": period,
        "ok": bool(fit.ok), "error": fit.error,
        "held_nrmse": float(fit.held_nrmse) if fit.ok else None,
    }
    if fit.ok:
        from scipy.signal import find_peaks
        from scipy.special import ellipk

        coeffs = dict(zip(fit.terms, fit.coeffs))
        k_sin = float(coeffs["sin(q)"])
        entry["k_sin"] = k_sin
        if k_sin < 0:
            t0 = 2.0 * math.pi / math.sqrt(-k_sin)
            # Per-interval comparison: a damped swing's period belongs
            # to the amplitude OF THAT INTERVAL, not the release
            # amplitude. Predict each peak-to-peak interval from its
            # local amplitude (peak to following trough) and compare
            # medians -- release-amplitude prediction overstates the
            # period once damping has bitten.
            smooth = T.smooth_position(angle)
            peaks, _ = find_peaks(smooth, distance=max(5, len(t) // 20))
            troughs, _ = find_peaks(-smooth, distance=max(5, len(t) // 20))
            predicted = []
            for pk in peaks[:-1]:
                after = troughs[troughs > pk]
                if len(after) == 0:
                    continue
                local_amp = float((smooth[pk] - smooth[after[0]]) / 2.0)
                m = math.sin(min(abs(local_amp), 1.5) / 2.0) ** 2
                predicted.append(t0 * (2.0 / math.pi) * float(ellipk(m)))
            if predicted:
                t_pred = float(np.median(predicted))
                entry["period_predicted_s"] = t_pred
                entry["rel_err_corrected"] = abs(t_pred - period) / period
                entry["pass_5pct"] = bool(entry["rel_err_corrected"] <= 0.05)
        # v1 continuity columns: linear fit vs raw -(2pi/T)^2.
        lin = P.score_candidate(list(P.BASELINE_TERMS), t[:cut], angle[:cut],
                                t[cut:], angle[cut:])
        if lin.ok:
            k_lin = float(dict(zip(lin.terms, lin.coeffs))["q"])
            expected = -(2.0 * math.pi / period) ** 2
            entry["v1"] = {"fitted_k": k_lin, "expected_k": expected,
                           "rel_err": abs(k_lin - expected) / abs(expected)}
    return entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pendulum consistency (v2 method).")
    parser.add_argument("--clips-dir", default="/tmp/stress2")
    parser.add_argument("--out", default="results/pendulum_consistency.json")
    args = parser.parse_args(argv)
    clips_dir = Path(args.clips_dir)

    out_clips: dict[str, dict] = {}
    for name, experiment in CLIPS.items():
        local = clips_dir / f"{name}.mp4"
        try:
            entry = analyse(local)
            entry = {"experiment": experiment, "local": str(local), **entry}
        except (ValueError, RuntimeError) as exc:
            entry = {"experiment": experiment, "local": str(local),
                     "error": f"{type(exc).__name__}: {exc}",
                     "pass_5pct": False}
        out_clips[name] = entry
        print(f"{name}: pass={entry.get('pass_5pct')} "
              f"T_meas={entry.get('period_measured_s')} "
              f"T_pred={entry.get('period_predicted_s')} "
              f"rel={entry.get('rel_err_corrected')}", flush=True)

    n_pass = sum(1 for c in out_clips.values() if c.get("pass_5pct"))
    doc = {
        "method": "v2: motion tracker -> circle-fit pivot (RMS reported) "
                  "-> unwrapped angle -> period from peak spacing -> "
                  "sin(q) fit, period predicted with elliptic amplitude "
                  "correction; v1 columns retained for continuity",
        "criterion": "corrected period agreement within 5% on at least "
                     "3 clips (bar unchanged from v1; estimator fixed)",
        "n_pass": n_pass, "n_total": len(out_clips),
        "success": bool(n_pass >= 3),
        "clips": out_clips,
    }
    out_path = REPO / args.out
    out_path.write_text(json.dumps(sanitize(doc), indent=1))
    print(f"{n_pass}/{len(out_clips)} pass -> success={doc['success']}; "
          f"wrote {args.out}")
    return 0 if doc["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
