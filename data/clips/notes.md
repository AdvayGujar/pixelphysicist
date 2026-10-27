# Filming notes (all takes; phone on books, perpendicular to motion plane)

No scale object in frame (steel ruler arrived after filming), so the pipeline
runs in pixel units. Scale can be added later with `--scale PX_PER_M`.

## Hero: tossed green tennis ball (prompt: "green tennis ball")

| Take | File | Frames | Role |
|---|---|---|---|
| 1 | `hero_toss_take1.mp4` | 345, 30 fps | Spare (hand-carry + tracking spike) |
| 2 | `hero_toss_take2.mp4` | 345, 30 fps | Spare (early hop before main arc) |
| 3 | `hero_toss_take3.mp4` | 96, 30 fps | Reject (tracking starts mid-fall) |
| 4 | `hero_toss_take4.mp4` | 77, 30 fps | Held-out partner (arc apex ~365 px) |
| 5 | `hero_toss_take5.mp4` | 110, 30 fps | **Hero** (clean arc, apex ~490 px @1.37 s; trims to 20 pts, 1.10–1.73 s) |

First two filmed takes were drops, not tosses (renamed `*_deferred.mp4`, not used).
Every take bounces after first impact — `trim_free_flight()` cuts launch→impact.
Free flight is short (~0.6 s), so the held-out horizon is min(2 s, clean tail).

## Negative control: pendulum, tennis-ball bob (prompt: "green tennis ball")

| Take | File | Frames | Notes |
|---|---|---|---|
| 1 | `pendulum_take1.mp4` | 267, 30 fps | Period 1.847 s, amplitude ~±330 px |
| 2 | `pendulum_take2.mp4` | 283, 30 fps | Period 1.850 s (agrees to 0.2%) |

Pivot (string top) exits the frame, so q is rest-relative horizontal
displacement, proportional to swing angle for small angles. A 7-frame
tracking gap at release in each take is handled by longest-run selection.

## Synthetic

`tests/fixtures/toss_drag.{mp4,json}`: `q'' = -g - c·v·|v|`, v0=6, 30 frames.
Regenerate with `render_toss_drag()`; tracker recovers 30/30 @ 0.5 px mean.
