# Demo script (60–90 s, record from SAVED runs — never a live API call)

## Assets (all on disk)
- Hero: `runs/hero_take5_onego/` — `plots/equation.png`, `overlay.mp4`, `report.html`
- Replay: `runs/mvp2_replay.mp4` (fitted synthetic drag, 3 s)
- Numbers: v1 ablation means (H 0.73→0.40, N 0.041, S 0.062→0.005), total API ≈ $1

## Shots
| Time | Visual | Narration |
|---|---|---|
| 0:00–0:08 | Hero toss clip, full frame | "Can a model discover the physics of this throw — from video alone?" |
| 0:08–0:20 | SAM mask overlay → trajectory points | "SAM 3.1 tracks the ball, frame by frame, into a single coordinate." |
| 0:20–0:32 | Equation graph: baseline curve peeling off held-out points (0.73) | "A default fit explains the throw — until motion it never saw. Held-out error: 0.73." |
| 0:32–0:55 | Baseline residual plot vs v → Spark's one-line diagnosis → refit line dropping | "Spark reads the residuals: error grows with speed squared — quadratic drag. Held error falls from 0.73 to 0.19, beating the default model 4×." |
| 0:55–1:10 | Equation graph with all three curves (baseline 0.73, gravity-only 0.065, final 0.19), ghost overlay tracking the toss | "The discovered equation — q double-dot equals −13530 plus drag — validated on motion it never saw. Note the third curve: plain gravity still wins on this short arc — the discovery mechanism is proven on synthetic truth, where drag is recovered exactly." |
| 1:10–1:25 | Replay asset (equation → ball re-flying it) + ablation card | "The equation, re-flown. Structure discovery, measured — hero −46%, synthetic −91%, 22 cents." |

## Honesty cards (on screen, 2 s each, no narration needed)
- "Fitting is classical; Spark proposes structure."
- "No drag claim on real arcs — gravity-only wins there."
- "Replay shows the fitted model, throw plane only."

## Rules
- Screen-record, no live calls. Terminal + figures only, no music.
- Show only event-time work. Under 90 s.
