# October 13 headset acceptance: partial fix, tracking still inadequate

## Verdict

**The measured >=10 processed-camera-FPS target passes. Reliable motion tracking
does not.** Boxes no longer disappear almost completely, but most displayed
tracking geometry is explicitly `YOLO_UNPROPAGATED`: original YOLO coordinates
that could not be carried forward. Keeping these bounded stale boxes visible
fixed detector handoff, not optical tracking itself. The current detector-only
control is faster than every tracking mode in both processed-camera throughput
and semantic result rate.

## What was evaluated

Eight user-supplied CSVs: October 13 timing/tracking pairs for OT3, OT5, OT10 and
Control. All report GPUCompute; all timing rows use budget 298 and native-camera
arrival timestamps. Tracking schema has the six new fields and unpropagated
sources, consistent with the updated pipeline. Neither exact Git commit nor
model identity/hash is recorded; they cannot be independently certified here.

Use within-session counter differences / elapsed-time differences **after 15 s
warmup**, never average cumulative FPS fields. Main sessions provide about
109 s (OT3), 109 s (OT5), 135 s (OT10) and 82 s (Control) after warmup.
Short restart sessions are retained in JSON but excluded from these comparisons.
Source percentages count **track observations**, not objects, accuracy, recall,
time-weighted occupancy or manually verified tracking success.

| Mode | Processed camera FPS | YOLO results/s | Mean inference wall ms | Unpropagated track observations |
|---|---:|---:|---:|---:|
| Control, detector only | 22.37 | 8.20 | 108.2 | n/a |
| OT3 | 19.67 | 6.52 | 126.7 | 98.61% |
| OT5 | 20.43 | 6.40 | 125.8 | 99.75% |
| OT10 | 21.40 | 5.27 | 126.4 | 87.14% |

YOLO rates in the table use tracking-CSV completion counter deltas. Timing-CSV
rates independently agree within .004 results/s; endpoints differ slightly and
the OT10/Control tracking logs extend beyond timing logs. All eight files have
the expected column counts and no malformed or duplicated rows.

## Positives

- Native camera callbacks are approximately 30 FPS in all four modes. Recorded
  post-warmup one-second camera-processing window minima are 15.57, 17.55, 18.81
  and 18.90 FPS for OT3/5/10/Control; no recorded window is below 10 FPS.
- Crop preparation counts equal detector submission counts over these windows,
  rather than every camera image: the admission optimization is operating.
- Frames with active boxes are 68.77%, 86.14% and 69.61% for OT3/5/10. Old October
  9 tracking logs had almost no active boxes. This is visibility recovery, not
  proof that the boxes follow the object or that recall improved.
- Historical logs show lower processing/semantic rates, but models, scenes,
  workload and conditions are not fully matched. Improvements cannot be assigned
  exclusively to this patch. Old processed-camera rates already exceeded 10;
  1–2 FPS/old detector rates must not be confused with capture/processing FPS.

## Remaining failures

- Only 1.39%, .25% and 12.86% of OT3/5/10 track observations are `TRACK` or
  `YOLO+TRACK`; these source flags are not independent ground-truth validation.
  Most boxes are frozen detector-frame geometry, not successful propagation.
- Tracker observation costs about **15.3–15.7 ms per processed image**. Mean
  frame work is 21.4–22.4 ms with tracking versus 6.62 ms for Control. Observation
  includes RGB-to-gray sampling, scene-jump tests and flow; it is not pure motion
  matching time. Asynchronous completion and GPU work are not fully captured by
  FrameWork_ms, so these values cannot be inverted into actual FPS.
- 71.55%, 91.43%, and 86.54% of new OT3/5/10 detector submissions are forced
  refreshes. YOLO runs on 33.16%, 31.32%, and 24.63% of processed frames,
  effectively every ~3.0, 3.2 and 4.1 frames—not reliably every 3/5/10.
  Unpropagated/lost tracks request refresh, undermining detector duty reduction.
- Callback overwrite rates are 34.43%, 31.93%, 28.68% and 25.44% for OT3/5/10/
  Control. Latest-frame dropping bounds queues, but the full 30-FPS stream is
  not processed. Occasional completion gaps >100 ms remain: 4, 3, 0 and 1;
  OT5 maximum gap is 179.67 ms. Passing window-average FPS does not imply a
  hard <=100-ms bound on every camera update.
- Inference wall time increases with tracking; readback remains approximately
  114 ms versus 96 ms for Control. This field includes unfinished compute and
  completion delivery, not pure GPU transfer. No evidence here isolates GPU
  occupancy, transfer bandwidth or the exact failed matcher condition.

## Decision and next step

Keep the updated **nano detector-only pipeline as the practical baseline**.
Do not call this tracker a successful smoother/faster pipeline. OT10 has the
largest share of propagated observations, but is still mostly stale; differing
scene contents prevent declaring it a robust optimal setting.

The next targeted diagnostic is to instrument **why Flow rejects boxes** and
separate grayscale conversion, scene-jump checking, live flow and replay time.
Use the same textured object with a stationary camera, then controlled slow
movement, then occlusion; a short recording is needed to check actual geometry.
The current translation-only patch matcher has a six-gray-pixel search radius
and strict texture/uniqueness/forward-backward/consensus guards. These are
plausible failure causes, not established from the CSVs. Refresh-interval changes
alone cannot repair a matcher that is rejecting nearly every propagation.

No runtime changes were made during this analysis. Derived JSON contains source
hashes, per-session exclusions, exact source counts, deltas and distributions.
Reproduce with `experiments/runtime/analyze_headset_tracking_acceptance.py`,
the eight named October 13 CSVs, and `--warmup-seconds 15`.
