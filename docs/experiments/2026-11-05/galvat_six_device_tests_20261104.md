# Six Magic Leap tests returned November 5

## Decision

Keep nano detector-only with expensive debugging disabled as the current speed baseline.
Camera processing exceeds 10 FPS in all six runs, but fresh semantic results reach
approximately 10/s only with nano and debugging disabled. Small remains a possible
accuracy tradeoff at 6.32 fresh results/s; these logs do not evaluate recognition accuracy.
Tracking still needs reliability work before inclusion.

## What was run

Twelve CSVs in `domains/galvat/trial_9/latest_test_results/`, 1,500–1,590 timing rows per
configuration; all rows have complete column counts, with no malformed rows excluded.
Models are identified by CSV ModelId and threshold profile. All use GPUCompute, Vulkan,
Unity 6000.3.11f1, 1280×720 camera and 298 layer budget. General debugging settings come
from Jadin's email; these logs record tracker diagnostics but not the full debug settings.
Test 6 records DevelopmentBuild=0; other tests record 1.

Jadin reports a minute of warmup before detection. Primary rates use all recorded
within-session counter/time differences. Test 2 has a short second session, combined
with the first using duration weights; off-session gaps are excluded. A sensitivity
analysis excluding another 60 seconds preserves the conclusions. The warmup itself
is not verifiable from detection-only logs. Paired files have distinct logger RunIds;
source frames and clock ranges support their correspondence. Tracking CSVs continue
slightly beyond the timing recordings, so counts are not identical across paired files.

| Test | Configuration | Fresh results/s | Camera processing FPS | Mean inference ms | Mean callback-to-result ms |
|---|---|---:|---:|---:|---:|
| 1 | Nano debug on, no tracking | 7.78 | 22.11 | 111.98 | 133.93 |
| 2 | Nano debug off, no tracking | 9.92 | 21.98 | 94.48 | 110.54 |
| 3 | Small debug on, no tracking | 5.18 | 15.35 | 176.42 | 200.35 |
| 4 | Small debug off, no tracking | 6.32 | 17.00 | 150.71 | 162.18 |
| 5 | Nano debug off, tracking on | 9.08 | 22.31 | 94.20 | 108.92 |
| 6 | Nano debug off, no tracking, Development Build off | 9.95 | 21.72 | 94.02 | 110.78 |

Fresh rates come from timing counter deltas; camera rates from tracking CSV counter
deltas. Callback-to-result stops at CPU completion, not marker/display/speech feedback.
GPUReadback_ms overlaps inference and includes waiting; it is not isolated data-transfer
cost. Do not add these stage fields to estimate latency.

## What improved and what did not

Disabling debugging improved nano fresh rate by 27.63% and small by 22.01%. Nano with
debugging disabled was 57.03% faster than small under the corresponding setting.
Turning Development Build off added only 0.30% over Test 2, which is too small to establish
a meaningful benefit from one run per setting. General debug overhead is the more useful
target than the Development Build checkbox in these recordings.

Camera callbacks stayed near 30/s throughout. Camera processing stayed near 22/s for
nano and 15–17/s for small. Nano debugging-off preprocessing averaged 0.63 ms and small
0.92 ms; no evidence here supports preprocessing as the principal bottleneck. Inference
wall time and debug-associated scheduling/postprocessing costs dominate the recorded
detector path. Exact CPU/GPU utilization remains unmeasured by these CSVs.

Timing p95 callback-to-result was about 155, 116, 223, 177, 113 and 115 ms in Tests 1–6
(Test 2 first session). Occasional stalls remain: maxima of approximately 500, 479,
584, 250, 195 and 197 ms respectively. Mean FPS does not imply a strict per-frame guarantee.

## Tracking findings

Test 5 settings: refresh every 5 frames, maximum 3 tracks, 3 failed refreshes, age 1500 ms,
safety refresh 1000 ms, fallback age 500 ms, replay cap 4, tracker diagnostics enabled.
The screenshot and CSV agree on tracker diagnostics; this remains a confounder when
comparing total tracker overhead against fully diagnostics-disabled operation.

Tracking mean observation cost was 1.63 ms per processed frame, overlay work 0.18 ms,
and fresh result rate fell 8.56% relative to Test 2. Camera processing rose only 1.47%,
which does not establish a practical camera-rate benefit from one unmatched scene run.
The logged overlay update rate (~19/s) counts frames with visible tracks, including
fallback boxes; it is not the rate of successful motion propagation.

Across 4,872 displayed track observations, 4,251 (87.25%) were YOLO_UNPROPAGATED,
218 (4.47%) YOLO+TRACK and 403 (8.27%) TRACK. This is a positive change from runs with
no successful propagation, but does not demonstrate reliable current geometry.
These are repeated observations, not unique objects or ground-truth accuracy.

There were 1,456 forced refreshes out of 1,597 submission increments (91.17%), with
369 lost tracks. The configured every-five schedule is therefore being overridden.
Submissions occurred on 40.70% of processed frames rather than the nominal 20%; this is
an invocation fraction, not GPU occupancy. All 1,597 refreshes were admitted, but admission
does not mean motion replay succeeded: many refreshes entered the stale fallback path.

Final cumulative flow counters show live propagation succeeded on 621/825 attempted
track transitions (75.27%), while delayed replay succeeded on only 359/1887 attempted
detection/frame transitions (19.02%). These are conditional algorithm outcomes, not
localization accuracy. Of 1732 failed flow calls, 957 had too few eligible points, 515
too few matched points, 192 low match fraction, and 68 bad consensus. Point-level
low-texture rejection events numbered 39617. This motivates inspecting feature availability
and source/box geometry on controlled scenes; it does not prove a coordinate bug.

Replay-cap rejections were zero, and replay histories spanned at most three transitions.
Increasing the configured replay cap of four is not supported as a fix for this run.
No scene resets or age expirations were logged. Grayscale capture averaged 1.277 ms,
core update 0.346 ms and accounts for most of the 1.626 ms update cost. Tracker computation
is much cheaper than inference here; reliability and repeated refresh requests undermine
its potential benefit more than raw tracker update latency.

## Source verification and implications

Inspected exact commit `1cd44868eaef49647fef45360e21499c92ad6a0c` from the dev fetch.
Relative to JHU8658703, changes include moving legacy model assets, documentation,
world-label vertical-coordinate conversion (`captureHeight - (...)`), and scaling the
whole marker by its largest box dimension. These are more than a branch rename, though
the inspected diff does not replace the detector/tracker algorithms or current models.
Device validation is still needed for label placement. Do not overwrite this work by
replacing dev with the older JHU branch.

These tests do not include our `bhavya-feedback-timing` commit 428c78b. Its cleanup,
application timing, default-off diagnostics and bounded font sizing require integration
with dev, explicitly reconciling both label-sizing approaches and preserving Jadin's
coordinate work. No Unity runtime files were changed during this review.

## Next tests and interpretation confidence

Use nano/debugging-off/detector-only as the fallback. Integrate the small cleanup and
measurement patches with dev, verify them, and obtain output application logs. For tracking,
reproduce failures on a fixed textured scene and a low-texture scene, with slow motion,
then enable tracker diagnostics separately to inspect replay and feature rejection.
Require visibly correct propagation and stable IDs before another interval sweep or a
small-plus-tracker trial. No need to repeat the Development Build comparison first.

Confidence is high in descriptive rates and the explicit fallback problem. Causal effect
sizes remain provisional: one run per setting, different build GUIDs, no verified identical
scene trajectory, and unknown thermal/order effects. The 1,500 frames in a run are not
1,500 independent experimental repetitions. Detection counts alone cannot establish
false positives, false negatives, or relative model accuracy.

Reproduce:

`python tools/analyze_november_six_tests.py --folder /path/to/latest_test_results --output /tmp/galvat-six-tests-review.json`

Machine-readable per-session metrics, SHA256 identities and warmup sensitivity are in
`galvat_six_device_tests_20261104.json` alongside this report.
