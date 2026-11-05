# Integrated candidate and next device test

## Objective and evidence

Keep the responsive nano detector-only configuration available while determining whether
reliable tracking can reduce detector use enough to justify a later small-model trial.
This candidate combines dev `1cd4486` with feedback `428c78b`; neither input branch is
discarded. Its changes are local correctness fixes and one optional tracking experiment.

The November 4 tests measured nano at 9.92 fresh results/s and 21.98 processed camera
frames/s with debugging disabled. Small produced 6.32 fresh results/s and 17.00 camera
frames/s. Turning Development Build off added only 0.30% in that comparison. Tracking
produced 9.08 fresh results/s and 22.31 camera frames/s, with 87.25% of track observations
using unpropagated fallback and 91.17% of submissions forced. Tracker update averaged
1.63 ms; delayed replay succeeded on 19% of attempted transitions. These are descriptive
single-run comparisons, not recognition-accuracy measurements or isolated causal estimates.

## Implementation decisions

1. Integrate dev's label work and legacy asset moves. Correct its world-label coordinate
   conversion to `savedCropY + roiHeight - detectorY`, matching the bottom-up raster and
   tracker coordinates. The earlier `frameHeight - (detectorY + cropY)` is displaced
   when crop Y is off-centre; validate crop Y values 0, 30 and 80 with synthetic camera rays.
   Reconcile
   label sizing once: use bounded text sizing relative to the prefab; remove the additional
   whole-marker dimension scaling so tiny objects do not shrink text twice. Position/depth
   must be checked on device before release.
2. Include immediate world-marker/canvas-label cleanup on session stop/start, rejection of
   old results, discard events, and source-linked output application timestamps. Measure
   tracking application after its overlay draw and texture upload, not at detector completion.
   Distinguish propagated, fallback, rejected, empty, and not-drawn outcomes. Application
   timestamps are not physical headset display or speech timestamps.
3. Default general expensive debugging off. Separate tracker diagnostics so feature
   rejection can be investigated without verbose gaze/controller/timing UI work. Keep
   ordinary CSV recording enabled. Record both switches and all tracker parameters.
4. Keep fixed-grid tracking as the baseline. Add an opt-in bounded search for textured
   points within object cells for both live flow and delayed replay. Keep patch matching,
   forward/backward checks, consensus, object boundaries, semantics and expiry guards.
   Flat objects should still fail safely. Do not increase replay limits or relax validation
   simply to increase visible box counts.
5. Cache frame dimensions during grayscale extraction and log forced refresh causes.
   These are small supporting changes; the experiment must still measure total cost.

## Local validation before handoff

Run the existing configuration, raster, core tracking and integration suites, plus new
sparse-texture, delayed-replay, flat-object, object-boundary and output-timing regressions.
Run real Unity 6000.3.11f1 Editor validation on Apple Silicon. Confirm world rays agree
with the saved crop geometry and source-frame metadata remains attached through delayed outputs.
Document which checks use managed doubles and which use actual Unity APIs. Local tests
cannot establish headset speed, physical readability, real-world drift or camera/gaze accuracy.

## Device runs

Use one candidate commit, nano, GPUCompute, the same camera settings and frozen class
thresholds. Keep general expensive diagnostics and tracker diagnostics off for A/B/C.
Keep the Development Build setting identical across A/B/C; another Development Build
comparison is not the priority. Record the exact commit and build settings.

| Run | Optical tracking | Texture-aware points | Purpose |
|---|---|---|---|
| A | Off | Off | Verify integrated detector-only speed, cleanup and output timing |
| B | On | Off | Measure the original grid tracker on this same candidate |
| C | On | On | Isolate the new point selector against B |
| D | On | B or C | Short diagnostic repeat with tracker diagnostics on only |

For B/C use refresh 5, maximum 3 tracks, reassociation IoU 0.30, 3 failed refreshes,
track age 1500 ms, safety refresh 1000 ms, fallback age 500 ms and replay cap 4. Confirm
the logged values rather than relying on a screenshot. These are experimental settings.

Warm up one minute, then record approximately three uninterrupted minutes per A/B/C run.
Use the same sequence: static textured object, slow camera pan, small or low-texture object,
brief occlusion, target exit/reentry, and gaze switching between two targets. Include the
invalid-gaze case and a late result arriving after a head turn. Check that neither produces
misleading feedback for the previous target. Source frame IDs improve traceability, but
this candidate does not yet log synchronized gaze/pose samples or implement a new gaze
relevance policy; failures here remain release blockers to resolve with JHU's feedback rules.
Include the
priority JHU task classes where practical. Record a synchronized screen/video reference
with consent under existing study arrangements so box motion can be judged. At minimum,
note the timed scene sequence and visible failures; CSV alone cannot prove localization.
Repeat B/C in reversed order if C appears beneficial before choosing it for release.

Collect timing, tracking and any discard-event CSVs. Run D only to explain specific
failures; do not mix its measurements into the ordinary performance comparison. Preserve
original raw files and identify model, device, build, diagnostics and scene for every run.

## Metrics and decision

Compare fresh detector results/s separately from processed camera FPS and overlay updates.
Report median/p95 callback-to-application latency, actual applied source frame, output
status, fallback/current geometry fractions and ages, live/replay transition successes,
forced refresh causes, lost/reacquired IDs, tracker cost and sustained frame gaps. Do not
count stale fallback as successful tracking or CPU completion as user feedback.

The existing camera objective remains at least 10 processed frames/s. No new clinical
latency or acceptable false-feedback budget is assumed agreed. For A, investigate a
repeatable loss from the prior nano baseline rather than accepting it as tracking benefit.
For C, require visibly correct motion plus less failed replay/fallback/ID churn than B,
without worsening false persistence or unacceptable latency/camera cost. An algorithm
success counter alone is insufficient. Compare total detector-plus-tracker work, not just
fewer YOLO calls. Log and explain any tradeoff before selecting a configuration.

If C passes those checks, repeat with small using its own unchanged thresholds, and
compare small-only against small-plus-tracking. If B/C remain unreliable, ship the tested
nano detector-only fallback, document translation-only limitations, and investigate
specific failed recordings before trying another tracker or increasing complexity.

## Risks and assumptions

There are no supplied labeled RGB recordings for replaying the November 4 failures.
Synthetic sparse-texture improvements may not transfer to motion blur, rotation, scale,
occlusion or flat targets. More candidate searches can cost CPU. General debug state in
the old logs was email-reported, and Test5 tracker diagnostics were enabled. Scene/order,
thermal state and single-run variability can confound comparisons. Freeze thresholds;
do not tune on the frozen recognition test set. No model retraining is part of this change.
The current world placement uses later camera intrinsics/head pose rather than a fully
frozen input-frame placement snapshot; device tests must check motion alignment. This
iteration does not certify gaze/pose placement, speech suppression, physical display
timing or wall-clock expiry of already rasterized pixels during a camera stall. Preserve
these as explicit follow-up acceptance items rather than marking them fixed by FPS results.
