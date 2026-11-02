# Coworker October 13 tracking CSV and GPU profile

## Decision

This tracking-enabled run fails the 10 processed-camera-FPS target and provides no evidence of successful optical propagation. The strongest observed bottleneck is the CPU tracker-observation path, not a stopped camera sensor. The reported `Camera.Pause` event remains an important, unverified lifecycle/profiling hypothesis; do not remove camera pause handling or change detector backends on this evidence alone.

## Inputs and method

- `domains/galvat/trial_9/tracking_experiment_2026-10-13_19-38-35_052a3621.csv`: 371 complete rows, one truncated final row excluded, one session lasting 49.144 seconds. GPUCompute, CPU_SparseOpticalFlow, refresh interval 5 throughout.
- `domains/galvat/trial_9/com.AUD.GALVAT_R61-20261013-153848.rgp`: 16,627,040 bytes; container/header and queue records inspected, not full SQTT shader decoding. The native RGP viewer is not available on this Mac.
- User-supplied screenshot and coworker's account. No Unity CPU Profiler capture or `Camera.Pause` hierarchy was supplied.
- First 15 seconds excluded for comparisons: 262 rows, 34.080-second counter-delta interval. CSV derived results and source hash are in the adjacent JSON. Track-source percentages describe observations, not detection accuracy.
- Baseline retrieved through experiments MCP: `galvat_trial9_headset_tracking_acceptance_20261013_v1`. Existing follow-up: `galvat_trial9_tracking_value_diagnostics_v1`, implementation run `galvat_trial9_tracking_value_diagnostics_20261013_v1`.

## Observed results

| Measurement | Coworker tracking-5 | Jadin tracking-5 | Jadin detector-only |
|---|---:|---:|---:|
| Processed camera FPS | 7.66 | 20.43 | 22.37 |
| YOLO completions/second | 1.85 | 6.40 | 8.20 |
| Mean tracker-observation time | 92.99 ms | 15.33 ms | 0 ms |

Cross-tester comparisons are descriptive, not controlled regressions: exact build/model identity, build type, profiling settings, scene, resolution, and power/thermal state are unknown in the new 35-column CSV.

- Native camera callbacks: **29.99/second**; **74.46% overwritten** while waiting for processing. Latest-frame overwrite is deliberate bounded backpressure, not proof of camera malfunction.
- Tracker observation: median 88.71 ms, p95 113.45 ms. This timer includes grayscale sampling, history/scene checks, and applicable tracking work; it is not pure optical-flow matching.
- Mean logged frame work: 103.43 ms. Tracker observation accounts for about **89.91% of that logged work**, not 89.91% of total CPU/GPU time.
- **244/262 post-warmup rows have no active tracks**, yet those rows average **93.07 ms** tracker-observation time. Object-count-dependent flow alone cannot explain the overhead. Common grayscale/history/scene work and instrumentation overhead are candidates.
- Tracks present on only **6.87%** of post-warmup rows. All 19 visible track observations are `YOLO_UNPROPAGATED`; no `TRACK` propagation is recorded. This is not a recall/accuracy measurement.
- YOLO submitted on 23.75% of processed frames; 45.16% of submissions were forced refreshes. Lower detector frequency here is not a successful efficiency result because frame processing and useful geometry deteriorate.
- Every logged post-warmup FPS-window value is below 10; values span 6.65–8.34. Whole-session processing is similarly 7.55 FPS, so warmup exclusion does not produce the failure.
- Largest gap between complete CSV records: **265.86 ms**, post-warmup **209.08 ms**, not 1237.05 ms. Camera callback counters increase by 5–8 across the ten largest gaps, consistent with ongoing capture during slow processing.
- Only the tracking CSV was supplied; YOLO completion frequency does **not** establish inference latency. The paired `timing_results` CSV is needed.

## GPU capture: meaning and limitations

The screenshot reports 87.49% GPU idle, a 787.31 ms profiled frame, and 1.3 Hz. This supports a GPU waiting/submission-limited interpretation in that capture, but does not identify which CPU task or wait is responsible, prove YOLO never uses the GPU, or establish normal live-camera FPS. AMD describes the CPU-bound label as an idle-threshold classification rather than a measurement of CPU saturation. The displayed 15.84 MB profiling overhead is trace-memory traffic, not a slowdown percentage. [AMD overview documentation](https://gpuopen.com/manuals/rgp_manual/overview_windows/).

Binary metadata confirms legacy SQTT container version 1.5, Vulkan API metadata 1.1, AuthenticAMD CPU brand `100-000000003`, four physical/eight logical cores, and GPU name `Unknown AMD GPU`. The chunk walk reaches exact EOF. Queue metadata has five queue records and 75 event records across frames 1343–1348 (and unassigned frame zero); those counts cover a wider interval than the screenshot's selected-frame summary, so are not contradictory.

There is a **timebase/scope discrepancy to resolve**: queue CPU timestamps, divided by their recorded 1 GHz CPU timestamp frequency, span 669.78 ms. First-submit gaps across frames 1343–1348 are 168.33, 133.06, 134.49, 116.85, and 116.81 ms. Their numeric clock range overlaps CSV callbacks around session seconds 10.0–10.7, making a common run plausible, not formally verified. This is much faster than the screenshot's reported 787 ms frame. AMD says frame-summary duration/rate use SQTT data, so verify clock metadata and CPU/GPU frame views in the Magic Leap-compatible viewer before treating its 1.3 Hz as camera throughput. A clock-conversion/viewer problem is an open possibility, not a proven cause. Container layouts checked against [AMD-authored Mesa SQTT source](https://chromium.googlesource.com/external/gitlab.freedesktop.org/mesa/mesa/+/dac0019373a2a2cb2ed8b68b066b13dd3d822ef2/src/amd/common/ac_rgp.c).

Magic Leap documents that RGP capture can freeze or destabilize heavy Unity applications, and that its viewer is Windows/Linux-only. A capture-induced stall is therefore a credible confound, not an established explanation. [Magic Leap RGP documentation](https://developer-docs.magicleap.cloud/docs/guides/developer-tools/lumin-aosp-tools/radeon-gpu-profiler/).

## Camera pause source audit

Current checkout: `/tmp/galvat-latency-ZAGmYk/GALVAT_R61`, `bhavya` commit `dd792ba28b1b0ee589986c33b0216625f616ee48`. This is inspected source, not a verified coworker build.

- `CameraSetup.cs:200–219`: stop/disconnect on cleanup/OnDestroy, not once per frame.
- SDK `MLCamera.cs:270–315`: `OnApplicationPause(true)` calls `Pause()`, invokes pause subscribers, and synchronously disconnects. Its explicit profiler marker is `MLCamera.ApplicationPause`, not a literal managed `Camera.Pause` marker.
- SDK `MLCameraBaseInternal.cs:316–350`: disconnect can stop video/preview and call native disconnect. It also has `disconnectPerfMarker.Begin()` both at entry and exit, with no matching `End()` in that method. This instrumentation defect warrants caution interpreting nested samples; it does not prove the reported stall is artificial.
- `MLProcessing.cs:285,409`: schedule advancement runs on the Unity thread and continues without new camera frames. A blocked main thread can delay further submissions/completion handling even if GPU execution is asynchronous.

Without the exact CPU Profiler parent/children, total/self time, event timestamp, and pause/focus logs, we cannot identify the coworker's marker or determine which native operation consumed 1237.05 ms. Do not bypass OS camera lifecycle handling.

## Recommended next evidence

Use the already-shipped diagnostics rather than another tracker/model change:

1. Build `bhavya` **dd792ba** or a verified descendant. Send commit, model/profile, negotiated resolution, OS/SDK/Unity version, build type, power state, and whether Deep Profiling was enabled. Its tracking log must have **69 columns** including `GrayCapture_ms`, `SceneCheck_ms`, `LiveFlow_ms`, build and model metadata; this returned log has only 35.
2. Run the existing nano-only versus tracking-5 comparison on the same device/objects with **both profilers disconnected**: 15-second warmup then at least 60 seconds measured, and send both CSVs. This separates normal workload from profiler effects. Keep model, threshold profile, refresh config, and other build settings fixed.
3. For a reproducible pause, use a separate diagnostic build/run and collect Unity CPU Profiler alone with Deep Profiling initially off. Supply the `.data` capture and Timeline/Hierarchy views showing the exact `Camera.Pause` parent/children and total/self time, plus focus/application-pause events. Repeat with RGP alone if necessary; do not combine captures first.
4. In RGP, provide CPU-based frame view, device/clock configuration, and Event timing dispatch details. Reconcile CPU submission timestamps with SQTT frame duration before inferring a normal 1.3 FPS limit.

If `GrayCapture_ms` dominates, investigate caching texture dimensions outside the pixel loop, CPU-native boundary/bounds-check cost, and lower-resolution/native-buffer grayscale capture before changing the matching algorithm. If scene checks dominate, optimize those. If actual camera lifecycle calls recur, diagnose why the application is repeatedly pausing. These are conditional engineering actions, not fixes already validated by this run.

## Confidence and changes

High confidence: slow tracker-observation path, continuing callbacks, sub-10 processing, no recorded successful propagation, old log schema. Moderate confidence: CPU submission/workload delays explain GPU idle. Low confidence: exact `Camera.Pause` cause, profiler overhead magnitude, build/model identity, or normal inference latency.

Analysis/report artifacts only; no Unity runtime, SDK package, model, thresholds, or branch changes. Data-analysis, debugging, and reporting skills guided counter-based comparison, bounded source inspection, and separation of facts from hypotheses. Reports were written directly, with the summary and artifact reference recorded through MCP.
