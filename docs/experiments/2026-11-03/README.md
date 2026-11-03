# November 3 local validation and next device tests

Execution of the available local portions of the November plan, not a new headset benchmark. The isolated source snapshot is pinned to JHU commit `865870369b5424e206d8829683de2fba9ee217c3`. No Unity branch, model, threshold or tracking algorithm was changed.

## Completed

- Both local ONNX binaries exactly match JHU's Git blobs and previously recorded SHA256 values. They pass ONNX structural checks, embedded45-name/profile alignment, float32 `[1,3,640,640]` input and `[1,40,6]` output checks, and actual ONNX Runtime CPU inference on a zero tensor.
- Preserved the distinct confidence profiles and canonical ordering. Raw model class20 is measuring spoon and maps to canonical GALVAT44. Small's older profile lacks export metadata, so the separate release manifest pins its model/configuration without altering thresholds.
- Generated reference zero-input outputs and135 threshold boundary cases per model. These are decoder fixtures, not recognition/recall tests. Threshold boundaries use Unity-compatible float32 precision.
- Existing C# checks passed:689 configuration assertions,33 tracker cases,28 managed integration cases and856339 raster assertions. Managed stubs do not certify Unity/Android/ML2 behavior. Existing34 Python camera tests and16 new Python validation tests also passed.
- Added a69-column tracking/27-column timing intake checker. Older logs produced the expected results:800-row control timing passed;2176-row tracking was identified as legacy35; the371-row coworker file was additionally identified as truncated. No new November CSVs were supplied.

See [package and decoder fixtures](package_and_candidate_check.json), [older CSV intake results](legacy_csv_audit.json), and repository-root `config/trial9/` and `config/runtime_csv_schema.json`. Header completeness is not full feedback-measurement completeness.

## Screenshot and class map

The supplied image is **Radeon GPU Profiler**, not Unity CPU Profiler. Its selected frame reports12.69006ms,78.8Hz and66.19%GPUidle. Matching RGP metadata spans only about59ms. These short rendering/submission intervals do not establish78.8YOLO results/s, sustained camera FPS, detector latency, or the cause of CPU delays. The screenshot and binary have matching capture names/frame range, but actual CPU work still requires Unity profiling.

The candidate map has43 classes, all shared by name with Trial9. It omits exit_sign and measuring_spoon and conflicts with the earlier11-class E012 description. Candidate rawID20 means pan; Trial9 rawID20 means measuring_spoon. The reference report includes a name-based crosswalk, explicitly marked **model identity unverified**. An exact model and its preprocessing/threshold contract are still needed; this map must not be wired blindly to either model.

## Source findings requiring targeted tests

1. `CameraSetup.SpawnCubes` returns on an invalid pending session before destroying existing cubes. `SetTrackingSession` invalidates the pending result and can request clearing. Already displayed cubes can therefore survive that cleanup request until another accepted result. Reproduce stop/restart and baseline-to-tracking transitions before a targeted fix.
2. World projection uses later camera extras/current head pose rather than frozen input-frame projection metadata. Baseline and tracker paths also use different Y conversions. Test an asymmetric top/bottom input and delayed head turn before selecting the correct Unity texture/pixel convention. Session checks alone do not establish current-ROI relevance; camera-driven expiry also does not enforce wall-clock expiry during a capture stall.
3. The shared grayscale/scene/history work runs synchronously while YOLO is pending. Repeated texture-dimension access is an optimization candidate, not a measured explanation for the earlier93ms cost. Use the existing stage diagnostics to locate the expensive operation.
4. Label font size remains fixed in the inspected drawing paths; bounding-box-area sizing is still planned, not implemented in this commit.

## Next device runs

Repeat nano-only and small-only on the same ML2/build/scene/backend/camera settings with profilers disconnected. Use15s warmup and at least60s measurement per run, with repeated runs and a separate longer stability sequence. Send both CSVs and exact commit/model/configuration identity, build type and power/thermal context.

Capture Unity CPU Timeline separately while detections are actively completing, initially without Deep Profiling or simultaneous RGP. Include main/render threads, camera/inference work and any Camera.Pause parent/children/self time. Correlate CPU capture, CSV and RGP by timestamp/frame IDs. A Radeon screenshot is not this capture.

Then compare tracking3/5/10 on the same static/slow-pan/occlusion sequence, verifying actual geometry and duty reduction. Include stale-cube transitions, gaze switching, target exit/reentry, invalid gaze and missing intrinsics. The present schemas still lack actual feedback timestamps, gaze/pose provenance, spoken-feedback timing and discarded-result events, so full capture-to-feedback acceptance remains unmeasured.

E012 comparison remains pending its authenticated model/package. Dataset MCP currently lists older manifests rather than a verified current45-class frozen-dataset record; these runtime checks do not establish dataset quality or split independence.

## Reproduction

Use an environment with numpy, onnx and onnxruntime; tested versions are recorded in the reference JSON. No new dependencies were installed for this run.

```sh
python tools/verify_trial9_package.py --models-dir /path/to/trial9/onnx/files --candidate-map config/comparison/candidate_43_class_map.json --output new_package_check.json
python -m unittest discover -s tests -p 'test_*.py'
python tools/audit_runtime_csv.py /path/to/timing.csv /path/to/tracking.csv --output new_csv_audit.json
```

Reports never overwrite existing outputs or input CSVs. Legacy/truncated failures are expected intake evidence, not new headset results. The root webcam demo and its requirements/legacy weights remain unchanged.
