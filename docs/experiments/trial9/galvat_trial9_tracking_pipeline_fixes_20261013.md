# Trial 9 tracking/camera pipeline fixes — 2026-10-13

## Decision and scope

Implemented fixes on the coworker's `bhavya` branch, based on `d6069b8`, rather
than upgrading Unity or retraining. Acceptance target is **>=10 processed camera
images/s with usable tracking**, not 10 semantic detections/s. No Magic Leap
device is available locally; meeting that target on the headset remains an
explicit acceptance gate, not a measured result.

Authoritative context: experiment plan `galvat_trial9_tracking_pipeline_fix_v1`,
prior audit `galvat_trial9_latency_audit_v1`, real Mac Sentis run
`galvat_trial9_sentis_mac_20261013_v1`. Dataset accuracy is not reevaluated.

## Implemented changes

- Failed delayed replay retains young original YOLO geometry explicitly as
  `YOLO_UNPROPAGATED`, never as successful flow; source-age grace 500 ms, replay
  cap 8, forced refresh, original timestamp and same-class association preserved.
- Native RGBA callback owns a double-buffered latest image. One reusable copy
  flips rows and strips padding, avoiding this SDK's allocated managed plane and
  separate allocated flip. Borrowed native pointer never escapes callback.
  Deeper SDK inspection showed managed planes are fresh allocations despite
  cached arrays, so baseline pixel/metadata corruption is **not established**.
- Clean 640x640 crop copied/uploaded only for admitted detector inputs. Gaze
  ROI uses actual negotiated dimensions; overlays excluded from model and flow.
  Actual webcam dimensions and annotated editor preview replace old hardcoding.
- Scheduling advances without camera arrivals, once per Unity frame, while
  retaining single-inference ownership and frozen crop/frame/session metadata.
  Pending/partial teardown waits for completion; request-setup failures latch
  restart off and retain storage where GPU completion cannot be proven.
- Bounding-box raster uses clipped native Color32 writes; tracker luma buffer
  reused; noisy per-marker logging removed; output resolution follows camera.
- Main nano scene defaults to tracking interval 5. Detector-only toggle retained.
  Separate camera-processing HUD/window rate and callback/drop/crop CSV counters.

Both ONNX hashes, 45-class mapping, measuring-spoon canonical ID 44, optimized
nano/small threshold profiles, package lock and end-to-end no-extra-NMS behavior
are unchanged. Input preparation remains necessary; no claim of zero-copy/GPU
camera import or CoreML/ANE acceleration in Unity.

## Verification and interpretation

- Full coworker project imported/compiled with installed Unity 6000.3.11f1 and
  Inference Engine 2.6.1; source was compiled again after current camera/lifecycle
  edits. Opt-in real Unity checks passed for owned native-pointer copy,
  raster-vs-pixel-API byte parity, ROI clamps and clean sampling/delayed fallback.
- 28 real-core standalone regression tests passed; managed integration passed
  26 cases in editor and non-editor defines; real model config passed 689
  assertions; raster managed tests passed 856,339 assertions. Managed stubs do
  not certify Android/IL2CPP or native headset behavior.
- Synthetic .NET Release benchmark, three moving objects, grayscale 320x180:
  default replay limit 8, delay 5 gives completion-frame core-work p95 **2.885
  ms**. At delay 31, cap 31 gives **15.337 ms**, cap 8 gives **0.522 ms**, but
  the latter retains existing tracked geometry and requests semantic refresh;
  it is not equivalent detector-corrected geometry. Flat fixtures keep three
  explicitly unpropagated boxes. These exclude camera, YOLO, Unity rendering,
  native SDK and thermal costs; fixed ordering/warming confounds short-delay
  comparisons. They cannot establish real camera FPS or a headset speedup.

Confidence is high in the deterministic replay/copy/admission contracts and
source/API compilation; uncertain in real-world motion quality and physical
performance. Remaining sparse matcher is translation-only, not LK or SAM.
Unpropagated annotations can briefly describe an occluded object; source-age
expiry is evaluated on incoming images and camera stalls can freeze the preview.

## Required next device decision

Jadin should run same-scene nano detector-only and tracking intervals 3/5/10,
15 s warmup +60 s each, preserving thresholds/backend/workload. Send both timing
and tracking CSVs, exact commit, thermal/power context and a motion/occlusion
screen recording if permitted. Measure post-warmup frame-count deltas / elapsed
time and inspect one-second dips, stable IDs, correct ROI edges, empty scenes and
bounded unpropagated geometry. If processing is <10 FPS, capture CPU/GPU/render
Profiler data: remaining XR/render/meshing workload, texture uploads, replay and
completion delivery cannot be resolved from pure readback elapsed time.

Delivered and pushed commit **31332ed5f42680fe09e4f041fa15d12a3f84ae2a** to bhavya.
Final real Unity validation passed all four checks and invokes the actual locked
camera slot-consumption method, not a duplicated test swap.
Coworker handoff: `README_TRACKING_PIPELINE_FIXES.md` on bhavya. Local Android
build support is not installed, so no certified Magic Leap APK was produced.
This report was written directly; plan/run/artifact records use MCP when saved.
