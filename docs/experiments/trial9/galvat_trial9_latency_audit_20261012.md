# Trial 9 latency audit: Mac workflow versus Magic Leap

## Decision

Do not add SAM or tune confidence thresholds to solve this bottleneck. The Mac reproduction of the camera pixel operations costs about 1.5–1.7 ms, not hundreds of milliseconds. The remaining leading suspects are Sentis/device execution, camera-gated scheduling, graphics contention, and output/frame synchronization. A hardware-utilization problem is plausible but **not established** by a GPUCompute backend label or the existing readback timer.

An opt-in isolated Unity/Sentis probe is prepared; the active app has not been rewritten. No Unity editor or Magic Leap is available locally, so exact runtime parity and a verified headset performance fix remain blocked on a Unity/device run. The new diagnostic files are local, not pushed to the remote branch.

## Objective and provenance

- Plan: `galvat_trial9_latency_audit_v1`, saved through experiments MCP. Rich controls: `../plans/galvat_trial9_latency_audit_details_v1.json`.
- Completed Mac run: `galvat_trial9_latency_audit_mac_20261012_v1`, saved through experiments MCP.
- Audited source: `jadinV/GALVAT_R61`, branch `bhavya`, commit `d6069b84cacc4c64fbe9bb5c44bb47551fc97b17`.
- Unity project version: **6000.3.11f1**; Inference Engine/Sentis **2.6.1**.
- Headset evidence: `trial_9/timing_results_2026-10-09_13-39-51_T9N.csv` and `...13-47-54_T9S.csv`.
- Mac artifacts: `../runtime/latency_audit_20261012T222816_752398Z/{samples.csv,summary.json,manifest.json}` plus raw ORT profiles and profile summaries.
- Both model hashes match the Unity assets byte for byte. Nano: `372eedae13afa798197998d39fff592f76323de23f30ca754cb98dbcdc73782b`; small: `992a329515ee0793e7ec0f78fc0576d0d937d38ec194bfee23b4a575c10cd498`.
- Both retain 640 input, 40 end-to-end output rows, all 45 classes, and their own optimized thresholds; no extra NMS or retraining.

## What was run

E0: source/timer audit and within-session analysis of existing headset logs.

E1: fixed seeded 1280×720 RGBA stimulus on the Mac. Compare cached identical ROI tensor, Unity-style vertical flip/full-frame copy/640×640 row crop/RGB float preparation, and previous full-frame letterbox. Cross nano/small with ORT CPU/Core ML. Three rotating-order repeats, eight warmups per session, 40 measured samples per repeat/cell. Sessions are reused; setup, warmup, recording and profiling are excluded. Four primary sessions are retained, but inference is serial.

E2: separate cached-input CPU thread sweep (automatic/one/four), then separate three-inference ORT diagnostic profiles after the latency suite. The thread order is fixed, not counterbalanced; it is diagnostic rather than a rigorous thermal/thread optimum search.

There are **2,160 measured rows**, 120 per cell across 18 cells. All IDs are unique, all stage values finite, and each exclusive stage sum equals total exactly. Initial sandbox Core ML compilation failed; its failed manifest is preserved in the earlier `latency_audit_20261012T204727_705309Z` directory. The successful run required permission to use Apple's model-compilation temporary directory.

This reproduces the **CPU pixel algorithm**, not Unity native texture uploads, actual camera callbacks, Sentis graphics execution, XR/compositor, eye tracking, physics, or device thermal conditions. Source/tensor origin parity is explicitly unverified. ROI and letterbox see different content; no accuracy comparison is made. No camera/display FPS was measured.

## Observed Mac results

Mean milliseconds, Unity-style ROI path; medians and p95 are provided to expose skew:

| Model / ORT backend | Pixel prep mean | Inference mean / median / p95 | Total mean / median / p95 |
|---|---:|---:|---:|
| Nano CPU | 1.48 | 18.84 / 18.52 / 20.54 | 20.87 / 20.58 / 22.88 |
| Small CPU | 1.67 | 65.06 / 60.65 / 83.64 | 67.35 / 62.72 / 86.92 |
| Nano Core ML + CPU nodes | 1.72 | 8.81 / 7.95 / 13.98 | 11.06 / 9.84 / 18.08 |
| Small Core ML + CPU nodes | 1.61 | 14.37 / 13.63 / 16.87 | 16.48 / 15.57 / 19.66 |

Totals include filtering and optional CPU draw/copy surrogate. **Every synthetic sample yielded zero accepted boxes**. The draw/copy stage only measures ROI decoration and image copies here, not per-object drawing. It cannot validate the cost of Unity's GetPixel/SetPixel implementation or world markers.

Cached input means: nano CPU 20.40 ms, small CPU 67.89 ms; nano Core ML 8.22 ms, small Core ML 14.44 ms. Do not subtract pooled totals and call the difference exclusive preprocessing: inference itself drifts across blocks. For example, small CPU ROI repeat means are 72.08, 63.22, 59.88 ms. Warmup/startup/background effects remain despite rotation.

CPU cached thread sweep inference means:

| Threads | Nano | Small |
|---|---:|---:|
| Automatic | 18.76 | 59.05 |
| One | 72.66 | 234.72 |
| Four | 21.18 | 67.90 |

This shows CPU configuration can matter greatly on this Mac; it does **not** establish the headset's thread count or justify copying ORT thread settings into Sentis/Burst.

Core ML capability reporting accepted 361/384 original nodes in nine partitions for both models. Separate ORT traces show Core ML fused partitions **and CPU MatMul, TopK, Concat, GatherElements and other nodes**. A registered provider is not an all-GPU/ANE guarantee. These are ORT host events, not Apple hardware occupancy or Sentis placement measurements.

## What the headset logs and code establish

| Metric | Nano T9N | Small T9S |
|---|---:|---:|
| Inference-window mean | 170.93 ms | 192.35 ms |
| Readback-wait mean | 154.26 ms | 179.26 ms |
| Scheduling-window mean | 16.17 ms | 12.57 ms |
| Callback-to-result mean | 198.66 ms | 209.06 ms |
| Callback-to-result p95 | 241.95 ms | 233.70 ms |
| Active-session result throughput | 4.96/s | 4.73/s |
| Postprocessing mean | 10.55 ms | 8.22 ms |

Nano saves about **11.1%** in the recorded inference window but only **4.9%** in result throughput; tail latency is worse in this unmatched pair. These are live sessions with different scenes/detection counts and model-specific thresholds, not a controlled accuracy or causal model-size experiment.

Crucially, excluding session boundaries, logged result-to-result Unity frame gaps are four in **623/626 small pairs (99.5%)** and **531/595 nano pairs (89.2%)**. `UnityFrame` is sampled when showTimes logs/display work, not at the semantic completion callback. Large 568 ms/721 ms stalls can still span four such frame increments. This is a frame-quantization clue, **not proof of four artificial waits or a fixed 200 ms cap**.

On every row, `Inference_ms - Scheduling_ms - GPUReadback_ms` is 0 or 1 ms. The timers are nested; summing them double-counts. `GPUReadback_ms` starts after commands are enqueued and waits for a CPU clone, therefore includes model execution backlog, graphics/readback and completion delivery. The output payload is only 240 floats (~960 bytes), so attributing the entire wait to transferring that payload is unwarranted. Official [async readback documentation](https://docs.unity.cn/Packages/com.unity.ai.inference@2.6/manual/read-output-async.html) describes the required completion step.

Official 2.6.1 package source (`Tensor.cs` → `ComputeTensorData.DownloadAsync` → `AsyncGPUReadback.RequestAsync`) does not contain a mandatory chain of four NextFrame waits for this compute path. Device/player-loop measurement is needed to explain the observed gaps. Polling is an experiment, not a presumed faster fix.

## Ranked issues, evidence and discriminating experiments

| Issue | Evidence / uncertainty | Test and candidate remedy |
|---|---|---|
| Camera-gated scheduling | CameraSetup.LateUpdate processes only new camera callbacks; ProcessModelInput is called there. ScheduleIterable continues only on those calls. Actual optimized Sentis layer count is not in the logs. | Compare isolated full Schedule vs budget298/100; log actual layer count/frame IDs. If slicing dominates, advance pending scheduling from the Unity loop independently of camera arrival or select a measured larger budget. Admit new input only when available; keep one in-flight worker and frozen ROI. |
| Scheduling configuration / lifecycle | K has no positive validation; the integer startup frames estimate omits iterator termination. A pending callback can outlive worker disposal on destruction; exceptions outside GetResult are incompletely covered. These are source risks, not established steady-state stall causes. | Validate and freeze K per inference, count actual MoveNext calls/exhaustion, and explicitly drain/guard worker teardown. Preserve model input/ROI ownership. |
| GPU/model/graphics or readback wait | ~90% of the inference-window time is after enqueueing. Four-frame output cadence is strong. Existing timer cannot partition the wait. | Same probe on Mac player and ML2: full/sliced, texture/direct, async/polling. Capture Unity GPU/CPU Timeline and Developer HUD. If polling improves completion timing reproducibly, change only readback policy; otherwise do not replace async blindly. |
| Graph dispatch / runtime optimization | Both ONNX exports have exactly 384 nodes and the same operator histogram, while channels/weights differ; small opset20 vs nano12 is another difference. Raw count is not Sentis optimized count. | Record runtime layer/operator counts. Profile actual slow layers and CPU/GPU transitions. Supported-op tables alone do not prove placement; GPUCompute is a compute-shader backend, not a claim of NN accelerator use. |
| Repeated uploads | Cropped texture Apply and full preview texture Apply run on each processed camera frame, even while inference is pending; drawing can add another full upload. | Texture/direct/upload opt-in isolated comparisons, then app upload/preview ablation. If significant, upload/crop only when consuming a new detector input and render preview/boxes separately. Preserve camera observation for tracking if later enabled. |
| Per-detection native calls and marker churn | BoundingBox performs thousands of GetPixel/SetPixel calls. Text labels and world markers are recreated; SpawnCubes performs raycasts/material accesses and several logs per object. Postprocessing correlates with detection count, unlike readback. | Fixed scene and box-count tests, independently disable 2D boxes/labels and 3D markers/logging. Candidate: direct Color32 buffer drawing or separate GPU overlay; pool labels/markers; reuse materials; rate-limit logs. Measure first and validate clipping/color/ID parity. |
| Scene/XR/spatial-mesh contention | The active scene includes headset services and rendering that ORT tests omit. Android quality is Balanced, HDR enabled, MSAA1; do not blame an inactive MSAA4 profile. | Empty diagnostic scene vs existing app with identical backend, then one app feature off at a time. Record actual quality/API/device/CPU/GPU frame times. Keep XR settings labeled, not silently changed. |
| CPU fallback / transfers / resources | Worker CPU fallback is reported at construction; per-op fallback may still occur. Mac CPU threading experiment shows configuration sensitivity, not headset misconfiguration. | Explicit separate CPU/GPU suites; profiler operator placement and upload/download markers. Check Burst/jobs and development-build overhead only with evidence. No automatic backend switch based on labels. |
| Thermal, power and debugger effects | Not measured in supplied logs. Removing five initial results per session changes the original means little. | Repeat counterbalanced runs cool and after five minutes; same power state, OS/build, HUD temperatures/clocks when available, profiler detached for timing. Compare release vs development separately. |
| Geometry / input correctness | ROI uses fixed1280×720 constants while SDK chooses a capability; Editor has hardcoded1920×1080 handling. Texture origin/color may differ from ORT. | Record actual plane dimensions/stride/format and preview corners; validate a spatial calibration input in Unity before camera benchmarks. This can explain missed detections, but is not established as the latency cause. |
| Measurement endpoint / UI cadence | Camera/ROI stopwatch can accumulate over multiple processed frames per result; callback timestamp is not sensor exposure time. World markers update on a later camera frame. | Use unique source IDs and exclusive per-frame/per-inference timers; distinguish camera callback, selected input, semantic result, marker display. Do not add cumulative columns to one-frame latency or call result throughput display FPS. |

Unity's [multi-frame scheduling guide](https://docs.unity.cn/Packages/com.unity.ai.inference@2.6/manual/split-inference-over-multiple-frames.html) describes the responsiveness/latency tradeoff. Its [runtime guide](https://docs.unity.cn/Packages/com.unity.ai.inference@2.6/manual/how-sentis-runs-a-model.html) explains that unsupported operators can cause CPU work and synchronization. None of these references proves that fallback occurred in these device runs.

## Next run for Jadin: minimal decisive protocol

1. Use the opt-in probe and README in the handoff patch. Unity **6000.3.11f1**, Sentis **2.6.1**, same two model assets. Empty diagnostics scene; no app camera/gaze/tracker/markers. Keep XR/build settings recorded. Run a Mac standalone player and ML2 player if possible; keep Editor runs separately labeled.
2. First select GPUCompute, both models, four default cases (texture/direct × sliced298/full), **10 warmups, 40 samples, three repeats; suite deadline zero** so it cannot silently truncate. Run async readback then polling as separate suites. Default controlled texture is linear; do a separately labeled app-style sRGB suite if needed. Input parity is verified before timing.
3. Then repeat explicitly with CPU. Do not change backend, readback and scene together in one result. Export every CSV and JSON manifest; verify requested and written sample counts. Do not report inverse timer as app FPS.
4. Send actual graphics device/type, runtime layer counts, observed frame gaps, CPU/GPU Timeline screenshots and [Magic Leap Developer HUD](https://developer-docs.magicleap.cloud/docs/guides/developer-tools/lumin-aosp-tools/developer-hud/) CPU/GPU/FPS readings. For timing, detach profiling after a diagnostic capture. No camera imagery required.
5. If isolated ML2 is materially faster than the app, do the app ablations above. If isolated ML2 remains slow but Mac Sentis is fast, profile device execution/transfers before spending effort on CPU image copies. If Mac Sentis is also slow while ORT is fast, inspect Sentis graph optimization/dispatch/readback behavior first.

Experimental change-selection gate: seek ≥15% repeated median callback-to-result reduction with no >10% p95 regression, stable output checksums/geometry, and no worse Unity rendering cadence. These are proposed gates, not established clinical/product thresholds. The domain's 100 ms latency target remains unmet by the existing device logs.

## Reproduce the Mac run

From the workbench, with a local `bhavya` checkout:

```bash
.venv/bin/python domains/galvat/experiments/runtime/benchmark_magic_leap_workflow.py \
  --unity-repo /path/to/GALVAT_R61 \
  --samples 40 --repeats 3 --warmup 8 \
  --cpu-surrogate --threads-sweep --profile

.venv/bin/python -m unittest discover -s tests -p test_galvat_ml_workflow_benchmark.py
```

The benchmark checks model/config identity and creates a fresh non-overwriting output directory. Eighteen benchmark tests and 34 existing camera tests pass. Probe Player and Editor configurations compile against API stubs with zero errors/warnings, using installed .NET 10 references (not Unity assemblies or unavailable offline .NET Standard 2.1 references). Prepared Unity code still requires a real Unity build and device validation; managed checks cannot establish graphics/runtime correctness.

## Confidence and remaining limits

High confidence: Mac pixel-preparation measurements, source control flow, matched model hashes, timer nesting, and the observed device frame-gap distribution. Moderate confidence: scheduling/graphics is a better next investigation than pure CPU cropping. Low confidence: which device GPU layer, driver/readback policy, resource contention or thermal condition is the principal cause. No verified headset optimization is claimed; exact Mac Sentis and ML2 probe execution is the next required evidence.
