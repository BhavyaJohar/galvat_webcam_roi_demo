# Trial 9 nano versus small: live Mac runtime comparison

Date: 2026-10-08 local / 2026-10-09 UTC. Run group: `20261009T000855Z`.

## Decision

The supplied nano deployment export materially reduces Mac inference cost.
CPU YOLO-only throughput increased from 12.19 to 25.68 FPS (2.11x).
Core ML already puts both exports near the ~30 FPS webcam ceiling; nano
reduces inference time but does not materially increase displayed throughput.
Nano is a strong candidate for a separate Magic Leap test, contingent on
verifying its reported recognition quality. No Unity branch/model was changed
by this Mac benchmark, and no detector was retrained or re-exported.

## What was run

Reused experiment plan `galvat_trial9_mac_camera_tracking_v1`. Hypothesis:
nano reduces detector cost, while optical propagation reduces invocation duty;
a camera-limited pipeline may hide model latency improvement in total FPS.
Contemporary small baselines were rerun instead of comparing only with yesterday.

12 cells: both exports x CPU or CoreML x YOLO-only, OpenCV CPU optical-flow
refresh5, or refresh10. Each displayed live-camera run processed 120 frames,
for 1440 total. Requested and observed camera dimensions were 1280x720;
640x640 letterbox, the same canonical45 mapping, the same optimized per-class
threshold config, camera0, max3 tracked objects, default ORT threading and
same visualization were retained. Run order alternated model precedence across
modes. There was one short run per cell, not repetitions/randomized control.

The Mac tracker is the existing OpenCV Lucas-Kanade implementation. It is NOT
SAM and NOT the Unity sparse patch-matching backend. The Python camera pipeline
is synchronous detector/propagation on processed images, with an asynchronous
latest-camera-frame slot; it is not Unity's asynchronous delayed-result replay.

The nano export is opset12 while small is opset20. This compares supplied
**deployment exports**, not an architecture-only ablation. Both ONNX files
passed structural validation and retain float input [1,3,640,640], end-to-end
float output [1,40,6], and the same embedded45 names. Thresholds were retained
for comparability; their calibration for nano has not been independently verified.

## Results

FPS is processed/display-submission throughput over active measurement time,
not monitor presentation rate. YOLO ms is average session.run time on invoked
frames; flow ms includes propagation/refresh work averaged over all processed
frames. Invocation duty is a percentage of processed frames, not GPU utilization.

| Backend / mode | Small FPS | Nano FPS | Small YOLO ms | Nano YOLO ms | Small duty | Nano duty |
|---|---:|---:|---:|---:|---:|---:|
| CPU / YOLO-only | 12.19 | 25.68 | 64.49 | 20.59 | 100.0% | 100.0% |
| CPU / flow refresh5 | 23.30 | 28.66 | 67.70 | 23.84 | 21.7% | 21.7% |
| CPU / flow refresh10 | 23.60 | 28.75 | 81.40 | 29.97 | 10.8% | 10.8% |
| CoreML+CPU / YOLO-only | 28.82 | 29.03 | 17.35 | 8.99 | 100.0% | 100.0% |
| CoreML+CPU / flow refresh5 | 29.08 | 28.80 | 22.89 | 16.19 | 20.8% | 20.8% |
| CoreML+CPU / flow refresh10 | 28.75 | 29.20 | 24.90 | 18.58 | 10.8% | 10.0% |

CPU baseline inference decreased 68.1% (64.49 to 20.59 ms), about 3.13x faster.
CoreML baseline inference decreased 48.2% (17.35 to 8.99 ms), about 1.93x faster.
Per-invocation times differ with refresh mode; the data do not establish why.
Do not assume tracker-mode inference cost equals the every-frame baseline.

Nano CPU tracking brings processing to about 28.7 FPS, versus 25.7 without
tracking. CoreML configurations all land around 28.8-29.2 FPS; tiny differences
are not robust improvement evidence. Tracking reduces semantic invocations to
~10-22% but has measured CPU overhead; compute/power savings are not established.
The source delivered approximately29.6-30.2 successful reads/s across runs.

## Track workload and frame-time tails

| Run | Flow mean ms | Frame p95 ms | Mean active objects | Frames with objects /120 | Loss events |
|---|---:|---:|---:|---:|---:|
| nano_cpu_baseline | 0.00 | 43.42 | 1.02 | 120 | 0 |
| small_cpu_baseline | 0.00 | 93.95 | 1.01 | 120 | 0 |
| small_cpu_simple5 | 4.20 | 111.06 | 0.98 | 118 | 2 |
| nano_cpu_simple5 | 4.02 | 66.21 | 0.93 | 112 | 3 |
| nano_cpu_simple10 | 4.17 | 70.75 | 0.99 | 119 | 1 |
| small_cpu_simple10 | 4.11 | 118.05 | 0.99 | 119 | 1 |
| nano_coreml_baseline | 0.00 | 36.77 | 1.02 | 119 | 0 |
| small_coreml_baseline | 0.00 | 35.97 | 0.99 | 119 | 0 |
| small_coreml_simple5 | 4.95 | 66.33 | 0.99 | 119 | 1 |
| nano_coreml_simple5 | 4.98 | 61.45 | 0.99 | 119 | 1 |
| nano_coreml_simple10 | 4.46 | 62.40 | 0.92 | 110 | 0 |
| small_coreml_simple10 | 5.32 | 69.70 | 1.24 | 119 | 2 |

Average active objects was roughly one, not a three-object stress test. Nano
CoreML refresh10 had objects on110/120 frames despite zero logged loss events;
this illustrates why loss counters alone do not establish persistent visibility
or quality. No ground truth, ID-switch score, drift rating, scene matching, or
recognition evaluation was performed. Optical-flow runs have slower refresh
frames even when average FPS is near 30; p95 frame time shows those tails.

## Integrity, backend and limitations

- Nano SHA256: `372eedae13afa798197998d39fff592f76323de23f30ca754cb98dbcdc73782b`.
- Small SHA256: `992a329515ee0793e7ec0f78fc0576d0d937d38ec194bfee23b4a575c10cd498`.
- Sizes: nano 9,839,757 bytes; small 38,236,526 bytes (~74.3% smaller file).
- CPU sessions report CPUExecutionProvider. Accelerated sessions report
  CoreMLExecutionProvider plus CPUExecutionProvider. Per-node placement and
  Apple GPU/ANE utilization were not profiled; this is not PyTorch MPS.
- Available providers: CoreMLExecutionProvider, AzureExecutionProvider,
  CPUExecutionProvider. Tracker always CPU. Full hashes/versions in each summary.
- Platform: macOS-27.0.1-arm64-arm-64bit.
- Packages: {"numpy":"2.5.3","opencv-python":"5.0.0.93","onnxruntime":"1.30.0","torch":null,"sam3":null}.
- Every run exited0 with no runtime/cleanup errors; all120-row frame sequences,
  refresh settings, invocation reasons and expected nonmissing fields checked.
  Empty refresh_reason on propagation-only frames is intentional. Config hashes
  matched across runs. Sandbox unittest suite:34 tests passed after benchmarks.
- Live scene/object mix, thermal/background workload and run order are not fully
  controlled; one120-frame run is a smoke test, not confidence intervals.
- Camera runs include initial inference; CLI --warmup applies only to synthetic
  mode. Initialization/provider compile cost is outside the FPS measurement.
  No keyboard pauses/interval changes occurred. Cleanup is excluded by Metrics.
- Similar nano accuracy is user-reported; this run neither confirms nor refutes
  it. No benchmark dataset was used and no camera images were saved.
- Mac results do not predict Sentis/Magic Leap speed or tracking quality.

## Artifacts and reproduction

Raw CSVs, JSON summaries, logs, exact commands, hashes and settings:
`domains/galvat/experiments/runtime/mac_nano_comparison_20261009T000855Z/`.
The manifest was written before execution and updated with exit codes.
Run records were saved through experiments MCP with IDs
`galvat_trial9_mac_nano_comparison_20261009T000855Z_<cell_name>`.
This report and the benchmark driver were written directly in the repository.

```sh
.venv/bin/python domains/galvat/experiments/runtime/benchmark_nano_camera.py --frames 120
```
Each invocation creates a new timestamped result directory and displays webcam
windows. No images are recorded. Longer repeats on a matched prerecorded scene
would isolate runtime differences better, but video decode/unpaced throughput
must not be confused with real live-camera FPS.

## Next decision

For Mac responsiveness, nano + CoreML YOLO-only already approaches the camera
limit and is the simpler measured configuration. Use tracking when testing
semantic duty/temporal continuity, not to promise further FPS. Repeat longer
matched scenes and manually inspect false tracks/ID switches before choosing an
interval. Test nano versus small on Magic Leap with identical settings next;
its opset/runtime differences and smaller graph may behave differently there.
