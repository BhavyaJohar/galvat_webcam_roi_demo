# Trial 9: actual Unity/Sentis Mac benchmark

## Main finding

**Headset-scale latency can be reproduced on the Mac without the Magic Leap camera or app.** In the same native Unity player, small-model input-to-CPU-result latency averages **27.27 ms** with desktop target `-1`, **62.31 ms** with target60, and **186.43 ms** with target20. The supplied headset small-model inference window averages about192.35 ms. Endpoints/workloads differ, so numerical proximity is a clue, not proof of the headset root cause.

The leading next investigation is **actual player-loop/XR cadence and asynchronous completion**, followed by CPU/GPU frame contention. CPU cropping alone is not an adequate explanation. However, this experiment cannot separate raw GPU kernel execution from graphics queues and readback completion delivery.

Important correction to the previous source-audit hypothesis: each imported model has **297 Sentis layers**; budget298 completes scheduling in the same Unity frame for every measured inference. Increasing the default budget is not supported as a fix by these Mac results. Confirm the headset's imported count before extrapolating.

## What was built and tested

- Run: `galvat_trial9_sentis_mac_20261013_v1`; continuation of E3 in MCP-backed plan `galvat_trial9_latency_audit_v1`.
- Baseline context: `galvat_trial9_latency_audit_mac_20261012_v1` and `bhavya` commit `d6069b84cacc4c64fbe9bb5c44bb47551fc97b17`.
- Actual Unity editor **6000.3.11f1**, Sentis **2.6.1**, Burst **1.8.28**, Collections **2.6.2**, Mathematics **1.3.3**. Resolved relevant dependency versions match the branch.
- Non-development Mono standalone player, **thin ARM64 executable**, Metal, Apple M3 Pro, 640×480 window. Process manifests independently report Arm64. Rosetta2 was installed with permission to satisfy the editor prerequisite; the timing player is native ARM64.
- Fixed640×640 RGBA stimulus RGB(64,128,192); no camera access or private images. Setup validates all NCHW tensor elements and constructs matching direct input. Output must be finite `[1,40,6]`.
- Neutral built-in-renderer scene with one black, culling-mask-zero camera; no GALVAT overlays, SDK camera/gaze, XR, tracking, meshing, speech, or world markers.
- Controlled linear texture in a **Gamma** project. The existing app uses Linear color space. This difference is recorded; these are isolated runtime tests, not an exact reproduction of the app's color/rendering configuration. Sentis2.6.1 includes explicit sRGB compensation in TextureConverter; real app spatial/color parity still needs a dedicated device check.
- Exact original nano/small ONNX assets, SHA256 verified against the branch and workbench. No model re-export, training, mapping, threshold, or application-scene changes.

Eight complete suites cross CPU/GPU, async/polling, and desktop pacing configurations. Each suite runs two models × texture/direct input × full/sliced298 scheduling × three rotated-order repeats. Ten warmups plus40 measured inferences per case: **192 case runs, 7,680 measured inferences and1,920 warmups**. A separate earlier24-sample smoke run used a universal/fullscreen build and is excluded from all principal comparisons.

All eight suites completed with exit0, matching requested/written counts and input parity. Outer runner completion and all selected suite statuses were checked separately: analyzer PASS alone cannot detect a protocol's never-started next suite. All192 cases have stable per-row checksums; 4,800 same-backend representative-output comparisons are exact and4,320 cross-backend comparisons pass tolerance (maximum absolute difference0.000427246). This is representative, not all-sample elementwise, output evidence. Full-case throughput includes warmups and the intentionally identical one-frame pause between inferences. It is not camera FPS, displayed FPS, or `1000 / mean latency`.

## Results

The following uses **texture input + sliced298** to resemble the branch's detector entry point. Each row has120 measured samples over three repeats. Times are input-conversion start through CPU output availability and raw scan, not pure GPU compute.

| Backend / readback / desktop target | Nano mean / median / p95 ms | Small mean / median / p95 ms | Nano / small full-case results per second |
|---|---:|---:|---:|
| GPU async / -1 | 14.18 / 14.12 / 15.85 | 27.27 / 27.11 / 29.73 | 67.67 / 36.37 |
| GPU async / 60 | 57.74 / 66.60 / 67.49 | 62.31 / 66.57 / 82.88 | 13.31 / 12.70 |
| GPU async / 20 | 131.93 / 149.89 / 152.63 | 186.43 / 199.94 / 203.28 | 5.50 / 4.21 |
| GPU polling / 60 | 58.86 / 66.70 / 69.28 | 61.97 / 66.65 / 67.20 | 13.20 / 12.53 |
| GPU polling / 20 | 131.03 / 149.97 / 152.81 | 194.68 / 200.24 / 203.67 | 5.51 / 4.11 |
| CPU async / -1 | 28.62 / 27.85 / 33.92 | 68.88 / 68.27 / 75.28 | 33.69 / 14.28 |
| CPU async / 60 | 33.56 / 33.40 / 34.84 | 66.96 / 66.72 / 67.36 | 19.89 / 11.97 |
| CPU polling / 60 | 54.29 / 33.47 / 124.27 | 67.86 / 66.74 / 72.90 | 11.39 / 11.85 |

`-1` is the desktop default/no explicit target cap, called `uncapped` by the runner's suite aliases. Observed player-loop counts are fast in this fixture; this does not imply a display or headset can physically present hundreds of frames per second.

### What improved, what did not

- Reducing the desktop frame limit has the largest observed effect. GPU async small falls from186.43 ms at20 to27.27 ms at-1, about85% lower. Nano falls from131.93 to14.18 ms, about89% lower. These are descriptive controlled-fixture changes, not guaranteed headset gains.
- At matched desktop-1 pacing, GPU is faster than CPU: nano14.18 vs28.62 ms; small27.27 vs68.88 ms. The earlier60-target CPU/GPU comparison alone would misleadingly suggest GPU provides little benefit.
- Polling does **not** consistently improve latency or throughput. Nano GPU60 changes57.74→58.86 ms; small changes62.31→61.97 ms. At20, small polling is worse on average. CPU polling has substantial late-repeat outliers. Do not replace async readback by default.
- Full scheduling versus budget298 does not provide a consistent improvement. Neither consumes an extra scheduling frame here.
- Direct input versus texture conversion likewise provides no consistent large win. At desktop-1 GPU, texture/sliced versus direct/sliced is14.18 versus14.09 ms nano,27.27 versus27.05 ms small. This excludes repeated texture Apply/full preview uploads and full camera handling; it cannot rule those out on the headset.

## Interpretation and limits

Unity documents that [asynchronous GPU readback adds frame latency](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/Rendering.AsyncGPUReadback.html). In these runs, scheduling takes zero additional Unity frames; completion latency is strongly related to player-loop cadence. This demonstrates a substantial engine/pacing contribution even on capable hardware.

The GPU worker's static policy inspection assigns **zero CPU-fallback layers and zero flush-policy layers** for both imported models. This is a real inspection of Sentis2.6.1 private worker policy, not a GPU occupancy measurement or a guarantee about every driver/kernel operation. The Mac is not silently using a CPU worker; the headset's actual backend/runtime policy still requires its own evidence.

There is meaningful repeat drift. GPU60 nano texture/sliced means are39.13,67.04,67.06 ms; GPU20 nano means are99.94,146.21,149.65 ms. Keep pooled mean, median, p95 and repeat ranges together. Sequential suite order, background activity, focus/power/thermal state and long gaps between operator approvals were not controlled. No statistical significance or optimal configuration is claimed.

Most importantly, **XR platforms ignore ordinary targetFrameRate/vSync settings; the XR SDK controls cadence**. Android-1 also does not mean desktop-style uncapped execution. Therefore “set targetFrameRate=-1 or60” is not a verified Magic Leap fix. [Unity frame-rate documentation](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/Application-targetFrameRate.html).

This fixed synthetic fixture tests execution and representative output parity, not detection accuracy, camera timing, realistic render workload, tracking stability, or assistive user experience. The input-to-result timer omits the branch's expensive box drawing/marker work. The original171/192 ms headset inference timers also use different stage boundaries. No Unity6.6 comparison, raw GPU timestamps, headset build or on-device run was performed.

## Next test for Jadin

Use the updated opt-in probe in a copy of the **existing** Magic Leap project, retaining its XR configuration and platform. Return all case CSVs/JSONs, actual runtime layer count, backend/graphics API and CPU/GPU frame timing captures.

1. Compare the isolated probe with the app's normal detector-only path on the same headset and backend. Keep nano/small model identity and optimized thresholds unchanged.
2. Record **actual Unity player-loop cadence and XR CPU/GPU frame budgets**, not only configured FPS. Find whether main-thread work, GPU execution, compositor waits or graphics queues limit cadence.
3. Compare explicit CPU/GPU using matched input and pacing. Mac backend ranking is not a prediction of AMD headset performance.
4. If imported count is297 and scheduling gaps arezero, stop treating budget298 as the main bottleneck. If the headset count differs, compare full and sliced scheduling explicitly.
5. If isolated execution is fast but the app is slow, independently disable per-pixel box drawing/labels, world-marker creation/raycast/logging, and redundant preview/input uploads. Then optimize the measured source of frame-budget pressure. Preserve the frozen input/ROI, stable class IDs, per-model thresholds and semantic authority.

Do not enable SAM, change threshold values, blindly force synchronous readback, migrate the working project to6.6, or switch production backend on the basis of this Mac-only test. A verified device fix remains pending, but we now have a much more specific explanation to test.

## Reproduce and handoff

Reusable runner: `../runtime/run_unity_latency_benchmark.py`. It stages a new isolated project, pins dependencies, validates model hashes, builds ARM64, runs suites sequentially, preserves partial results, and terminates only its own child on timeout. Analyzer: `../runtime/analyze_unity_latency_probe.py`.

```bash
.venv/bin/python domains/galvat/experiments/runtime/run_unity_latency_benchmark.py \
  --unity-repo /path/to/GALVAT_R61 --suite protocol

# Reuse the saved local player for a targeted repeat; output root must be NEW.
.venv/bin/python domains/galvat/experiments/runtime/run_unity_latency_benchmark.py \
  --existing-player domains/galvat/experiments/runtime/sentis_mac_20261013T060845Z/builds/Trial9SentisLatency.app \
  --backend gpu --readback async --target-fps 20
```

Saved build: `../runtime/sentis_mac_20261013T060845Z/builds/Trial9SentisLatency.app`. Full build inputs, package lock, scene/settings snapshot, build logs, graph policy and complete app-payload file hashes are retained in the same result root. License preambles were removed from retained Editor logs; original temporary logs were preserved.

The reusable diagnostic source and updated handoff remain local; **no remote branch push or active application change** has been made. Actual Mac build/runtime validation supersedes the old stub-only status for this version; Magic Leap validation remains outstanding.

Verification: 99 Python tests pass (47 runner/analyzer,34 camera regressions,18 ORT-workflow tests), plus56 CLI policy assertions. Version2 patch application/source parity checks pass. Full app-payload tree digest: `3ca817a9b7db467d3bf3321e0022d4f1b61cbe91fb9d3fb486eb9159c857a05d` over165 file/symlink records. These checks validate this diagnostic, not the production headset app.
