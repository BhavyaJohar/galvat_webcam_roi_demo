# GALVAT model and runtime status

Verified 2026-11-02. This is a catch-up and repository-ownership guide, not a new benchmark.

## Two project repositories

| Repository | Authoritative responsibility |
|---|---|
| [BhavyaJohar/galvat_webcam_roi_demo](https://github.com/BhavyaJohar/galvat_webcam_roi_demo) | Training/evaluation/export notebooks; model-specific class and confidence configuration; ML optimization decisions, reproducible Mac experiments, model release manifests and recognition evidence. |
| [jadinV/GALVAT_R61](https://github.com/jadinV/GALVAT_R61/tree/bhavya) | Unity scenes/C# code, Magic Leap camera integration, inference scheduling, tracking, headset instrumentation, imported deployment models/configs and device test instructions. |

The separate `BhavyaJohar/ml-workflow` workbench generated the imported historical artifacts. It remains reusable tooling/provenance, not a third GALVAT release destination. This first import does not migrate every script, model, dataset or raw headset capture.

Keep historical artifacts immutable. New GALVAT model work belongs here; changes affecting the headset must also be implemented and verified in the Unity repository. Do not assume a change in one repo updates the other.

## What is pushed to Unity?

GitHub branch checks on 2026-11-02:

- `bhavya`: **dd792ba28b1b0ee589986c33b0216625f616ee48**.
- `main`: **3da9995eef27c6d9a12d78da8fbdab32296ab749**.
- `main` is an ancestor of `bhavya`; the latter is 17 commits ahead. Our updates are pushed to **bhavya**, not merged into **main**.
- Important commits: `0e27bae` Trial 9 mapping/model/threshold integration; `46d9b54` and `0a83ed5` timing; `1709039` optional optical tracking; `d6069b8` nano and its confidence profile; `31332ed` camera/tracking pipeline fixes; `dd792ba` detector-only default and expanded diagnostics.

These AUD changes were already delivered. JHU's newer `JHU_Bhavya_1026` branch at `865870369b5424e206d8829683de2fba9ee217c3` includes `dd792ba` and adds session-safe pending-result handling and result processing without camera intrinsics. Use that JHU branch as the next joint integration baseline, preserve its changes, and follow the [November2–20 delivery plan](plans/galvat_delivery_plan_nov2_nov20_2026.md). No branch was automatically merged or overwritten.

## Best measured headset baseline: two different FPS numbers

Jadin's October 13 detector-only logs measured **22.37 processed camera images/s**, **8.20 fresh YOLO results/s**, and **108.2 ms mean inference wall latency**, after excluding 15 seconds of warmup. This is not 22 YOLO inferences/s, a guaranteed display refresh rate, or an all-scenes minimum.

The logs are consistent with the updated pipeline but do not contain exact commit/model hashes. Consequently, do not claim that `dd792ba` itself has been independently headset-benchmarked, or that a controlled experiment established the contribution of every individual fix.

The practical changes were the smaller nano deployment model; preserving the correct 45-class map and model-specific confidence thresholds; owned, double-buffered latest-frame capture with bounded overwrite; preparing/uploading the detector crop only when a new inference can be admitted; allowing pending inference scheduling to progress without requiring another camera arrival; and reducing avoidable overlay/allocation/logging work. Grayscale preparation, resizing/cropping, channel order and normalization remain necessary—YOLO does not replace its input pipeline. The older pipeline is not established to have processed an accumulating queue.

See [headset acceptance results](experiments/trial9/galvat_trial9_headset_tracking_acceptance_20261013.md).

## What we tried

| Work | Outcome and limitation |
|---|---|
| Trial 9 small model integration | Existing 640 input, end-to-end 40-row output, canonical 45 classes including measuring spoon at ID44, and optimized class-specific confidence profiles. No redundant conventional NMS added. |
| Mac YOLO-only baseline, CPU versus ORT CoreML | Small CPU about11–12 FPS versus about28–29 with CoreML+CPU. CoreML provider use is not proof of a particular Apple GPU/Neural Engine allocation. |
| Mac OpenCV Lucas–Kanade tracking every5/10 frames | Small CPU throughput roughly23–25 FPS; reduced detector invocation frequency, but drift/occlusion/identity quality was not formally scored. |
| Nano versus small Mac comparison | CPU detector-only25.68 versus12.19 FPS; CoreML both near29 FPS. Supplied exports also differed in opset, so this was not an architecture-only ablation. Similar recognition accuracy was user-reported, not established by this runtime test. |
| SAM3.1 feasibility investigation | October investigation found CUDA/Triton-oriented paths and missing live ingest/MPS adaptation. No SAM execution/latency was measured. A Hugging Face key provides checkpoint access, not accelerator compatibility. This is historical evidence, not a fresh November compatibility audit. |
| Unity CPU sparse patch-matching tracker | Optional refresh3/5/10 tested on headset. This implementation is neither OpenCV LK nor SAM. It was slower than detector-only and mostly retained unpropagated detector boxes. |
| Matching the Unity workflow on Mac | Pixel-preparation measurements alone did not explain the device gap. Actual Unity6.3/Sentis tests showed major player-loop/pacing effects; both imports had297 layers, fitting the298-step schedule budget. No evidence that merely upgrading Unity to6.6 solves this; no equivalent6.6 A/B was run. |
| Camera and tracking-handoff fixes | Improved ownership/admission/scheduling and bounded stale-box handling. The returned detector-only run reached8.20 fresh results/s; tracking modes reached5.27–6.52 and19.67–21.40 processed-cameraFPS. No controlled attribution to each patch. |
| Later coworker profiling | Tracking5 fell to7.66 processed-cameraFPS/1.85 YOLO results/s; the tracker-observation path averaged93 ms even without objects. Camera callbacks continued near30/s. GPU-idle screenshot and reported Camera.Pause did not establish a root cause; profiler effects and timebase discrepancies remain unresolved. |
| Latest diagnostic update | `dd792ba` restores nano-only default and expands tracking logs from35 to69 columns with stage times, rejection reasons and runtime/build/model context. Local tests passed; the new controlled headset diagnostic comparison is still pending. |

Historical details and limitations are in [the experiment archive](experiments/trial9/). Mac results do not establish headset speed, power savings, recognition quality or clinical benefit.

## Next tests, in order

Follow Unity's [README_TRACKING_VALUE_DIAGNOSTICS.md](https://github.com/jadinV/GALVAT_R61/blob/bhavya/README_TRACKING_VALUE_DIAGNOSTICS.md), rather than varying many parameters at once:

1. Verify `dd792ba` or a descendant and repeat nano-only with profilers disconnected. Send exact commit/model/profile/build type, both CSVs, and power/thermal context. Use15s warmup plus at least60s measurement on a consistent scene.
2. Run small-only under the same conditions, preserving its own threshold profile. Determine whether it already meets the camera-rate target and whether recognition is usefully better; do not assume a larger model is more useful.
3. Run nano+tracking5 with the69-column diagnostics, separately observing static textured objects, slow head motion and occlusion. Locate grayscale/scene-check/flow/replay cost and rejection reasons; inspect actual box motion, not just box counts.
4. Separately obtain the Unity CPU Profiler capture showing the exact Camera.Pause hierarchy/self time and lifecycle events, initially without Deep Profiling or simultaneous RGP. Reconcile RGP CPU submission timestamps with its SQTT duration. Do not bypass required camera lifecycle handling.
5. Only if reliable propagation and net benefit are demonstrated, compare small+tracking with small-only. Otherwise keep detector-only. SAM, larger-model export and a Unity-version upgrade are not the next default intervention.

Longer-term, conditional work includes reduced/cached grayscale preparation or native-buffer processing if those stages dominate, GPU input preparation if device measurements justify it, adaptive refresh, recognition parity checks, repeatable motion clips, and end-user usability validation. Quantization, lower input resolution and pruning are **not demonstrated optimizations in the recorded tests**.

## Notebook availability and model release discipline

- Eleven historical notebooks were imported under [notebooks/archive](../notebooks/archive/README.md). Outputs/execution counts and transient metadata were removed. One embedded Trial5 API-key literal was replaced by a secure prompt in the publication copy; rotate that key if still active. Original source hashes and the redaction are recorded in [import_manifest.json](import_manifest.json).
- **The archived workbench Trial9 notebook is the old44-class template.** The supplied [small-model Colab notebook](../notebooks/current/README.md) is now available separately; its active code asserts45 rated classes. Legacy shared44 headings,300-versus40 evaluation/export settings, exact class order and checkpoint-to-ONNX provenance still require reconciliation. The nano Colab notebook is still outstanding.
- The handoff reported small-model GALVAT0.7478 on validation and0.6920 on frozen test. Do not describe0.75 as frozen-test accuracy or apply it to nano without its evaluation report.
- Keep model-specific thresholds distinct. The early Mac nano/small comparison intentionally shared the existing threshold config; it does not validate nano calibration. The Unity nano profile was added later from the supplied nano thresholds.
- Each future release should record a version, exact ONNX SHA256, class ordering, threshold-file hash, input/output semantics, training/evaluation notebook version and split-specific metrics. Unity should import that matched bundle and record its Git commit; do not swap only weights and keep stale labels/thresholds.
- Existing root `app.py`, dependencies and legacy weights were deliberately not changed by this documentation import. They remain the old demo, not the latest Trial9 sandbox. No new models, retraining, live benchmarks, Unity build or certified APK were produced here.
