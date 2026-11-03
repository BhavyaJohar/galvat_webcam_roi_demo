# GALVAT Final Delivery Plan

November 2 to November 20 2026

## Purpose and delivery decision

We will deliver a versioned GALVAT Magic Leap package by November 20, with a reproducible build, matched model configuration, device measurements, and documented limitations. This schedule replaces the October 7 to 30 milestones while retaining JHU's requirements for independent evaluation, fair model comparisons, traceable feedback, and gaze safety.

Nano detector-only is the practical baseline. Tracking is optional and will enter the final package only if it reliably follows objects and improves the complete device pipeline. SAM and Trial 10 are not prerequisites for delivery. We will not delay the tested detector-only fallback for an unproven tracker or late model change.

## Current status and integration baseline

- All earlier AUD Unity changes were pushed through `dd792ba`. JHU's `JHU_Bhavya_1026` branch at `865870369b5424e206d8829683de2fba9ee217c3` includes that commit. No earlier AUD Unity changes were missing.
- JHU added session-safe pending-result handling, saved result timing context, and result processing when camera intrinsics are unavailable. We will continue from this branch, review its changes jointly, and preserve them. Jadin will identify the release branch and own the merge/build; no automatic overwrite or merge is assumed.
- Earlier October 13 detector-only logs measured 8.2 fresh YOLO results per second, 22.4 processed camera frames per second, and 108 ms mean inference wall latency. These are different metrics, not a new benchmark of JHU's November build. Exact historical build/model hashes were not logged.
- The November 2 capture was reported as nano with tracking disabled. An RGP file alone does not establish sustained camera FPS, inference latency, GPU utilization, or a camera-pause root cause. Jadin's Unity CPU capture, viewer screenshots and matching CSVs are needed.
- The supplied small-model notebook includes measuring spoon and asserts 45 rated classes. Legacy shared44 names and evaluation at 300 versus deployment export at 40 require reconciliation. Saved outputs report validation GALVAT 0.7478 and frozen-test GALVAT 0.6920; these are not nano results or newly reproduced scores.

## Proposed ownership

Owners and acceptance limits will be confirmed jointly by November 4.

| Responsibility | Lead | Support |
|---|---|---|
| Model package, notebooks and reference inference | Bhavya and AUD | JHU review |
| Unity integration, release merge and ML2 builds | Jadin | Bhavya, Elnaz and Olivia |
| Device profiling, recordings and test execution | Jadin and JHU | AUD analysis |
| Images, scene coverage and annotation review | Yujia and JHU | AUD |
| E012 package and 11-class definition | JHU | AUD comparison |
| Acceptance and study readiness decision | AUD and JHU jointly | JHU study team |

<!-- pagebreak -->

## Part 1 Reconcile source and freeze the baseline

### November 2 to 4

AUD will publish the supplied small-model notebook and verify the exact nano and small ONNX files, canonical 45-class ordering, measuring spoon ID 44, model-specific thresholds, input normalization, output semantics and hashes. The nano notebook/evaluation export remains a required source. We will preserve the existing frozen test set and document its provenance; recordings and related frames must not cross evaluation splits.

JHU will provide the November build's exact commit, build flags, Unity/SDK/runtime versions, device/camera settings, both timing and tracking CSVs, RGP viewer screenshots, and the Unity CPU capture through Drive. Yujia will share the requested images through an agreed access-controlled location. The teams will identify missing labels rather than counting every unlabelled object as a false positive.

JHU will supply E012's 11-class list, model and preprocessing/decoder contract. AUD and JHU will explicitly map shared categories; unsupported or additional categories will be reported separately, not forced into an invented mapping. By November 4 we will confirm test conditions, error definitions, feedback latency/age limits, sample counts and the interpretation of label depth.

**Exit evidence:** agreed baseline commit/configuration manifest, reference input/output fixtures, shared-category mapping, test protocol and assigned owners. Unavailable inputs will be logged with an owner and due date.

## Part 2 Establish a controlled device baseline

### November 5 to 7

JHU will run nano-only and small-only on the same ML2 and application revision, retaining each model's verified preprocessing, decoder and validation-selected confidence thresholds. The comparison will reuse independent recordings and a scripted live sequence with static objects, slow head movement, changing gaze and occlusion. Freeze thresholds before scoring the independent test material.

For each mode, run at least three repetitions with 15 seconds of warmup and at least 60 seconds measured. Also perform a five-minute continuous run to inspect memory, thermal behavior, output stability and crashes. Match build type, input resolution, backend, scene order and device power state. Collect ordinary performance runs with profilers disconnected; collect Unity CPU and RGP diagnostic runs separately, initially without Deep Profiling.

Record camera callbacks, processed images, fresh detector results, frame gaps, preprocessing, inference scheduling/readback, postprocessing, rendering and feedback timestamps. Use a common monotonic clock or a documented clock conversion. Report median and p95 capture-to-feedback latency, not just average inference time.

**Exit evidence:** repeatable detector-only measurements and a ranked bottleneck explanation. The short November RGP capture will not substitute for continuous-video CSVs.

## Part 3 Fix the measured bottleneck and decide on tracking

### November 8 to 10

Bhavya and Jadin will implement one evidence-driven runtime change at a time and compare it with the frozen baseline. Candidates include grayscale preparation, repeated pixel/texture access, scene checks, unnecessary copies/uploads, scheduling/completion delays, and feedback placement. Actual stage measurements determine which change proceeds.

Only after detector-only measurements are stable, run nano plus tracking at interval 5 with the 69-column diagnostics. Examine static textured objects, slow movement and occlusion; use 3 or 10 only when the failure reasons justify the comparison. Measure tracker cost, successful propagation, ID continuity, drift, forced refreshes and stale boxes. A visible unpropagated YOLO box is not successful tracking.

**Decision by November 10:** retain tracking only with verified current geometry and a material duty-cycle or stability benefit without breaching the camera/latency budget. Otherwise freeze detector-only. Small plus tracking is conditional on this result. SAM stays outside the delivery critical path unless its complete ML2 runtime, memory and latency feasibility is separately demonstrated.

<!-- pagebreak -->

## Part 4 Compare models and validate user facing behavior

### November 11 to 13

AUD and JHU will compare E012 with the selected Trial 9 candidates on the agreed shared categories, using the same independent inputs and ML2 application build. Each model retains its own class map, preprocessing, decoder and frozen threshold profile. Report per-class precision, recall, sample counts and false-positive/false-negative definitions; report additional class coverage separately. Neither model is presumed better.

Where both 300-output and 40-output exports are available, AUD will run paired inputs and thresholds and document where the limit is applied. JHU will measure device latency, while AUD checks recall and small targets in clutter. Include fixed expected decoder outputs and coordinate checks. Do not add conventional NMS to the end-to-end export merely because an IoU setting exists; document whether IoU 0.65 is applicable to the evaluated path. If the 300-output reference is unavailable, document the gap rather than inventing a comparison.

JHU will test gaze switching between two objects, targets leaving view, occlusion/reacquisition, invalid gaze, and late detections after a head turn. Record the source frame, ROI/gaze and camera pose used for placement. The team will specify fixed display-plane cues versus measured-depth placement and the fallback when depth/intrinsics are unavailable. JHU's new intrinsics-independent result handling requires regression tests, not an assumption of correct 3D placement.

**Exit evidence:** model-selection rationale, feedback/placement tests and confirmed stale-output suppression. Any Trial 10 proposal requires a specific failure, joint approval, and enough time to repeat evaluation/export/device checks before November 17; otherwise defer it.

## Part 5 Complete regression tests and freeze the release

### November 14 to 17

JHU will run the agreed control-subject scenarios across lighting and occlusion conditions and review readiness for patient tasks. Record failures by class and condition, including empty ROIs, entry/exit, clutter, small objects, camera stop/restart, session changes and missing intrinsics. Preserve representative successful and failed examples with permission.

AUD will reconcile notebook metric definitions, model/export identity and release configuration. Jadin will freeze the selected model, Unity commit, thresholds, detector duty policy, association/expiry rules and feedback behavior by November 17. Optional changes that have not passed the same regression tests will be excluded. A simpler tested fallback takes priority over experimental features.

**Exit evidence:** joint release-candidate review, signed acceptance results or explicit exceptions, regression evidence and a rollback package. Patient testing depends on JHU's approved protocol, permissions and study readiness. The November 20 milestone covers engineering delivery, not completion of a patient study.

## Part 6 Reproduce package and deliver

### November 18 to 20

Reserve November 18 and 19 for a clean build/install, final continuous-video runs, configuration checks and a joint handoff rehearsal. AUD supplies the model/research package and deployment instructions; Jadin/JHU verifies the Unity build on the target ML2. Any late fix must repeat the affected regression tests and be recorded in the change log.

**November 20 delivery:** provide the final versioned package, source commits, installation/run instructions, reproducibility evidence, known limitations and the tested rollback configuration. If a target remains unmet, report the measured shortfall and agreed operating restrictions rather than presenting an unverified system as accepted.

<!-- pagebreak -->

## Measurement and acceptance requirements

The following engineering targets are proposed for joint confirmation by November 4. They are not claims about achieved November performance or clinical effectiveness.

- **Camera responsiveness:** at least 10 processed camera frames per second over each measured run. Separately report callback/render/tracking rates, minimum rolling-window rate and frame-gap tails; an average does not guarantee every update arrives within 100 ms.
- **Semantic rate:** report fresh detector results per second separately. Compare with the repeated current-build nano-only baseline, not a mixed-device historical average. A tracking configuration must not conceal reduced semantic freshness behind a higher display rate.
- **Feedback latency:** propose median capture-to-feedback at most 250 ms and p95 at most 500 ms, subject to joint approval. Instrument displayed and spoken feedback separately where implemented; mark missing paths as unmeasured.
- **Feedback validity:** propose a maximum 500 ms age since detector confirmation for feedback. On an ROI/target change, expired source, lost target or invalid gaze, suppress stale feedback and request a fresh result; do not wait for the age limit to suppress a known target mismatch. Final expiry, re-detection and repeated-speech rules must be explicit and regression-tested.
- **Traceability:** log inference/session ID, source frame/capture time, completion and feedback time, source gaze/ROI/pose, model/config hashes, build commit, device settings, discarded results and rejection reasons. Retain JHU's session-change invalidation and saved timing context.
- **Recognition:** freeze validation-selected thresholds and score the held-out set without further tuning. Report per-class precision/recall and counts on the 18 priority items and shared E012 categories. JHU/AUD will agree task-critical error limits and handling of partial annotations before choosing a winner.
- **Stability:** report memory and continuous-video behavior; investigate recurrent unexplained stalls, session leakage, stale feedback and crashes. If feedback safety or acceptance fails, restrict the build to engineering testing until resolved and approved.

## Subject testing scope

Subject to JHU's confirmed protocol, permissions, recruitment and readiness, control-subject evaluation would measure detection errors across lighting, occlusion, clutter and movement. Patient evaluation would assess everyday task support and practical reliability. Runtime benchmarks alone do not establish patient benefit.

| JHU difficulty grouping | Priority items |
|---|---|
| Easy | Exit sign, chair, table, coffee mug, wine glass, bathroom sign, trash can |
| Medium | Eyeglasses, TV remote, salt/pepper shaker, wallet, light switch, fork |
| Difficult | Hearing aid, cane, medicine bottle, paper towel dispenser, soap dispenser |

## Final package checklist

- Selected .pt/.onnx assets as available, exact hashes, 45-class mapping and model-specific thresholds; retain E012's separate contract if included in the comparison.
- Selected checkpoint/notebook and metric-definition provenance, validation/frozen-test reports, per-class counts, dataset/split manifest and documented annotation limitations. Do not relabel a validation score as test accuracy.
- Input/output fixtures, preprocessing/decoder/coordinate contract, output-cap comparison or stated limitation, and the actual NMS/IoU behavior.
- Unity release commit, package versions, ML2 build/install/run instructions, verified configuration and rollback path. Keep default tracking off unless it passes the acceptance gate.
- Matched device CSVs, timing summaries, permitted example recordings/images, profiler findings, failure cases and unmeasured/open issues with owners.
- AUD/JHU signoff by November 20, including model choice, accepted operating limits and any study-readiness restrictions.

## Repository ownership

Model building, notebooks and ML optimization evidence belong in `BhavyaJohar/galvat_webcam_roi_demo`. Unity application/deployment changes belong in `jadinV/GALVAT_R61`. The November integration baseline is `JHU_Bhavya_1026`; the teams will agree the final release branch before merging. Model and Unity release manifests must cross-reference each other.
