# Project Requirements Based on Plans Shared by AUD on 10/07/2026

JHU Comments date: 10/09/2026

Updated timeline: November 2–20, 2026

## Subject Testing Plans by JHU

Our testing will use the selected model and deployment configuration prepared for the November 20 delivery and will focus on two objectives:

Initial Testing with Control Subjects: Evaluate the model's detection performance by measuring false-positive and false-negative rates in different conditions, particularly different lighting conditions and levels of occlusion. The goal is to determine whether the system can reliably identify objects of interest in different conditions.

Patient Testing in Real-World Scenarios: Assess the proposed technology with patients performing everyday tasks across a range of scenarios. Evaluate its performance and reliability in practical use, including its ability to support users in completing daily activities. Subject testing will follow JHU's approved testing arrangements.

Proposed list of items to be tested:

Easy: exit sign, chair, table, coffee mug, wine glass, bathroom sign, trash can

Medium: eyeglasses, tv remote, salt pepper shaker, wallet, light switch, fork

Difficult: hearing aid, cane, medicine bottle, paper towel dispenser, soap dispenser

## Project Requirements

## Part 1_AUD

### Nov 2–4 — Confirm Trial 9 and deployment package

Review the completed Trial 9 training, evaluation and dataset audit.
Confirm the 45-class mapping, including measuring spoon, matches the deployed models.
Confirm the frozen test set and save the associated evaluation reports.
Verify the small and nano deployment ONNX files and their reduced 40-detection output.
Package the model-specific confidence thresholds and final inference settings.
Upload the current training notebooks and document the model/export versions.
Keep Trial 9 as the deployment candidate unless testing identifies a significant issue.

## Part 1_JHU Comments

Share the completed audit, the exact 45-class model and label map, and per-class precision/recall with sample counts on an independently held-out set. That would let us distinguish missing coverage from annotation problems and identify which task-critical classes still need work. Related frames from the same recording should stay in one split, and thresholds should be selected on validation data.

Related to the output limit: Include a paired comparison of the 300-output and 40-output exports, where both are available, using the same inputs and thresholds. Document where the limit is applied and report both device latency and any change in recall, especially for small targets in clutter. Include fixed input/output examples so the Unity decoder can be checked against the reference implementation. Document any unavailable comparison rather than assuming the exports are equivalent.

## Part 2_AUD

### Nov 2–6 — Integration and baseline runtime testing

Continue from the JHU_Bhavya_1026 branch, which includes the earlier dd792ba updates, and preserve the newer integration changes.
Verify preprocessing, output decoding, class mapping, confidence thresholds and bounding-box coordinates.
Use CPU and GPU profilers to identify pauses and slowdowns.
Benchmark continuous video and measure camera FPS, fresh detector-result rate, inference latency, preprocessing, postprocessing and application overhead.
Establish a detector-only baseline before comparing tracking configurations.
Continue separate tracking tests with different refresh intervals.
Update label size based on bounding-box area and check readability and sizing stability.
Begin comparing Trial 9 with E012 using matched test inputs and verified model settings.
Identify the largest runtime bottlenecks before making additional model changes.

## Part 2_JHU Comments

Add a controlled comparison of E012 and AUD's latest Trial 9 to guide the next optimization steps. Confirm Trial 9's final class list and map the shared task categories to E012's 11 classes. Use the same independent recordings, including new objects/scenes, and the same Magic Leap device and application build. Each model should use its own verified preprocessing/decoder, with thresholds chosen on validation data and then frozen.

Compare per-class precision/recall and sample counts on the shared categories, and report additional class coverage separately. Also compare fresh detection rate, capture-to-feedback latency, memory and continuous-video stability, starting with a detector-only baseline before adding tracking. Both models remain development candidates; the aim is to identify whether data, the detector or the application pipeline should be improved, without assuming either model is better.

Related to timing: Each result should record its source camera frame and capture timestamp, inference completion, and displayed/spoken feedback time on a common clock. Report fresh detector-result rate separately from rendering/tracking rate, plus median and p95 capture-to-feedback latency and discarded-result counts. Repeated runs should identify the model, build, device configuration and scene. Agree latency and task-error acceptance limits during the baseline phase, before selecting an optimization. Keep profiler captures separate from ordinary performance runs.

## Part 3_AUD

### Nov 9–13 — Frame-rate and temporal-tracking development

Make frame-rate improvement and reliable tracking the primary development goals.
Test periodic YOLO detection with tracking between inference passes.
Test refresh intervals such as every 3, 5 and 10 frames against the detector-only baseline.
Measure tracker latency separately from YOLO latency.
Evaluate track stability, reacquisition, false-track persistence and behavior during movement, clutter, occlusion and object entry/exit.
Keep YOLO as the semantic authority while tracking maintains object position and identity.
Check that CSV files contain all intended fields and that the recorded values and timestamps are valid.
Use the timing and failure data to fix the largest bottlenecks.
Investigate adaptive refresh only if the fixed-interval pipeline is reliable.

## Part 3_JHU Comments

Keep an on-device feasibility check before committing the delivery to a tracking workflow. Compare the complete detector-plus-tracker pipeline with detector-only, including tracker latency and memory use. If SAM is considered, specify its variant, export/runtime path and compatibility rather than assuming it can run on Magic Leap.

Make the tracking inclusion decision by November 13. If tracking does not meet the agreed performance and reliability requirements, retain a simpler or detector-only fallback. A reliable tracking goal should not be treated as an achieved result before the tests demonstrate it.

## Part 4_AUD

### Nov 9–13 — Magic Leap integration and real-world testing

Integrate and test the improved inference pipeline on Magic Leap.
Benchmark detector inference, tracking, preprocessing, rendering and total frame rate.
Test whether lower detector duty cycle produces a meaningful improvement over detector-only.
Complete the Trial 9 and E012 comparison under matched conditions.
Collect real-world performance data while moving gaze between two targets.
Test a target leaving the frame and reentering.
Test target occlusion and reidentification.
Test invalid or unusable gaze input and late results after a head turn.
Evaluate area-based label sizing during realistic movement.
Check CSV completeness in both detector-only and tracking modes, distinguishing missing measurements from fields that are not applicable.
Select the confidence thresholds, output limits, tracking parameters, refresh schedule and association settings supported by the results.

## Part 4_JHU Comments

For acceptance tests, check that feedback corresponds to the user's current gaze/ROI target. Include gaze moving between two objects, the target leaving view and reentering, occlusion and reacquisition, invalid gaze input, and a late detection arriving after the user turns away.

Define a maximum age since the last detector confirmation, track-expiry and re-detection rules, and suppression of stale or repeated feedback. The frame, gaze and camera pose used for placement should be traceable. For label-depth changes, state whether placement is a fixed display-plane cue or uses measured depth, and define the behavior when depth is unavailable. Keep label sizing separate from depth/placement behavior.

## Part 5_AUD

### Nov 16–17 — Final tuning and system freeze

Use the integration results to determine whether additional model work is justified.
Train Trial 10 only if a specific model or dataset issue materially affects usability and there is enough time to repeat all required checks.
Otherwise, keep Trial 9 fixed and focus on runtime stability and deployment.
Begin final testing and evaluation procedures.
Select and freeze the final detector, ONNX export, class thresholds, tracking configuration, refresh policy and inference settings.
Run the final test-set evaluation and save all metrics.
Run continuous-video and Magic Leap performance tests, including the gaze and target scenarios above.
Save representative successful examples and remaining failure cases.

## Part 5_JHU Comments

Identify the exact branch/commit containing the suggested changes and list what is being reused and what still needs implementation. Trial 9 and E012 should retain their own class maps and preprocessing/decoding contracts if both are evaluated.

Use the November 13 integration checkpoint to agree the tested tracking or detector-only fallback. Reserve November 16–17 for regression checks and the release freeze. A Trial 10 change should proceed only if there is enough time to repeat the model, export and device checks before delivery.

## Part 6_AUD

### Nov 18–20 — Final testing deliverable and documentation

Complete the final testing and evaluation procedures started during November 16–17.
Organize the final .pt and .onnx models, configuration files, class mappings, thresholds, metrics and dataset information.
Document the changes from earlier trials and the final model results.
Document the full detection pipeline and Unity integration, including preprocessing/postprocessing, tracking and refresh behavior where used.
Report final camera FPS, fresh detector-result rate and end-to-end latency alongside recognition metrics.
Document known limitations and remaining deployment bottlenecks.
Verify the final build, installation steps and files needed to reproduce and use the system.
Final GALVAT delivery by November 20.
