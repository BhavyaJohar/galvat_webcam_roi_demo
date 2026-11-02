# Trial 9 Mac camera runtime results — 2026-10-07

What ran: four sequential 120-frame live webcam/window runs at 1280×720 on
Apple Silicon, macOS 27.0.1. Default ORT thread count. The existing ONNX and
all 45 supplied thresholds were held fixed. No image recordings were saved.
Full reproducibility metadata and per-frame timing are in `experiments/runtime/`.
Run records and plan were saved through experiments MCP.

| Runtime | Display submissions/sec | YOLO inference mean | Tracker mean/frame | YOLO frame fraction | Losses / forced loss refreshes |
|---|---:|---:|---:|---:|---:|
| CPU YOLO every frame | 11.26 | 70.20 ms | 0 | 100% | 0 / 0 |
| CPU YOLO + simple every 5 | 22.70 | 67.28 ms | 3.43 ms | 24.17% | 8 / 8 |
| CPU YOLO + simple every 10 | 24.87 | 77.07 ms | 4.04 ms | 10% | 1 / 0 |
| Core ML + CPU YOLO every frame | 28.47 | 15.48 ms | 0 | 100% | 0 / 0 |

Camera reads were about 30/sec. OpenCV submissions are not measured physical
monitor presentation. These original wall summaries include brief shutdown;
the final runner freezes timing before cleanup and records pause duration.
No runs were paused. Active processing FPS is separately stored in each JSON.
YOLO mean excludes non-detector frames; tracker mean includes initialization
and refresh as well as propagation. The ten-frame loss count includes a
refresh expiry, which need not produce an automatic loss refresh.

What changed: periodic detection reduced CPU detector duty; the independent
Core ML comparison changed only detector provider. Core ML profiling confirms
mixed CPU and Core ML execution, with nine fused Core ML partitions; it does
not determine GPU versus Neural Engine placement. A separate zero-frame Core
ML inference test averaged 14.22 ms and CPU averaged 62.30 ms (10 repetitions,
one warmup), but the live figures above are the relevant pipeline results.

What improved: CPU five-frame propagation approximately doubled display
throughput, while Core ML every-frame inference approached camera throughput.
What deteriorated: tracking introduced loss/expiry events and opportunities
for geometry drift; the aggregate counts do not measure harmful drift or ID
switches. Recognition quality was not evaluated and no improvement is claimed.

Likely reason: LK propagation is much cheaper than CPU YOLO; Core ML accelerates
most graph computation. Confidence is high in the recorded timing of these
short runs, but low in generalization or assistive usability because scenes
were uncontrolled and no ground truth or deliberate head-motion protocol
was used. Different loss counts cannot establish superiority of interval 10.

SAM 3.1 was not executed: official multiplex import queries CUDA without a
guard and imports Triton-dependent paths. Its public finite-video session API
also needs a live-camera adapter. No SAM/MPS latency or memory feasibility is
claimed. The explicit `sam3` backend fails without selecting CPU or LK. The
user's Hugging Face key helps checkpoint access, not these compute blockers.
See the sandbox README for official implementation links and HF login steps.

Review found and corrected camera release/read concurrency, resolution-change
ID reuse, and pause/cleanup contamination of wall timing. Focused tests cover
geometry, thresholds, mapping, association, occlusion, timing and scheduling.

Next decision: start with live Core ML YOLO-only and compare CPU LK using the
same prerecorded challenging scenes at 3/5/10/15. Check small-object drift,
motion, occlusion, ID persistence, false persistent tracks and reacquisition.
Core ML alone may already satisfy Mac responsiveness; SAM viability requires
a separately scoped MPS port or CUDA worker. None of these results establish
Magic Leap FPS, wearable memory viability or clinical benefit.
