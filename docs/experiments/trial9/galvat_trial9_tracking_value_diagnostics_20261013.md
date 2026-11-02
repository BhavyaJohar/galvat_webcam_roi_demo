# Tracking value diagnostic update

Delivered to bhavya: **dd792ba28b1b0ee589986c33b0216625f616ee48**.

Objective: determine whether tracking is worth its cost, especially for a heavier
detector, without assuming heavier means better useful recognition. Baseline is
the returned October 13 acceptance run
`galvat_trial9_headset_tracking_acceptance_20261013_v1`: camera processing met
10 FPS, but tracking was slower than detector-only and mostly unpropagated.

Implemented:

- Nano detector-only is again the scene default. All earlier camera/ownership/
  rendering fixes remain. Models, 45-class mapping, profiles and NMS handling
  are unchanged.
- Optional tracking diagnostics distinguish gray capture, snapshot copy, scene
  checks, live flow, delayed replay and association. Counters explain terminal
  box failures, first point failures, refresh rejections and admitted fallback.
- Track geometry/source age plus model/profile/build/runtime/development/GPU
  context improve future CSV interpretation. First 35 fields retained; 69 total.
- Core instrumentation is observational; acceptance, search radius, confidence,
  refresh and stale-source policies were not loosened. Timings are CPU wall
  intervals; nested stages and cumulative totals must not be added naively.

Validation: 33 Release core tests, 28 integration cases in each editor/device
conditional build, 689 configuration assertions and 856,339 managed raster
assertions pass. Five real Unity Editor checks pass, including enabled/disabled
track neutrality and timing/rejection accounting. Independent review found no
critical defect. No new headset speedup or recognition-quality gain is claimed.

Jadin's next runs, all controlled for scene/build type/backend/object motion:

1. Nano detector-only.
2. Small detector-only, using its preserved optimized profile.
3. Nano + tracking interval5, diagnostics on; separate stationary, slow pan and
   occlusion segments. Each gets 15 s warmup plus 60 s measured, both CSVs and
   a short recording if permitted. A separate development Profiler capture must
   not be mistaken for a same-build-type release performance comparison.

Only test small+tracking after propagation is demonstrably reliable and cost
acceptable. Retain detector-only if tracking is still slower/stale. A larger
checkpoint would need separate export, recognition and latency validation.

Protocol: README_TRACKING_VALUE_DIAGNOSTICS.md on bhavya. Plan:
`galvat_trial9_tracking_value_diagnostics_v1`; local validation run:
`galvat_trial9_tracking_value_diagnostics_20261013_v1`; reproducible bundle:
`experiments/runtime/tracking_value_diagnostics_20261013/bundle.json`.

Physical ML2 and Android module are unavailable locally; this is tested source,
not a certified APK. Plan/run records use MCP; this report/bundle were written
directly before artifact registration.
