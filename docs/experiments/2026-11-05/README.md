# November 5 integrated tracking experiment

The [six returned device runs](galvat_six_device_tests_20261104.md) support nano with
expensive debugging disabled as the speed baseline. Tracker computation is relatively
cheap, but delayed replay and usable-feature availability remain unreliable. Raw inputs
were preserved locally; per-file hashes and metrics are in the paired JSON report.

The [next test plan](NEXT_TEST_PLAN.md) combines Jadin's dev changes with the newer feedback
work and adds an opt-in point-selection experiment. Unity implementation and regression
fixtures live in [Jadin's candidate branch](https://github.com/jadinV/GALVAT_R61/tree/bhavya-feedback-timing).
The pinned candidate is [c0e3e0d](https://github.com/jadinV/GALVAT_R61/commit/c0e3e0d1e9620526d5387bfed2d6419cfada7f63),
which incorporates dev 1cd4486 and feedback 428c78b. It also corrects the off-centre
crop conversion for world-label rays; it preserves dev's intended vertical conversion
while matching the actual saved crop origin. Font sizing is applied once within bounds.

Local validation passed 57 integration tests in both managed conditional paths, 38 core
tracker tests, nine real Unity Editor checks, and the existing model/label/raster checks.
Research tooling passed 25 Python tests. These are local regressions, not Magic Leap
acceptance results. Unity's Editor report is versioned in the candidate's Validation folder.

The point selector searches for strong texture inside 25 object cells with at most 81
candidates per cell. It retains the existing patch match, uniqueness, forward/backward,
consensus, semantic and expiry checks. Fixed-grid selection remains the default baseline.
Synthetic tests show one sparse-texture failure can be recovered; they do not establish
Magic Leap reliability. The [local core benchmark](point_selection_benchmark.json) records
the added CPU cost, including cases where baseline rejection does less work. No device
speedup is inferred from that benchmark.

Updated CSV auditing supports the integrated 43-column timing and 82-column tracking
schemas, historical formats, and the separate nine-column discard-event file. It labels
missing historical fields and keeps full JHU acceptance completeness false: synchronized
gaze/pose, speech and physical display measurements remain unavailable.

Reproduce the supplied device analysis with:

`python tools/analyze_november_six_tests.py --folder /path/to/latest_test_results --output /tmp/six-tests.json`

Validate returned logs with:

`python tools/audit_runtime_csv.py timing.csv tracking.csv optional_discards.csv --output /tmp/intake.json`

Run the Python checks with `python -m unittest discover -s tests -p 'test_*.py'`.
Use a new output path for each report; the tools do not overwrite raw inputs or reports.
