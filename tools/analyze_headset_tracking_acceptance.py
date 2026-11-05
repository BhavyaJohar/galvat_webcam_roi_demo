"""Read-only CSV analysis; optional derived JSON never replaces input data."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def distribution(values):
    values = sorted(values)
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "p95": values[math.ceil(.95 * len(values)) - 1],
            "min": values[0], "max": values[-1]}


def analyze(path, warmup_seconds):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        fields = [f.strip() for f in reader.fieldnames]
        raw = list(reader)
    rows = [{k.strip(): v.strip() for k, v in row.items()} for row in raw
            if None not in row and all(v is not None for v in row.values())]
    valid = [r for r in rows if r.get("SessionId", "").isdigit()
             and r.get("SessionElapsed_ms")]
    result = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "columns": len(fields), "valid_rows": len(valid),
              "excluded_rows": len(raw) - len(valid), "sessions": []}
    groups = collections.defaultdict(list)
    for row in valid:
        groups[row["RunId"], row["SessionId"]].append(row)
    for (run, session), group in groups.items():
        elapsed = [float(r["SessionElapsed_ms"]) for r in group]
        if any(b <= a for a, b in zip(elapsed, elapsed[1:])):
            raise ValueError(f"Nonmonotonic session clock: {path}, {session}")
        sample = [r for r in group if float(r["SessionElapsed_ms"]) >= warmup_seconds * 1000]
        item = {"run_id": run, "session_id": session, "rows": len(group),
                "last_elapsed_ms": elapsed[-1], "postwarm_rows": len(sample),
                "settings": {key: dict(collections.Counter(r[key] for r in group))
                             for key in ("Mode", "WorkerBackend", "TrackerBackend",
                                         "RefreshFrames", "K_Layers", "TimestampSource")
                             if key in fields}}
        result["sessions"].append(item)
        if len(sample) < 2:
            item["excluded_from_postwarm_comparison"] = True
            continue
        dt = (float(sample[-1]["SessionElapsed_ms"]) - float(sample[0]["SessionElapsed_ms"])) / 1000
        item["postwarm_span_seconds"] = dt
        def delta(key):
            return float(sample[-1][key]) - float(sample[0][key])
        if "Inference_ms" in fields:
            item["detector_completions_per_second"] = delta("CompletedDetections") / dt
            item["metrics"] = {key: distribution([float(r[key]) for r in sample if r[key]])
                               for key in ("Inference_ms", "Scheduling_ms", "GPUReadback_ms",
                                           "Preprocess_ms", "Postprocess_ms", "CaptureToResult_ms",
                                           "Output_ms", "Detections")}
            item["nonzero_detection_row_percent"] = 100 * sum(float(r["Detections"]) > 0 for r in sample) / len(sample)
        else:
            counters = [key for key in ("CameraFrames", "OverlayFrames", "YOLOSubmissions",
                        "YOLOCompletions", "LostTracks", "ForcedRefreshes", "AcceptedRefreshes",
                        "RejectedRefreshes", "CameraCallbacks", "OverwrittenCallbacks",
                        "CropPreparations") if key in fields]
            item["counter_deltas"] = {key: delta(key) for key in counters}
            item["processed_camera_fps"] = delta("CameraFrames") / dt
            item["detector_completions_per_second"] = delta("YOLOCompletions") / dt
            item["detector_submission_frame_percent"] = 100 * delta("YOLOSubmissions") / delta("CameraFrames")
            item["forced_refresh_submission_percent"] = 100 * delta("ForcedRefreshes") / delta("YOLOSubmissions")
            item["nonzero_track_row_percent"] = 100 * sum(float(r["ActiveTracks"]) > 0 for r in sample) / len(sample)
            if "CameraCallbacks" in fields:
                item["callback_fps"] = delta("CameraCallbacks") / dt
                item["overwritten_callback_percent"] = 100 * delta("OverwrittenCallbacks") / delta("CameraCallbacks")
            sources = collections.Counter(source for r in sample for source in r["TrackSources"].split(";") if source)
            item["track_observation_sources"] = dict(sources)
            item["track_observation_source_percent"] = {k: 100 * v / sum(sources.values()) for k, v in sources.items()}
            item["metrics"] = {key: distribution([float(r[key]) for r in sample])
                               for key in ("Tracker_ms", "Overlay_ms", "FrameWork_ms",
                                           "RefreshReplay_ms", "ActiveTracks", "CallbackQueue_ms",
                                           "CameraProcessingWindowFPS") if key in fields}
            gaps = [float(b["SessionElapsed_ms"]) - float(a["SessionElapsed_ms"])
                    for a, b in zip(sample, sample[1:])]
            item["processed_frame_completion_gap_ms"] = distribution(gaps)
            item["processed_frame_gaps_over100ms"] = sum(gap > 100 for gap in gaps)
            if "CameraProcessingWindowFPS" in fields:
                item["logged_window_fps_rows_below10"] = sum(float(r["CameraProcessingWindowFPS"]) < 10 for r in sample)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--warmup-seconds", type=float, default=15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"schema": "galvat.headset_tracking_acceptance.v1", "warmup_seconds": args.warmup_seconds,
              "method": "Within-session final-minus-first counter delta / elapsed delta after warmup; all short sessions retained but excluded. Source percentages count track observations, not ground-truth accuracy. Nearest-rank p95; repeated window fields are not independent windows. FrameWork excludes some asynchronous completion, native callback and CSV work. No commit/model identity is supplied by CSV.",
              "files": [analyze(path, args.warmup_seconds) for path in args.files]}
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(payload)
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
