#!/usr/bin/env python3
"""Read-only, stdlib intake checks for GALVAT runtime CSVs; not acceptance analysis."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "config/runtime_csv_schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def audit(path):
    report = {"path": str(path), "schema": None, "rows": 0, "errors": [],
              "warnings": [], "current_complete": False,
              "schema_gaps": SCHEMA["schema_gaps"],
              "measurement_limit": SCHEMA["measurement_limit"]}
    errors, warnings = report["errors"], report["warnings"]
    previous = {}
    try:
        report["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        report["bytes"] = Path(path).stat().st_size
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, [])
            if len(header) != len(set(header)):
                errors.append("exact duplicate headers")
            kind = "tracking" if "Mode" in header or "TrackerBackend" in header else "timing"
            spec = SCHEMA[kind]
            legacy = kind == "tracking" and header == spec["legacy_columns"]
            report["schema"] = kind + ("_legacy35" if legacy else "_current")
            report["current_complete"] = header == spec["columns"]
            if legacy:
                warnings.append("legacy35: not current-complete; appended diagnostics missing")
            elif header != spec["columns"]:
                errors.append("current schema mismatch: missing=" + repr(sorted(set(spec["columns"]) - set(header)))
                              + "; extra=" + repr(sorted(set(header) - set(spec["columns"])))
                              + "; exact source column order required")
                return report
            footer = False
            for cells in reader:
                line = reader.line_num
                if kind == "timing" and not cells:
                    continue
                if kind == "timing" and cells == ["SESSION ENDED"] and not footer:
                    footer = True
                    continue
                if footer:
                    errors.append(f"line {line}: data after SESSION ENDED")
                    continue
                if len(cells) != len(header):
                    errors.append(f"line {line}: expected {len(header)} columns, got {len(cells)}")
                    continue
                report["rows"] += 1
                row = dict(zip(header, cells))
                prefix = f"line {line}: "
                tracking = kind == "tracking"
                stage_only = not tracking and all(not row[c] for c in header[16:])
                if stage_only and "stage-only timing rows: no completion metadata" not in warnings:
                    warnings.append("stage-only timing rows: no completion metadata")
                optional = set(spec.get("optional_fields", []))
                if tracking:
                    optional.update(spec["optional_callback_fields"])
                    optional.add("LastRefreshSourceAge_ms")  # M(NaN) before a refresh.
                    if row.get("TrackerDiagnosticsEnabled", "0") == "0":
                        optional.update(spec["diagnostic_fields"])
                elif stage_only:
                    optional.update(header[16:])
                text = {"RunId", "Mode", "WorkerBackend", "TrackerBackend", "TrackIds", "TrackSources",
                        "DiagnosticsSchema", "BuildGUID", "AppVersion", "ModelId", "ThresholdProfile",
                        "UnityVersion", "GraphicsDevice", "GraphicsAPI", "TimestampSource"}
                values = {}
                for field, cell in row.items():
                    if field.endswith("Json"):
                        try:
                            obj = json.loads(cell, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                            expected = list if field == "TrackGeometryJson" else dict
                            if not isinstance(obj, expected):
                                raise ValueError("wrong JSON container")
                        except ValueError:
                            errors.append(prefix + field + ": invalid JSON")
                    elif field in text:
                        if not cell and field in {"RunId", "Mode"} and not stage_only:
                            errors.append(prefix + field + ": required value blank")
                    elif not cell and field in optional:
                        continue
                    else:
                        try:
                            value = float(cell)
                            minimum = -1 if field == "LastCompletionInputFrameId" else 0
                            if not math.isfinite(value) or value < minimum:
                                raise ValueError("nonfinite or negative")
                            is_counter = not field.endswith(("_ms", "FPS", "Percent"))
                            if is_counter and not value.is_integer():
                                raise ValueError("fractional counter")
                            if field in {"InferencePending", "TrackerDiagnosticsEnabled", "DevelopmentBuild"} and value not in (0, 1):
                                raise ValueError("invalid flag")
                            values[field] = value
                        except ValueError:
                            errors.append(prefix + field + ": expected finite nonnegative number/counter")
                if tracking and not legacy and row["DiagnosticsSchema"] != "galvat.tracking_diagnostics.v1":
                    errors.append(prefix + "unexpected DiagnosticsSchema")
                if not stage_only:
                    key = (row.get("RunId"), row.get("SessionId"))
                    last = previous.setdefault(key, {})
                    for field in spec["monotonic_fields"]:
                        if field in values:
                            if field in last and values[field] < last[field]:
                                errors.append(prefix + field + ": decreased within run/session")
                            last[field] = values[field]
                if not tracking:
                    start, end = (values.get(c) for c in ("InputCallbackMonotonic_ms", "CompletionMonotonic_ms"))
                    if start is not None and end is not None and end < start:
                        errors.append(prefix + "completion precedes callback")
                    latency = values.get("CaptureToResult_ms")
                    if start is not None and end is not None and latency is not None:
                        if not math.isclose(end - start, latency, rel_tol=1e-6, abs_tol=0.01):
                            errors.append(prefix + "CaptureToResult_ms disagrees with clocks")
            if not report["rows"]:
                errors.append("no data rows")
    except (OSError, UnicodeError, csv.Error) as exc:
        errors.append(str(exc))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, help="create a new JSON report (never overwrite)")
    args = parser.parse_args(argv)
    reports = [audit(path) for path in args.csv]
    valid = all(not r["errors"] and r["current_complete"] for r in reports)
    payload = {"source_commit": SCHEMA["source_commit"], "valid": valid, "files": reports,
               "full_jhu_acceptance_measurement_complete": False}
    rendered = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if args.output:
        try:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
        except OSError as exc:
            parser.exit(2, f"cannot create report: {exc}\n")
    print(rendered, end="")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
