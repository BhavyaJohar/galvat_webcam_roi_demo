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
VARIANTS = json.loads(SCHEMA_PATH.with_name("runtime_csv_variants.json").read_text(encoding="utf-8"))


def csv_boundary_lines(stream):
    """Ignore padding after closing quotes; retain quoted contents and strict errors."""
    state = "start"
    padding = ""
    for line in stream:
        output = []
        for char in line:
            if state == "closed" and char in " \t":
                padding += char
                continue
            if state == "closed":
                # Only a delimiter/newline may follow padded closing quotes.
                if char not in ",\r\n":
                    output.append(padding)
                if char == '"' and not padding:
                    state = "quoted"  # Escaped quote inside a quoted field.
                elif char in ",\r\n":
                    state = "start"
                else:
                    state = "unquoted"
                padding = ""
            elif state == "quoted":
                if char == '"':
                    state = "closed"
            elif char in ",\r\n":
                state = "start"
            elif state == "start" and char == '"':
                state = "quoted"
            elif char != " " or state != "start":
                state = "unquoted"
            output.append(char)
        yield "".join(output)


def select_schema(header):
    for kind in ("timing", "tracking", "discard"):
        latest = VARIANTS[kind]
        base = SCHEMA.get(kind, {})
        candidates = [(latest["columns"], True)]
        if base:
            candidates.append((base["columns"], False))
            candidates.append((latest["columns"][:38 if kind == "timing" else 70], False))
            if "legacy_columns" in base:
                candidates.append((base["legacy_columns"], False))
        for columns, current in candidates:
            if header == columns:
                spec = dict(base)
                spec.update(latest)
                spec["columns"] = columns
                for field in ("optional_fields", "text_fields", "monotonic_fields"):
                    spec[field] = list(dict.fromkeys(base.get(field, []) + latest.get(field, [])))
                return kind, spec, current
    return None, None, False


def check_output(row, values, errors, prefix):
    spec = VARIANTS["timing"]
    status = row["OutputStatus"]
    measured = status in spec["measured_statuses"]
    if not measured and status not in spec["unmeasured_statuses"]:
        errors.append(prefix + "unexpected OutputStatus")
    if row["OutputTimingBoundary"] != "ApplicationMutation_NotPhysicalDisplay":
        errors.append(prefix + "unexpected OutputTimingBoundary")
    application = values.get("OutputApplicationMonotonic_ms")
    latency = values.get("CallbackToOutput_ms")
    callback = values.get("InputCallbackMonotonic_ms")
    completion = values.get("CompletionMonotonic_ms")
    if measured:
        if application is None or completion is None:
            errors.append(prefix + "measured output requires application and completion clocks")
        elif application < completion:
            errors.append(prefix + "application precedes completion")
        if values.get("CaptureToResult_ms") is not None:
            if latency is None or callback is None or application is None or not math.isclose(
                    application - callback, latency, rel_tol=1e-6, abs_tol=0.01):
                errors.append(prefix + "CallbackToOutput_ms disagrees with clocks")
        elif latency is not None:
            errors.append(prefix + "CallbackToOutput_ms without measured callback-to-result")
    elif application is not None or latency is not None:
        errors.append(prefix + "unmeasured output has application clocks")
    if row.get("TimestampSource") and row["TimestampSource"] not in spec["timestamp_sources"]:
        errors.append(prefix + "unexpected TimestampSource")
    if "InputCameraFrameIdSource" in row:
        source = row["InputCameraFrameIdSource"]
        if source not in spec["camera_frame_id_sources"]:
            errors.append(prefix + "unexpected InputCameraFrameIdSource")
        if (source == "Unavailable") != (not row["InputCameraFrameId"]):
            errors.append(prefix + "InputCameraFrameId/source availability mismatch")


def audit(path):
    report = {"path": str(path), "schema": None, "rows": 0, "errors": [],
              "warnings": [], "current_complete": False,
              "schema_gaps": VARIANTS["schema_gaps"],
              "measurement_limit": VARIANTS["measurement_limit"],
              "variant_source_commit": VARIANTS["source_commit"],
              "variant_source_status": VARIANTS["source_status"]}
    errors, warnings = report["errors"], report["warnings"]
    previous = {}
    lifetime = {}
    last_run = None
    try:
        report["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        report["bytes"] = Path(path).stat().st_size
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            header_reader = csv.reader(stream, strict=True)
            raw_header = next(header_reader, [])
            header = [field.strip() for field in raw_header]
            report["raw_header"] = raw_header
            report["normalized_header"] = header
            if header != raw_header:
                warnings.append("header boundary whitespace normalized; raw_header and input hash preserved")
            if len(header) != len(set(header)):
                errors.append("duplicate headers after boundary whitespace normalization")
            reader = csv.reader(csv_boundary_lines(stream), strict=True, skipinitialspace=True)
            kind, spec, current = select_schema(header)
            if kind is None:
                errors.append("unrecognized schema: exact source column order required")
                return report
            report["schema"] = kind + ("_current" if current else "_legacy" + str(len(header)))
            report["column_count"] = len(header)
            report["current_complete"] = current
            report["missing_current_columns"] = [c for c in VARIANTS[kind]["columns"] if c not in header]
            if not current:
                warnings.append(f"legacy{len(header)}: not current-complete; later fields missing")
                if kind == "timing" and len(header) == 27:
                    report["schema_gaps"] = SCHEMA["schema_gaps"]
                    report["measurement_limit"] = SCHEMA["measurement_limit"]
            footer = False
            for cells in reader:
                line = header_reader.line_num + reader.line_num
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
                stage_only = kind == "timing" and all(not row[c] for c in header[16:27])
                if stage_only and "stage-only timing rows: no completion metadata" not in warnings:
                    warnings.append("stage-only timing rows: no completion metadata")
                optional = set(spec.get("optional_fields", []))
                if tracking:
                    optional.update(spec["optional_callback_fields"])
                    optional.add("LastRefreshSourceAge_ms")  # M(NaN) before a refresh.
                    if row.get("TrackerDiagnosticsEnabled", "0") == "0":
                        optional.update(spec["diagnostic_fields"])
                elif stage_only:
                    optional.update(header[16:27])
                text = {"RunId", "Mode", "WorkerBackend", "TrackerBackend", "TrackIds", "TrackSources",
                        "DiagnosticsSchema", "BuildGUID", "AppVersion", "ModelId", "ThresholdProfile",
                        "UnityVersion", "GraphicsDevice", "GraphicsAPI", "TimestampSource"}
                text.update(spec.get("text_fields", []))
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
                            minimum = -1 if field == "LastCompletionInputFrameId" or (kind == "discard" and field == "InputFrameId") else 0
                            if not math.isfinite(value) or value < minimum:
                                raise ValueError("nonfinite or negative")
                            is_counter = not field.endswith(("_ms", "FPS", "Percent")) and field != "ReassociationIoU"
                            if is_counter and not value.is_integer():
                                raise ValueError("fractional counter")
                            if field in {"InferencePending", "TrackerDiagnosticsEnabled", "DevelopmentBuild", "ExpensiveDiagnosticsEnabled"} and value not in (0, 1):
                                raise ValueError("invalid flag")
                            values[field] = value
                        except ValueError:
                            errors.append(prefix + field + ": expected finite nonnegative number/counter")
                if tracking and "DiagnosticsSchema" in row:
                    expected = "galvat.tracking_diagnostics.v2" if len(header) == 82 else "galvat.tracking_diagnostics.v1"
                    if row["DiagnosticsSchema"] != expected:
                        errors.append(prefix + "unexpected DiagnosticsSchema")
                if tracking and "PointSelectionMode" in row and row["PointSelectionMode"] not in {"TextureAwareCells", "FixedGrid"}:
                    errors.append(prefix + "unexpected PointSelectionMode")
                if tracking and "MaxDetectorConfirmationAge_ms" in row:
                    if (values.get("ActiveTracks") == 0) != (not row["MaxDetectorConfirmationAge_ms"]):
                        errors.append(prefix + "MaxDetectorConfirmationAge_ms availability disagrees with ActiveTracks")
                if not stage_only:
                    key = (row.get("RunId"), row.get("SessionId"))
                    last = previous.setdefault(key, {})
                    for field in spec["monotonic_fields"]:
                        if field in values:
                            if field in last and values[field] < last[field]:
                                errors.append(prefix + field + ": decreased within run/session")
                            last[field] = values[field]
                run_id = row.get("RunId")
                # Stage-only rows omit RunId, but their appended lifetime counters
                # belong to the same logger as the surrounding completion rows.
                if run_id:
                    if last_run is None and None in lifetime:
                        lifetime[run_id] = lifetime.pop(None)
                    last_run = run_id
                run_last = lifetime.setdefault(run_id or last_run, {})
                for field in spec.get("run_monotonic_fields", []):
                    if field in values:
                        if field in run_last and values[field] < run_last[field]:
                            errors.append(prefix + field + ": decreased within run")
                        run_last[field] = values[field]
                if kind == "discard":
                    reason_key = ("ReasonCount", row["Reason"])
                    if not row["Reason"]:
                        errors.append(prefix + "Reason: required value blank")
                    if "ReasonCount" in values:
                        if reason_key in run_last and values["ReasonCount"] < run_last[reason_key]:
                            errors.append(prefix + "ReasonCount: decreased within run/reason")
                        run_last[reason_key] = values["ReasonCount"]
                if kind == "timing":
                    start, end = (values.get(c) for c in ("InputCallbackMonotonic_ms", "CompletionMonotonic_ms"))
                    if start is not None and end is not None and end < start:
                        errors.append(prefix + "completion precedes callback")
                    latency = values.get("CaptureToResult_ms")
                    if start is not None and end is not None and latency is not None:
                        if not math.isclose(end - start, latency, rel_tol=1e-6, abs_tol=0.01):
                            errors.append(prefix + "CaptureToResult_ms disagrees with clocks")
                    if "OutputStatus" in row:
                        check_output(row, values, errors, prefix)
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
               "variant_source_commit": VARIANTS["source_commit"],
               "variant_source_status": VARIANTS["source_status"],
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
