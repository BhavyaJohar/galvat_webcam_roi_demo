"""Synthetic validator fixtures only; none represent runtime measurements."""
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("audit_runtime_csv", ROOT / "tools/audit_runtime_csv.py")
audit_csv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit_csv)


class RuntimeCsvTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "input.csv"

    def row(self, kind="tracking", diagnostics=False):
        schema = audit_csv.SCHEMA[kind]
        row = dict.fromkeys(audit_csv.VARIANTS[kind]["columns"], "0")
        row.update(RunId="fixture-run", SessionId="1", SessionElapsed_ms="10", InputFrameId="1")
        if kind == "tracking":
            row.update(Mode="YOLO_FLOW" if diagnostics else "YOLO_ONLY",
                       WorkerBackend="GPU", TrackerBackend="None", TrackIds="", TrackSources="",
                       DiagnosticsSchema="galvat.tracking_diagnostics.v2", PointSelectionMode="FixedGrid",
                       ReassociationIoU="0.3", MaxDetectorConfirmationAge_ms="",
                       TrackerDiagnosticsEnabled=str(int(diagnostics)), LastCompletionInputFrameId="-1")
            for field in schema["optional_callback_fields"]:
                row[field] = ""
            for field in schema["diagnostic_fields"]:
                row[field] = "0" if diagnostics else ""
            row["LastRefreshSourceAge_ms"] = ""  # Legitimate even with diagnostics on.
            for field in ("FlowRejectCountsJson", "PointRejectCountsJson", "RefreshRejectCountsJson"):
                row[field] = '{"reason,quoted": 1}'
            row["TrackGeometryJson"] = '[{"id":1,"box_bottom_up":[1,2,3,4]}]'
        else:
            row.update(InputCallbackMonotonic_ms="100", CompletionMonotonic_ms="110",
                       CaptureToResult_ms="10", CompletedDetections="1", TimestampSource="CameraCallbackArrival",
                       WorkerBackend="GPU", Latency_ms="", Interval_ms="")
            row.update(OutputApplicationMonotonic_ms="115", CallbackToOutput_ms="15",
                       OutputStatus="MarkersApplied", OutputTimingBoundary="ApplicationMutation_NotPhysicalDisplay",
                       InputCameraFrameId="", InputCameraFrameIdSource="Unavailable", OutputFrameId="",
                       TimestampSource="NativeCameraCallbackArrival")
        return row

    def write(self, rows, kind="tracking", header=None, footer=False):
        header = header or audit_csv.VARIANTS[kind]["columns"]
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for row in rows:
                writer.writerow([row[c] for c in header] if isinstance(row, dict) else row)
            if footer:
                writer.writerow([])
                writer.writerow(["SESSION ENDED"])
        return audit_csv.audit(self.path)

    def test_current82_disabled_diagnostics(self):
        report = self.write([self.row()])
        self.assertEqual(len(audit_csv.SCHEMA["tracking"]["columns"]), 69)
        self.assertEqual(report["errors"], [])
        self.assertTrue(report["current_complete"])
        self.assertIn("gaze", report["schema_gaps"])

    def test_all_header_variants_and_historical_provenance(self):
        for kind, counts in (("timing", (27, 38, 43)), ("tracking", (35, 69, 70, 82))):
            for count in counts:
                with self.subTest(kind=kind, count=count):
                    row = self.row(kind)
                    if kind == "tracking" and count != 82:
                        row["DiagnosticsSchema"] = "galvat.tracking_diagnostics.v1"
                    if kind == "timing" and count == 27:
                        row["TimestampSource"] = "CameraCallbackArrival"
                    header = audit_csv.VARIANTS[kind]["columns"][:count]
                    report = self.write([row], kind, header=header)
                    self.assertEqual(report["errors"], [])
                    self.assertEqual(report["current_complete"], count == len(audit_csv.VARIANTS[kind]["columns"]))
                    if not report["current_complete"]:
                        self.assertTrue(report["warnings"])
                        self.assertTrue(report["missing_current_columns"])
                        with contextlib.redirect_stdout(io.StringIO()):
                            self.assertEqual(audit_csv.main([str(self.path)]), 1)
        self.assertEqual(audit_csv.SCHEMA["source_commit"], "865870369b5424e206d8829683de2fba9ee217c3")
        self.assertRegex(audit_csv.VARIANTS["source_commit"], r"^[0-9a-f]{40}$")
        self.assertNotEqual(audit_csv.VARIANTS["source_commit"], audit_csv.SCHEMA["source_commit"])

    def test_output_statuses_and_source_clock_consistency(self):
        for status in audit_csv.VARIANTS["timing"]["measured_statuses"]:
            self.assertEqual(self.write([dict(self.row("timing"), OutputStatus=status)], "timing")["errors"], [])
        for status in audit_csv.VARIANTS["timing"]["unmeasured_statuses"]:
            row = dict(self.row("timing"), OutputStatus=status,
                       OutputApplicationMonotonic_ms="", CallbackToOutput_ms="")
            self.assertEqual(self.write([row], "timing")["errors"], [])
            row["OutputApplicationMonotonic_ms"] = "115"
            self.assertTrue(self.write([row], "timing")["errors"])
        for changes in ({"OutputStatus": "Invented"}, {"OutputTimingBoundary": "PhysicalDisplay"},
                        {"OutputApplicationMonotonic_ms": "109", "CallbackToOutput_ms": "9"},
                        {"CallbackToOutput_ms": "14"}, {"CallbackToOutput_ms": ""},
                        {"OutputApplicationMonotonic_ms": ""}, {"TimestampSource": "SensorExposure"},
                        {"InputCameraFrameId": "1"}, {"InputCameraFrameIdSource": "Unknown"},
                        {"InputCameraFrameIdSource": "MLCamera.ResultExtras.FrameNumber"},
                        {"ExpensiveDiagnosticsEnabled": "2"}):
            with self.subTest(changes=changes):
                self.assertTrue(self.write([dict(self.row("timing"), **changes)], "timing")["errors"])
        row = dict(self.row("timing"), InputCameraFrameId="42",
                   InputCameraFrameIdSource="MLCamera.ResultExtras.FrameNumber", OutputFrameId="15")
        self.assertEqual(self.write([row], "timing")["errors"], [])
        # A real application clock may exist when callback-to-result was not measurable.
        row.update(InputCallbackMonotonic_ms="", CaptureToResult_ms="", CallbackToOutput_ms="")
        self.assertEqual(self.write([row], "timing")["errors"], [])

    def test_current_stage_row_with_feedback_metadata(self):
        row = self.row("timing")
        for field in audit_csv.SCHEMA["timing"]["columns"][16:]:
            row[field] = ""
        row.update(OutputStatus="NotMeasured", OutputApplicationMonotonic_ms="", CallbackToOutput_ms="")
        report = self.write([row], "timing", footer=True)
        self.assertEqual(report["errors"], [])
        self.assertTrue(report["warnings"])
        # Lifetime counters must not reset through rows whose RunId is omitted.
        first = dict(self.row("timing"), DiscardedResults="5")
        row["DiscardedResults"] = "4"
        self.assertTrue(self.write([first, row], "timing")["errors"])
        self.assertTrue(self.write([dict(row, DiscardedResults="6"), first], "timing")["errors"])

    def test_track_age_only_present_with_tracks_and_flags(self):
        self.assertTrue(self.write([dict(self.row(), MaxDetectorConfirmationAge_ms="0")])["errors"])
        self.assertTrue(self.write([dict(self.row(), ActiveTracks="1")])["errors"])
        row = dict(self.row(), ActiveTracks="1", MaxDetectorConfirmationAge_ms="0.25")
        self.assertEqual(self.write([row])["errors"], [])
        for changes in ({"ExpensiveDiagnosticsEnabled": "2"}, {"TrackerDiagnosticsEnabled": "2"},
                        {"PointSelectionMode": "Unknown"}, {"DiagnosticsSchema": "galvat.tracking_diagnostics.v1"}):
            self.assertTrue(self.write([dict(self.row(), **changes)])["errors"])

    def test_lifetime_discard_counters_survive_session_reset(self):
        for field in audit_csv.VARIANTS["timing"]["run_monotonic_fields"]:
            first = dict(self.row("timing"), **{field: "5"})
            second = dict(first, SessionId="2", CompletedDetections="0", **{field: "4"})
            self.assertTrue(self.write([first, second], "timing")["errors"])
            self.assertEqual(self.write([first, dict(second, RunId="new-run")], "timing")["errors"], [])

    def test_separate_discard_events_and_reason_counter_scope(self):
        first = dict.fromkeys(audit_csv.VARIANTS["discard"]["columns"], "1")
        first.update(RunId="run", Reason="ReadbackFailed", DiscardedResults="5", ReasonCount="3",
                     EventMonotonic_ms="100", InputFrameId="-1")
        second = dict(first, SessionId="2", DiscardedResults="6", EventMonotonic_ms="101", ReasonCount="4")
        report = self.write([first, second], "discard")
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["schema"], "discard_current")
        self.assertTrue(report["current_complete"])
        for changes in ({"DiscardedResults": "4"}, {"ReasonCount": "2"},
                        {"EventMonotonic_ms": "99"}, {"Reason": ""}, {"InputFrameId": "-2"}):
            self.assertTrue(self.write([first, dict(second, **changes)], "discard")["errors"])
        other_reason = dict(second, Reason="SessionMismatch", ReasonCount="1")
        self.assertEqual(self.write([first, other_reason], "discard")["errors"], [])
        self.assertEqual(self.write([first, dict(first, RunId="other", DiscardedResults="1", ReasonCount="1")], "discard")["errors"], [])

    def test_enabled_diagnostics_and_quoted_json(self):
        row = self.row(diagnostics=True)
        self.assertEqual(self.write([row])["errors"], [])
        row["LiveFlow_ms"] = ""
        self.assertTrue(self.write([row])["errors"])
        row["LiveFlow_ms"] = "0"
        row["TrackGeometryJson"] = "[broken"
        self.assertTrue(self.write([row])["errors"])

    def test_timing27_footer_duplicate_inference_ids_and_stage_row(self):
        first = self.row("timing")
        second = dict(first, CompletedDetections="2", SessionElapsed_ms="20")
        stage = dict(first)
        for field in audit_csv.SCHEMA["timing"]["columns"][16:]:
            stage[field] = ""
        stage.update(OutputApplicationMonotonic_ms="", CallbackToOutput_ms="", OutputStatus="NotMeasured")
        report = self.write([first, stage, second], "timing", header=audit_csv.SCHEMA["timing"]["columns"], footer=True)
        self.assertEqual(len(audit_csv.SCHEMA["timing"]["columns"]), 27)
        self.assertEqual(report["errors"], [])
        self.assertEqual(report["rows"], 3)
        self.assertTrue(report["warnings"])

    def test_legacy35_is_warning_but_cli_fails_current_requirement(self):
        report = self.write([self.row()], header=audit_csv.SCHEMA["tracking"]["legacy_columns"])
        self.assertEqual(report["errors"], [])
        self.assertFalse(report["current_complete"])
        self.assertTrue(report["warnings"])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(audit_csv.main([str(self.path)]), 1)

    def test_truncated_extra_missing_and_duplicate_columns(self):
        columns = audit_csv.VARIANTS["tracking"]["columns"]
        for cells in (["0"] * 81, ["0"] * 83):
            self.assertTrue(self.write([cells])["errors"])
        for header in (columns[:-1], columns + ["Extra"], columns[:-1] + [columns[0]], columns[::-1]):
            row = dict(self.row(), Extra="0")
            self.assertTrue(self.write([row], header=header)["errors"])

    def test_padded_headers_and_real_style_csv_boundaries_preserve_provenance(self):
        for count in (27, 43):
            header = audit_csv.VARIANTS["timing"]["columns"][:count]
            padded = [" " + field + "  " for field in header]
            row = self.row("timing")
            row["Interval_ms"] = ""
            self.write([], "timing", header=padded)
            # Real timing output pads unquoted numbers, empty cells, and quoted enums.
            cells = []
            for field in header:
                cell = row[field]
                if field in {"TimestampSource", "WorkerBackend", "RunId", "OutputStatus",
                             "OutputTimingBoundary", "InputCameraFrameIdSource"}:
                    cell = '"' + cell.replace('"', '""') + '"'
                cells.append(" " + cell + " ")
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(",".join(cells) + "\n")
            before = self.path.read_bytes()
            report = audit_csv.audit(self.path)
            self.assertEqual(report["errors"], [])
            self.assertEqual(report["raw_header"], padded)
            self.assertEqual(report["normalized_header"], header)
            self.assertEqual(report["current_complete"], count == 43)
            self.assertEqual(report["sha256"], hashlib.sha256(before).hexdigest())
            self.assertEqual(self.path.read_bytes(), before)

    def test_duplicates_after_header_normalization_rejected(self):
        header = list(audit_csv.VARIANTS["timing"]["columns"])
        header[-1] = "  " + header[0] + " "
        report = self.write([], "timing", header=header)
        self.assertIn("duplicate headers after boundary whitespace normalization", report["errors"])

    def test_csv_padding_retains_quoted_spaces_escapes_and_multiline_values(self):
        source = '  "  arbitrary, ""quoted""\n value  "  , "next" \n"last" '
        reader = csv.reader(audit_csv.csv_boundary_lines(io.StringIO(source)),
                            strict=True, skipinitialspace=True)
        self.assertEqual(list(reader), [['  arbitrary, "quoted"\n value  ', 'next'], ['last']])
        # Padding inside quotes is part of an enum, not harmless CSV syntax.
        row = dict(self.row("timing"), OutputStatus=" MarkersApplied ")
        self.assertIn("line 2: unexpected OutputStatus", self.write([row], "timing")["errors"])
        for source in ('"unterminated\n', '"value" garbage,0\n', '"value"  "more",0\n'):
            with self.subTest(source=source), self.assertRaises(csv.Error):
                list(csv.reader(audit_csv.csv_boundary_lines(io.StringIO(source)),
                                strict=True, skipinitialspace=True))

    def test_nonfinite_negative_fractional_and_bad_clocks(self):
        for field, value in (("Tracker_ms", "NaN"), ("SessionElapsed_ms", "-1"),
                             ("CameraFrames", "inf"), ("CameraFrames", "0.5"),
                             ("YOLOCompletions", "-1")):
            self.assertTrue(self.write([dict(self.row(), **{field: value})])["errors"])
        for changes in ({"CompletionMonotonic_ms": "90"}, {"CaptureToResult_ms": "9"}):
            self.assertTrue(self.write([dict(self.row("timing"), **changes)], "timing")["errors"])

    def test_monotonic_counters_and_normal_new_sessions(self):
        for kind, counter in (("tracking", "CameraFrames"), ("timing", "CompletedDetections")):
            first = dict(self.row(kind), **{counter: "5"})
            reset = dict(first, SessionElapsed_ms="1", **{counter: "0"})
            self.assertTrue(self.write([first, reset], kind)["errors"])
            self.assertEqual(self.write([first, dict(reset, SessionId="2")], kind)["errors"], [])
            self.assertEqual(self.write([first, dict(reset, RunId="another-run")], kind)["errors"], [])

    def test_malformed_empty_and_data_after_footer(self):
        self.assertTrue(self.write([])["errors"])
        self.path.write_text('"unterminated\n', encoding="utf-8")
        self.assertTrue(audit_csv.audit(self.path)["errors"])
        self.write([self.row("timing")], "timing", footer=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write("bad\n")
        self.assertTrue(audit_csv.audit(self.path)["errors"])

    def test_cli_multiple_inputs_new_output_and_input_preservation(self):
        self.write([self.row()])
        before = self.path.read_bytes()
        output = Path(self.temp.name) / "report.json"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(audit_csv.main([str(self.path), str(self.path), "--output", str(output)]), 0)
            with self.assertRaises(SystemExit):
                audit_csv.main([str(self.path), "--output", str(self.path)])
        self.assertEqual(self.path.read_bytes(), before)
        payload = json.loads(output.read_text())
        self.assertEqual(len(payload["files"]), 2)
        self.assertFalse(payload["full_jhu_acceptance_measurement_complete"])


if __name__ == "__main__":
    unittest.main()
