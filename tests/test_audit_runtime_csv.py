"""Synthetic validator fixtures only; none represent runtime measurements."""
import contextlib
import csv
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
        row = dict.fromkeys(schema["columns"], "0")
        row.update(RunId="fixture-run", SessionId="1", SessionElapsed_ms="10", InputFrameId="1")
        if kind == "tracking":
            row.update(Mode="YOLO_FLOW" if diagnostics else "YOLO_ONLY",
                       WorkerBackend="GPU", TrackerBackend="None", TrackIds="", TrackSources="",
                       DiagnosticsSchema="galvat.tracking_diagnostics.v1",
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
        return row

    def write(self, rows, kind="tracking", header=None, footer=False):
        header = header or audit_csv.SCHEMA[kind]["columns"]
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for row in rows:
                writer.writerow([row[c] for c in header] if isinstance(row, dict) else row)
            if footer:
                writer.writerow([])
                writer.writerow(["SESSION ENDED"])
        return audit_csv.audit(self.path)

    def test_baseline69_disabled_diagnostics(self):
        report = self.write([self.row()])
        self.assertEqual(len(audit_csv.SCHEMA["tracking"]["columns"]), 69)
        self.assertEqual(report["errors"], [])
        self.assertTrue(report["current_complete"])
        self.assertIn("capture-to-feedback", report["schema_gaps"])

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
        report = self.write([first, stage, second], "timing", footer=True)
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
        columns = audit_csv.SCHEMA["tracking"]["columns"]
        for cells in (["0"] * 68, ["0"] * 70):
            self.assertTrue(self.write([cells])["errors"])
        for header in (columns[:-1], columns + ["Extra"], columns[:-1] + [columns[0]], columns[::-1]):
            row = dict(self.row(), Extra="0")
            self.assertTrue(self.write([row], header=header)["errors"])

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
