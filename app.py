import argparse
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, cast

import cv2
import numpy as np
from ultralytics import YOLO


WINDOW_NAME = "GALVAT ROI Demo"
MIN_ROI_SIZE = 48


@dataclass
class ROI:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h


@dataclass
class DetectionStats:
    label: str
    confidence: float
    local_box: tuple[int, int, int, int]
    global_box: tuple[int, int, int, int]
    center_global: tuple[float, float]
    center_normalized: tuple[float, float]
    area: int
    roi_coverage: float


class TerminalStatsReporter:
    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self.last_print_t = 0.0
        self.frame_count = 0
        self.total_detections = 0
        self.class_counts: Counter[str] = Counter()
        self.best_confidence = 0.0
        self.started_t = time.time()

    def maybe_print(
        self,
        fps: float,
        roi: ROI,
        frame_size: tuple[int, int],
        detection_stats: list[DetectionStats],
        speed_ms: dict | None,
    ) -> None:
        self.frame_count += 1
        self.total_detections += len(detection_stats)
        for stat in detection_stats:
            self.class_counts[stat.label] += 1
            self.best_confidence = max(self.best_confidence, stat.confidence)

        now_t = time.time()
        if now_t - self.last_print_t < self.interval_s:
            return
        self.last_print_t = now_t

        frame_w, frame_h = frame_size
        roi_area = roi.w * roi.h
        roi_fraction = roi_area / max(1, frame_w * frame_h)

        print(
            "[STATS] "
            f"fps={fps:.1f} frame={frame_w}x{frame_h} "
            f"roi=(x={roi.x}, y={roi.y}, w={roi.w}, h={roi.h}, area={roi_area}, frame_pct={roi_fraction * 100:.1f}%) "
            f"detections={len(detection_stats)}"
        )

        if speed_ms:
            preprocess = speed_ms.get("preprocess", 0.0)
            inference = speed_ms.get("inference", 0.0)
            postprocess = speed_ms.get("postprocess", 0.0)
            total = preprocess + inference + postprocess
            print(
                "[TIMING] "
                f"preprocess={preprocess:.1f}ms inference={inference:.1f}ms "
                f"postprocess={postprocess:.1f}ms total={total:.1f}ms"
            )

        if not detection_stats:
            print("[DETECTIONS] none")
            return

        for idx, stat in enumerate(
            sorted(detection_stats, key=lambda item: item.confidence, reverse=True),
            start=1,
        ):
            gx1, gy1, gx2, gy2 = stat.global_box
            cx, cy = stat.center_global
            nx, ny = stat.center_normalized
            print(
                f"[DETECTION {idx}] "
                f"label={stat.label} conf={stat.confidence:.3f} "
                f"box_global=({gx1},{gy1})-({gx2},{gy2}) "
                f"center=({cx:.1f},{cy:.1f}) normalized=({nx:.3f},{ny:.3f}) "
                f"area={stat.area} roi_pct={stat.roi_coverage * 100:.1f}%"
            )

    def print_summary(self) -> None:
        elapsed = max(1e-6, time.time() - self.started_t)
        avg_det_per_frame = self.total_detections / max(1, self.frame_count)
        class_summary = ", ".join(
            f"{label}={count}" for label, count in self.class_counts.most_common()
        ) or "none"
        print(
            "[SUMMARY] "
            f"elapsed={elapsed:.1f}s frames={self.frame_count} "
            f"avg_fps={self.frame_count / elapsed:.1f} total_detections={self.total_detections} "
            f"avg_detections_per_frame={avg_det_per_frame:.2f} "
            f"best_confidence={self.best_confidence:.3f} "
            f"class_counts={class_summary}"
        )


class ROIController:
    def __init__(self, frame_w: int, frame_h: int) -> None:
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.roi = self._default_roi()

        self.dragging = False
        self.drag_start = (0, 0)
        self.drag_current = (0, 0)

    def _default_roi(self) -> ROI:
        w = int(self.frame_w * 0.6)
        h = int(self.frame_h * 0.6)
        x = (self.frame_w - w) // 2
        y = (self.frame_h - h) // 2
        return ROI(x=x, y=y, w=w, h=h)

    def reset(self) -> None:
        self.roi = self._default_roi()

    def update_frame_size(self, frame_w: int, frame_h: int) -> None:
        if frame_w == self.frame_w and frame_h == self.frame_h:
            return
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.roi = clamp_roi(self.roi, frame_w, frame_h)

    def move(self, dx: int, dy: int) -> None:
        moved = ROI(self.roi.x + dx, self.roi.y + dy, self.roi.w, self.roi.h)
        self.roi = clamp_roi(moved, self.frame_w, self.frame_h)

    def resize(self, delta: int) -> None:
        new_w = max(MIN_ROI_SIZE, self.roi.w + delta)
        new_h = max(MIN_ROI_SIZE, self.roi.h + delta)
        cx = self.roi.x + self.roi.w // 2
        cy = self.roi.y + self.roi.h // 2
        resized = ROI(cx - new_w // 2, cy - new_h // 2, new_w, new_h)
        self.roi = clamp_roi(resized, self.frame_w, self.frame_h)

    def start_drag(self, x: int, y: int) -> None:
        self.dragging = True
        self.drag_start = (x, y)
        self.drag_current = (x, y)

    def drag(self, x: int, y: int) -> None:
        if self.dragging:
            self.drag_current = (x, y)

    def end_drag(self, x: int, y: int) -> None:
        if not self.dragging:
            return
        self.drag_current = (x, y)
        self.dragging = False

        x1 = min(self.drag_start[0], self.drag_current[0])
        y1 = min(self.drag_start[1], self.drag_current[1])
        x2 = max(self.drag_start[0], self.drag_current[0])
        y2 = max(self.drag_start[1], self.drag_current[1])

        w = x2 - x1
        h = y2 - y1
        if w < MIN_ROI_SIZE or h < MIN_ROI_SIZE:
            return
        self.roi = clamp_roi(ROI(x1, y1, w, h), self.frame_w, self.frame_h)

    def draw_overlay(self, frame) -> None:
        if not self.dragging:
            return
        x1 = min(self.drag_start[0], self.drag_current[0])
        y1 = min(self.drag_start[1], self.drag_current[1])
        x2 = max(self.drag_start[0], self.drag_current[0])
        y2 = max(self.drag_start[1], self.drag_current[1])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(
            frame,
            "Drawing ROI...",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def clamp_roi(roi: ROI, frame_w: int, frame_h: int) -> ROI:
    w = min(max(roi.w, MIN_ROI_SIZE), frame_w)
    h = min(max(roi.h, MIN_ROI_SIZE), frame_h)

    x = max(0, min(roi.x, frame_w - w))
    y = max(0, min(roi.y, frame_h - h))
    return ROI(x=x, y=y, w=w, h=h)


def draw_help(frame, fps: float, count: int, roi: ROI) -> None:
    help_lines = [
        f"FPS: {fps:.1f}  Detections: {count}",
        f"ROI: x={roi.x}, y={roi.y}, w={roi.w}, h={roi.h}",
        "Controls: q quit | r reset ROI | w/a/s/d move ROI | -/= resize ROI | mouse drag new ROI",
    ]
    y = 24
    for line in help_lines:
        cv2.putText(
            frame,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        y += 26


def mouse_callback(event, x, y, _flags, param) -> None:
    controller: ROIController = param
    if event == cv2.EVENT_LBUTTONDOWN:
        controller.start_drag(x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        controller.drag(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        controller.end_drag(x, y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GALVAT webcam ROI detector demo")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Path to YOLO .pt weights")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", type=str, default="", help="Inference device: '', 'cpu', or '0'")
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=1.0,
        help="Seconds between terminal stats updates",
    )
    return parser.parse_args()


def to_numpy_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "cpu") and hasattr(value, "numpy"):
        return cast(np.ndarray, value.cpu().numpy())
    if hasattr(value, "numpy"):
        return cast(np.ndarray, value.numpy())
    return np.asarray(value)


def _open_camera_once(index: int):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None, None

    ok, frame = cap.read()
    if not ok:
        cap.release()
        return None, None

    return cap, frame


def open_camera_with_fallback(requested_index: int, max_index_to_scan: int = 5):
    cap, frame = _open_camera_once(requested_index)
    if cap is not None:
        return cap, frame, requested_index

    available_indices = []
    for idx in range(max_index_to_scan + 1):
        probe, _ = _open_camera_once(idx)
        if probe is not None:
            available_indices.append(idx)
            probe.release()

    if available_indices:
        fallback_index = available_indices[0]
        cap, frame = _open_camera_once(fallback_index)
        if cap is not None:
            print(
                f"[WARN] Camera index {requested_index} could not be opened. "
                f"Falling back to camera index {fallback_index}."
            )
            return cap, frame, fallback_index

    scanned = ", ".join(str(i) for i in range(max_index_to_scan + 1))
    raise RuntimeError(
        f"Could not open camera index {requested_index}. "
        f"No working cameras found in scanned indices [{scanned}]. "
        "Try --camera 0 and ensure camera permissions are enabled for Terminal/Python on macOS."
    )


def main() -> None:
    args = parse_args()

    model = YOLO(args.weights)
    reporter = TerminalStatsReporter(interval_s=max(0.1, args.stats_interval))

    cap, frame, active_camera = open_camera_with_fallback(args.camera)
    print(f"[INFO] Using camera index {active_camera}")
    print(f"[INFO] Terminal stats enabled every {reporter.interval_s:.1f}s")
    if frame is None:
        raise RuntimeError("Could not read initial frame from camera")

    h, w = frame.shape[:2]
    controller = ROIController(frame_w=w, frame_h=h)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, controller)

    prev_t = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        h, w = frame.shape[:2]
        controller.update_frame_size(frame_w=w, frame_h=h)
        roi = controller.roi

        crop = frame[roi.y : roi.y2, roi.x : roi.x2]
        result = model.predict(
            source=crop,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        det_count = 0
        detection_stats: list[DetectionStats] = []
        if result.boxes is not None and len(result.boxes) > 0:
            names = result.names
            xyxy = to_numpy_array(result.boxes.xyxy)
            confs = to_numpy_array(result.boxes.conf)
            classes = to_numpy_array(result.boxes.cls).astype(int)

            for box, conf, cls_id in zip(xyxy, confs, classes):
                det_count += 1
                x1, y1, x2, y2 = [int(v) for v in box]
                gx1, gy1 = x1 + roi.x, y1 + roi.y
                gx2, gy2 = x2 + roi.x, y2 + roi.y
                area = max(0, x2 - x1) * max(0, y2 - y1)
                center_x = gx1 + (gx2 - gx1) / 2.0
                center_y = gy1 + (gy2 - gy1) / 2.0
                label = names.get(cls_id, str(cls_id))

                detection_stats.append(
                    DetectionStats(
                        label=label,
                        confidence=float(conf),
                        local_box=(x1, y1, x2, y2),
                        global_box=(gx1, gy1, gx2, gy2),
                        center_global=(center_x, center_y),
                        center_normalized=(center_x / max(1, w), center_y / max(1, h)),
                        area=area,
                        roi_coverage=area / max(1, roi.w * roi.h),
                    )
                )

                cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
                label = f"{label} {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (gx1, max(18, gy1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        # Draw active ROI
        cv2.rectangle(frame, (roi.x, roi.y), (roi.x2, roi.y2), (255, 180, 0), 2)
        cv2.putText(
            frame,
            "Active ROI",
            (roi.x, max(22, roi.y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )

        controller.draw_overlay(frame)

        now_t = time.time()
        fps = 1.0 / max(1e-6, now_t - prev_t)
        prev_t = now_t
        draw_help(frame, fps=fps, count=det_count, roi=roi)
        reporter.maybe_print(
            fps=fps,
            roi=roi,
            frame_size=(w, h),
            detection_stats=detection_stats,
            speed_ms=result.speed,
        )

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("r"):
            controller.reset()
        if key == ord("w"):
            controller.move(0, -12)
        if key == ord("s"):
            controller.move(0, 12)
        if key == ord("a"):
            controller.move(-12, 0)
        if key == ord("d"):
            controller.move(12, 0)
        if key in (ord("-"), ord("_")):
            controller.resize(-24)
        if key in (ord("="), ord("+")):
            controller.resize(24)

    cap.release()
    cv2.destroyAllWindows()
    reporter.print_summary()


if __name__ == "__main__":
    main()
