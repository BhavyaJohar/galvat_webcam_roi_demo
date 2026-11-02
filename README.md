# GALVAT Webcam ROI Demo

## Project research and deployment status

This repository is the home for GALVAT model-building notebooks and ML optimization evidence. Unity and Magic Leap deployment code live in [jadinV/GALVAT_R61](https://github.com/jadinV/GALVAT_R61/tree/bhavya).

See the [verified project status and next tests](docs/PROJECT_STATUS.md), [historical notebook archive](notebooks/archive/README.md), and [Trial9 experiment reports](docs/experiments/trial9/). The current45-class Colab notebook exports are still needed; the archived Trial9 notebook is an older44-class template. The existing demo instructions below have not been updated to the latest Trial9 runtime.

Minimal local demo to visualize your trained GALVAT model on a computer webcam with a Region of Interest (ROI).

## What This Does
- Opens your webcam.
- Runs YOLO detection only inside a user-defined ROI.
- Draws detections back on the full frame.
- Lets you move/resize/select ROI in real time.
- Prints live terminal stats for ROI state, model timing, detections, and object position.

## 1) Setup

```bash
cd galvat_webcam_roi_demo
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 2) Add Your Model

Place your model file in one of these locations:
- `galvat_webcam_roi_demo/weights/best.pt` (recommended), or
- any path you pass with `--weights`.

## 3) Run (Mac)

```bash
python app.py --weights weights/best.pt --conf 0.30 --iou 0.50 --imgsz 640 --camera 1
```

With terminal stats slowed down to one update every 2 seconds:

```bash
python app.py --weights weights/best.pt --stats-interval 2.0
```

If your webcam is not camera 0:

```bash
python app.py --weights weights/best.pt --camera 1
```

## Controls
- `q`: quit
- `r`: reset ROI to center
- `w/a/s/d`: move ROI
- `-` / `=`: shrink / expand ROI
- mouse drag (left click + drag): draw a new ROI

## Useful Args
- `--conf 0.25` confidence threshold
- `--iou 0.45` NMS IoU threshold
- `--imgsz 640` inference size
- `--device cpu` force CPU (or `0` for first GPU if available)
- `--stats-interval 1.0` seconds between terminal stats updates

Example:

```bash
python app.py --weights weights/trial4.pt --conf 0.30 --iou 0.50 --imgsz 640
```

## Notes
- This is a visualization tool, not a production runtime.
- Detections are intentionally ROI-constrained by running inference on the ROI crop.
- Terminal output includes FPS, ROI dimensions and frame coverage, YOLO timing breakdown, per-detection confidence, bounding box position, normalized center, and ROI coverage.
- On exit, the app prints a short session summary with total frames, average FPS, total detections, best confidence, and class counts.
