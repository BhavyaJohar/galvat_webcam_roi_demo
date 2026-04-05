# GALVAT Webcam ROI Demo

Minimal local demo to visualize your trained GALVAT model on a computer webcam with a Region of Interest (ROI).

## What This Does
- Opens your webcam.
- Runs YOLO detection only inside a user-defined ROI.
- Draws detections back on the full frame.
- Lets you move/resize/select ROI in real time.

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

Example:

```bash
python app.py --weights weights/best.pt --conf 0.30 --iou 0.50 --imgsz 640
```

## Notes
- This is a visualization tool, not a production runtime.
- Detections are intentionally ROI-constrained by running inference on the ROI crop.
