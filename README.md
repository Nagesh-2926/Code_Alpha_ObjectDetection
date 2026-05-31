# Real-Time Object Detection and Tracking

This project is a complete beginner-friendly, resume-ready computer vision application that detects objects with YOLO, tracks them with stable IDs, draws live overlays, saves the processed video, and can optionally count line crossings.

## What this project does

- Reads from a webcam or video file
- Detects objects with YOLOv8
- Tracks objects with `BoT-SORT` or `ByteTrack`
- Draws bounding boxes, labels, IDs, trails, and FPS
- Saves the final annotated video to `artifacts/runs`
- Supports optional line-crossing counts for traffic or people analytics

## Architecture Overview

```text
Webcam / Video File
        ->
OpenCV VideoCapture
        ->
YOLOv8 Detection
        ->
BoT-SORT / ByteTrack
        ->
Custom Overlay Renderer
        ->
Live Window + Saved MP4 Output
```

## Free Technology Stack

- `Python` for application logic
- `OpenCV` for video I/O and drawing
- `Ultralytics YOLOv8` for object detection
- `BoT-SORT` or `ByteTrack` for tracking
- `PyTorch` as the deep learning runtime
- `PyYAML` for project configuration
- `pytest` for a small automated test layer

## Project Structure

```text
object_detection/
├── app.py
├── configs/
│   └── default.yaml
├── artifacts/
├── src/
│   └── object_tracking/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── geometry.py
│       ├── pipeline.py
│       └── visualization.py
├── tests/
│   └── test_geometry.py
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Setup

### 0. One-command Windows setup

```powershell
.\setup.ps1
```

### 1. Activate the virtual environment

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install PyTorch CPU wheels

```powershell
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install the project

```powershell
python -m pip install ".[dev]"
```

### Important Windows note

This project automatically uses ASCII-safe temporary paths when OpenCV reads or writes video files. That matters on some Windows setups, including folders with names like `文档`, because OpenCV file I/O can be unreliable with non-ASCII paths.

## Quick Start

### Run on webcam

```powershell
python app.py --source 0
```

### Run on a video file

```powershell
python app.py --source .\sample_video.mp4
```

### Run without opening the display window

```powershell
python app.py --source .\sample_video.mp4 --no-show
```

### Use ByteTrack instead of BoT-SORT

```powershell
python app.py --source 0 --tracker bytetrack.yaml
```

### Count objects crossing a custom line

```powershell
python app.py --source .\sample_video.mp4 --line-counting --line 200 300 1000 300
```

## Important CLI Options

- `--source`: `0` for webcam, or a video path
- `--model`: model weights such as `yolov8n.pt` or `yolov8s.pt`
- `--tracker`: `botsort.yaml` or `bytetrack.yaml`
- `--conf`: confidence threshold
- `--iou`: IoU threshold
- `--imgsz`: inference size
- `--device`: `cpu` or a CUDA device index when available
- `--classes`: class filter, for example `0,2,3`
- `--no-show`: skip the live preview window
- `--no-save`: skip writing the output video
- `--display-scale`: resize only the preview window

## Output

- Processed videos are saved into `artifacts/runs/`
- The output filename includes a timestamp so runs do not overwrite each other
- Press `Q` in the OpenCV window to stop early

## Docker

Build the image:

```powershell
docker build -t object-tracking-cpu .
```

Run a headless batch job:

```powershell
docker run --rm -v ${PWD}/artifacts:/app/artifacts -v ${PWD}/data:/data object-tracking-cpu
```

Place your input video at `data/input.mp4` before running the container, or change the Docker `CMD`.

## How tracking works

- YOLO finds objects in each frame
- The tracker compares motion and appearance between frames
- Each object gets a persistent ID like `ID 4`
- The trail lines show where each tracked object moved

## Troubleshooting

### The webcam does not open

- Close any other app using the camera
- Try `--source 1` if another camera index is active
- Confirm Windows camera permissions are enabled

### The app is slow

- Use the smaller `yolov8n.pt` model
- Lower the image size with `--imgsz 512`
- Run with `--display-scale 0.8`
- Disable the preview using `--no-show`

### I see a missing package error

- Activate `.venv`
- Re-run the install commands
- Check that `torch`, `opencv-python`, and `ultralytics` installed successfully

## Resume-Ready Project Ideas

- Vehicle counting at an entrance gate
- Person counting in a retail store
- Safety monitoring for helmets or vests
- Intrusion detection in a restricted area
- Traffic analytics with line crossing and zones

## Next Upgrade Ideas

- Polygon zone counting
- Speed estimation
- Streamlit or FastAPI dashboard
- Model export to ONNX
- Docker packaging for headless batch processing
