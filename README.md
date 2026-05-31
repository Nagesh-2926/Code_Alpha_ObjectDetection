# CodeAlpha Object Detection Pro

Advanced real-time object detection, tracking, analytics, pose estimation, privacy protection, and deployment toolkit built on YOLOv8, OpenCV, and Ultralytics tracking.

This project has evolved from a basic detection-and-tracking demo into a more complete computer vision application that can serve as an internship showcase, portfolio project, or foundation for production-style experimentation.

## Core Capabilities

- Real-time object detection for common living and non-living object classes supported by YOLOv8
- Persistent multi-object tracking with `BoT-SORT` or `ByteTrack`
- Live bounding boxes, labels, centroids, trails, and FPS overlays
- Polygon-based zone monitoring and occupancy counting
- Line-crossing analytics
- Intrusion detection for restricted zones
- Speed estimation based on motion calibration
- Privacy masking with face blur or full-person blur
- Human pose estimation with body landmark visualization
- Structured analytics export as JSON
- Streamlit interface for upload-and-process workflows
- FastAPI interface for service-style video processing
- ONNX export utility for deployment workflows

## System Architecture

```text
Camera / Reference Video
        ->
Frame Capture (OpenCV)
        ->
YOLOv8 Detection + Tracking
        ->
Analytics Engine
  - line counting
  - zone counting
  - intrusion alerts
  - speed estimation
        ->
Optional Privacy Masking
        ->
Optional Pose Estimation
        ->
Visualization Layer
        ->
Preview Window + Output Video + Analytics JSON
```

## Feature Breakdown

### 1. Detection and Tracking

The pipeline uses a YOLOv8 detection model with Ultralytics tracking to maintain stable IDs across frames. This allows the system to move beyond frame-by-frame detection and reason about object movement, occupancy, and behavior.

### 2. Zone-Based Analytics

Custom polygon zones can be defined in configuration. Each zone can:

- count current tracked objects
- record cumulative entries
- restrict counting to selected classes
- act as a restricted area for intrusion alerts

### 3. Intrusion Detection

Restricted zones can raise alerts when specific classes, such as `person`, enter protected regions. Cooldown logic prevents the same track from triggering duplicate alerts every frame.

### 4. Speed Estimation

Speed is estimated from tracked motion using a configurable `pixels_per_meter` calibration value. This is especially useful for traffic-style analytics or monitored movement zones.

### 5. Privacy Masking

Two masking modes are available:

- `face`: blur detected faces, typically inside person detections
- `person`: blur the full bounding box of selected classes

### 6. Pose Estimation

An optional YOLO pose model can overlay body landmarks and skeleton connections. This upgrades the project from plain object detection to richer human-activity analysis and body-part visualization.

### 7. Interfaces and Deployment

The project now includes:

- a CLI pipeline for direct terminal usage
- a Streamlit app for interactive usage
- a FastAPI app for service-based processing
- an ONNX export path for deployment optimization

## Repository Layout

```text
CodeAlpha_Object_Detection/
|-- app.py
|-- api.py
|-- streamlit_app.py
|-- configs/
|   |-- default.yaml
|   `-- pro_demo.yaml
|-- src/
|   `-- object_tracking/
|       |-- __init__.py
|       |-- analytics.py
|       |-- cli.py
|       |-- config.py
|       |-- export.py
|       |-- geometry.py
|       |-- pipeline.py
|       |-- pose.py
|       |-- privacy.py
|       `-- visualization.py
|-- tests/
|   |-- test_analytics.py
|   |-- test_api.py
|   `-- test_geometry.py
|-- .vscode/
|-- artifacts/
|-- Dockerfile
|-- pyproject.toml
|-- requirements.txt
|-- setup.ps1
`-- README.md
```

## Module Responsibilities

- [app.py](D:\Downloads\CodeAlpha_Object_Detection\app.py:1)
  Root launcher for the CLI tracking pipeline.

- [api.py](D:\Downloads\CodeAlpha_Object_Detection\api.py:1)
  FastAPI service for uploaded-video processing.

- [streamlit_app.py](D:\Downloads\CodeAlpha_Object_Detection\streamlit_app.py:1)
  Interactive web dashboard for upload, configuration, and result review.

- [configs/default.yaml](D:\Downloads\CodeAlpha_Object_Detection\configs\default.yaml:1)
  Baseline project configuration.

- [configs/pro_demo.yaml](D:\Downloads\CodeAlpha_Object_Detection\configs\pro_demo.yaml:1)
  Example pro-mode configuration with multiple advanced features enabled.

- [analytics.py](D:\Downloads\CodeAlpha_Object_Detection\src\object_tracking\analytics.py:1)
  Zone counting, intrusion alerts, class counts, and speed estimation.

- [config.py](D:\Downloads\CodeAlpha_Object_Detection\src\object_tracking\config.py:1)
  Strongly structured configuration model and YAML loading logic.

- [pipeline.py](D:\Downloads\CodeAlpha_Object_Detection\src\object_tracking\pipeline.py:1)
  Main runtime orchestration for capture, inference, analytics, overlays, output writing, and summary export.

- [pose.py](D:\Downloads\CodeAlpha_Object_Detection\src\object_tracking\pose.py:1)
  Pose-model result parsing and body landmark definitions.

- [privacy.py](D:\Downloads\CodeAlpha_Object_Detection\src\object_tracking\privacy.py:1)
  Face blur and person blur logic.

- [visualization.py](D:\Downloads\CodeAlpha_Object_Detection\src\object_tracking\visualization.py:1)
  Rendering for boxes, zones, alerts, trails, speed, and pose annotations.

- [export.py](D:\Downloads\CodeAlpha_Object_Detection\src\object_tracking\export.py:1)
  ONNX export command for deployment-oriented workflows.

## Installation

### Recommended setup

From the project root:

```powershell
.\setup.ps1
```

### Manual setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[dev]"
```

## Running the Project

### 1. Live webcam

```powershell
cd "D:\Downloads\CodeAlpha_Object_Detection"
.\.venv\Scripts\python.exe app.py --source 0
```

### 2. Reference video

```powershell
cd "D:\Downloads\CodeAlpha_Object_Detection"
.\.venv\Scripts\python.exe app.py --source "D:\Downloads\WhatsApp Video 2026-05-31 at 21.00.28.mp4"
```

### 3. Advanced CLI example

```powershell
.\.venv\Scripts\python.exe app.py --source "D:\Downloads\sample.mp4" --tracker bytetrack.yaml --zone-counting --intrusion --speed --pixels-per-meter 12 --pose --privacy-mode face
```

### 4. Run the Streamlit app

```powershell
cd "D:\Downloads\CodeAlpha_Object_Detection"
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

### 5. Run the FastAPI app

```powershell
cd "D:\Downloads\CodeAlpha_Object_Detection"
.\.venv\Scripts\python.exe -m uvicorn api:app --reload
```

### 6. Export YOLO to ONNX

```powershell
cd "D:\Downloads\CodeAlpha_Object_Detection"
.\.venv\Scripts\export-yolo-onnx.exe --weights yolov8n.pt --imgsz 640 --simplify
```

## Configuration Model

Main configuration groups:

- `model`
  Detection weights, confidence, image size, device, and tracker selection

- `output`
  Preview, video saving, analytics JSON saving, and output naming

- `features.analytics`
  Line counting, zone counting, intrusion detection, and speed estimation

- `features.privacy`
  Privacy masking configuration

- `features.pose`
  Pose model weights, confidence, and label display

## Output Artifacts

The project can produce:

- annotated output videos in `artifacts/runs/`
- analytics JSON summaries in `artifacts/runs/`
- real-time visual overlays in the preview window

## VS Code Support

The repository includes a ready-to-use VS Code launch configuration in:

- [launch.json](D:\Downloads\CodeAlpha_Object_Detection\.vscode\launch.json:1)
- [settings.json](D:\Downloads\CodeAlpha_Object_Detection\.vscode\settings.json:1)

You can press `F5` in VS Code and launch the configured video run directly.

## Validation

Current validation includes:

- syntax compilation for the upgraded modules
- unit tests for geometry, analytics, and FastAPI flows
- successful processing of the WhatsApp reference video with output video generation
- successful smoke processing of `artifacts/demo_input.mp4` with synchronized video and analytics output naming

Run tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Troubleshooting

### Slow inference on CPU

- Use `yolov8n.pt`
- keep `imgsz` moderate, such as `640` or `512`
- disable pose when not needed
- disable preview using `--no-show` for offline runs

### Privacy masking not triggering

- use `--privacy-mode face` for face blur
- use `--privacy-mode person` to blur the full person box
- ensure the source has visible faces if using face mode

### Speed values look unrealistic

- adjust `pixels_per_meter` in config or CLI
- speed estimation depends on scene calibration and camera perspective

### Zone counts look wrong

- update zone polygons in the YAML config to match the real frame layout
- restrict `count_classes` per zone to reduce noise

## Project History

### Version 1.0

- basic YOLOv8 object detection
- tracking IDs
- line counting
- output video saving

### Version 2.0

- zone-based counting
- intrusion detection
- speed estimation
- privacy masking
- pose estimation
- Streamlit interface
- FastAPI interface
- ONNX export support
- richer configuration model and analytics output

### Version 2.1

- synchronized run IDs for output video and analytics JSON artifacts
- stronger FastAPI input validation and error handling
- stronger Streamlit input validation and failure reporting
- API test coverage for health and upload-processing flows

## Future Extensions

Possible next upgrades:

- polygon heatmaps
- multi-camera fusion
- ReID-based long-term tracking
- database-backed event storage
- alert notification integrations
- TensorRT or OpenVINO deployment optimization
