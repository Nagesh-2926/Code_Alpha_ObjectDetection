from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from .analytics import AnalyticsEngine
from .config import AppConfig
from .geometry import bbox_center, crossing_direction, point_line_side
from .pose import extract_pose_people
from .privacy import apply_privacy_masking, load_face_cascade
from .visualization import OverlayStats, draw_frame

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class Detection:
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    class_id: int
    class_name: str
    confidence: float
    track_id: int | None = None
    speed_kph: float | None = None
    zone_names: tuple[str, ...] = ()
    is_intrusion: bool = False


@dataclass(slots=True)
class RunSummary:
    source: str
    frames_processed: int
    average_fps: float
    output_path: str | None
    analytics_path: str | None
    tracker: str
    model_weights: str
    total_intrusions: int = 0


class LineCounter:
    def __init__(self, line_points: tuple[tuple[int, int], tuple[int, int]] | None) -> None:
        self.line_points = line_points
        self.previous_sides: dict[int, int] = {}
        self.counted_track_ids: set[int] = set()
        self.in_count = 0
        self.out_count = 0

    def update(self, detections: list[Detection]) -> None:
        if not self.line_points:
            return

        start, end = self.line_points
        active_track_ids: set[int] = set()

        for detection in detections:
            if detection.track_id is None:
                continue

            track_id = detection.track_id
            active_track_ids.add(track_id)
            current_side = point_line_side(detection.center, start, end)
            previous_side = self.previous_sides.get(track_id)

            if (
                previous_side is not None
                and track_id not in self.counted_track_ids
                and previous_side != current_side
            ):
                direction = crossing_direction(previous_side, current_side)
                if direction == "in":
                    self.in_count += 1
                    self.counted_track_ids.add(track_id)
                elif direction == "out":
                    self.out_count += 1
                    self.counted_track_ids.add(track_id)

            self.previous_sides[track_id] = current_side

        stale_ids = set(self.previous_sides) - active_track_ids
        for track_id in stale_ids:
            self.previous_sides.pop(track_id, None)


def _is_ascii_safe(path: Path) -> bool:
    try:
        str(path).encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def _prepare_temp_ascii_path(original_path: Path) -> Path:
    safe_name = "".join(character for character in original_path.name if ord(character) < 128).strip()
    if not safe_name:
        safe_name = f"video_{uuid.uuid4().hex}{original_path.suffix or '.mp4'}"
    elif "." not in safe_name and original_path.suffix:
        safe_name = f"{safe_name}{original_path.suffix}"

    temp_dir = Path(tempfile.gettempdir()) / "object_tracking_ascii_io"
    temp_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = Path(safe_name).stem
    safe_suffix = Path(safe_name).suffix or original_path.suffix or ".mp4"
    return temp_dir / f"{safe_stem}_{uuid.uuid4().hex[:8]}{safe_suffix}"


def _prepare_capture_source(source: str | int) -> tuple[str | int, Path | None]:
    if isinstance(source, int):
        return source, None

    source_path = Path(source)
    if source_path.exists() and not _is_ascii_safe(source_path):
        temp_path = _prepare_temp_ascii_path(source_path)
        shutil.copy2(source_path, temp_path)
        LOGGER.info("Copied input video to ASCII-safe temp path: %s", temp_path)
        return str(temp_path), temp_path

    return str(source), None


def _prepare_writer_target(output_path: Path) -> tuple[Path, Path | None]:
    if _is_ascii_safe(output_path):
        return output_path, None

    temp_path = _prepare_temp_ascii_path(output_path)
    LOGGER.info("Using ASCII-safe temp output path: %s", temp_path)
    return temp_path, temp_path


def _resolve_output_path(config: AppConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{config.output.video_name_prefix}_{timestamp}.mp4"


def _resolve_analytics_path(config: AppConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{config.output.analytics_name_prefix}_{timestamp}.json"


def _build_video_writer(capture: cv2.VideoCapture, output_path: Path) -> cv2.VideoWriter:
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


def _extract_detections(result) -> list[Detection]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.int().cpu().tolist()
    classes = boxes.cls.int().cpu().tolist()
    confidences = boxes.conf.float().cpu().tolist()
    track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)

    detections: list[Detection] = []
    names = result.names
    for bbox, class_id, confidence, track_id in zip(xyxy, classes, confidences, track_ids):
        bbox_tuple = tuple(int(value) for value in bbox)
        class_name = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
        detections.append(
            Detection(
                bbox=bbox_tuple,
                center=bbox_center(bbox_tuple),
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                track_id=track_id,
            )
        )
    return detections


def _resize_for_display(frame, scale: float):
    if scale == 1.0:
        return frame
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (width, height))


def _write_analytics_summary(
    analytics_path: Path,
    summary: RunSummary,
    line_counter: LineCounter,
    analytics_engine: AnalyticsEngine,
    peak_class_counts: dict[str, int],
) -> None:
    report = analytics_engine.build_report()
    payload = {
        "source": summary.source,
        "frames_processed": summary.frames_processed,
        "average_fps": summary.average_fps,
        "output_path": summary.output_path,
        "tracker": summary.tracker,
        "model_weights": summary.model_weights,
        "line_counts": {
            "in": line_counter.in_count,
            "out": line_counter.out_count,
        },
        "peak_class_counts": peak_class_counts,
        "analytics": report,
    }
    analytics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_tracking(config: AppConfig) -> RunSummary:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    LOGGER.info("Loading YOLO model: %s", config.model.weights)
    detection_model = YOLO(config.model.weights)
    pose_model = YOLO(config.features.pose.weights) if config.features.pose.enabled else None

    capture_source, temp_input_path = _prepare_capture_source(config.source)
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {config.source}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    output_path: Path | None = _resolve_output_path(config) if config.output.save_video else None
    analytics_path: Path | None = _resolve_analytics_path(config) if config.output.save_analytics_json else None
    writer_path: Path | None = None
    temp_output_path: Path | None = None
    if output_path:
        writer_path, temp_output_path = _prepare_writer_target(output_path)
    writer = _build_video_writer(capture, writer_path) if writer_path else None

    track_history: dict[int, deque[tuple[int, int]]] = {}
    analytics_engine = AnalyticsEngine(config.features.analytics)
    line_counter = LineCounter(config.features.analytics.count_line if config.features.analytics.enable_line_counting else None)
    face_cascade = load_face_cascade() if config.features.privacy.enabled and config.features.privacy.mode == "face" else None
    peak_class_counts: Counter[str] = Counter()

    frame_index = 0
    total_processing_time = 0.0
    total_intrusions = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            frame_index += 1
            loop_start = time.perf_counter()
            detection_results = detection_model.track(
                source=frame,
                persist=True,
                tracker=config.model.tracker,
                conf=config.model.confidence,
                iou=config.model.iou,
                imgsz=config.model.image_size,
                device=config.model.device,
                classes=config.model.classes,
                verbose=False,
            )
            detections = _extract_detections(detection_results[0])
            tracked_ids = [item.track_id for item in detections if item.track_id is not None]

            for detection in detections:
                if detection.track_id is None:
                    continue
                history = track_history.setdefault(
                    detection.track_id,
                    deque(maxlen=config.features.trail_length),
                )
                history.append(detection.center)

            active_track_ids = set(tracked_ids)
            stale_track_ids = set(track_history) - active_track_ids
            for track_id in stale_track_ids:
                track_history.pop(track_id, None)

            if config.features.analytics.enable_line_counting:
                line_counter.update(detections)

            pose_people = []
            if pose_model is not None:
                pose_results = pose_model.predict(
                    source=frame,
                    conf=config.features.pose.confidence,
                    imgsz=config.model.image_size,
                    device=config.model.device,
                    classes=config.features.pose.classes,
                    verbose=False,
                )
                pose_people = extract_pose_people(pose_results[0])

            elapsed = time.perf_counter() - loop_start
            total_processing_time += elapsed
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            analytics_fps = source_fps if source_fps > 0 else fps
            analytics_snapshot = analytics_engine.update(detections, frame_index=frame_index, fps=analytics_fps)
            total_intrusions = analytics_snapshot.total_intrusions
            for class_name, count in analytics_snapshot.class_counts.items():
                peak_class_counts[class_name] = max(peak_class_counts[class_name], count)

            privacy_frame = apply_privacy_masking(frame, detections, config.features.privacy, face_cascade)
            annotated = draw_frame(
                frame=privacy_frame,
                detections=detections,
                track_history=track_history,
                features=config.features,
                stats=OverlayStats(
                    fps=fps,
                    frame_index=frame_index,
                    total_detections=len(detections),
                    tracked_objects=len(active_track_ids),
                    unique_classes=len({item.class_name for item in detections}),
                    in_count=line_counter.in_count,
                    out_count=line_counter.out_count,
                ),
                analytics=analytics_snapshot,
                pose_people=pose_people,
            )

            if writer:
                writer.write(annotated)

            if config.output.show_window:
                display_frame = _resize_for_display(annotated, config.output.display_scale)
                cv2.imshow(config.output.window_name, display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    LOGGER.info("Stopping early because the user pressed Q.")
                    break

        average_fps = frame_index / total_processing_time if total_processing_time > 0 else 0.0
        LOGGER.info("Finished processing %s frames. Average FPS: %.2f", frame_index, average_fps)
        if output_path:
            LOGGER.info("Saved output video to %s", output_path)

        summary = RunSummary(
            source=str(config.source),
            frames_processed=frame_index,
            average_fps=average_fps,
            output_path=str(output_path) if output_path else None,
            analytics_path=str(analytics_path) if analytics_path else None,
            tracker=config.model.tracker,
            model_weights=config.model.weights,
            total_intrusions=total_intrusions,
        )
        if analytics_path:
            _write_analytics_summary(analytics_path, summary, line_counter, analytics_engine, dict(peak_class_counts))
            LOGGER.info("Saved analytics summary to %s", analytics_path)
        return summary
    finally:
        capture.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        if temp_output_path and output_path and temp_output_path.exists():
            shutil.move(temp_output_path, output_path)
        if temp_input_path and temp_input_path.exists():
            temp_input_path.unlink(missing_ok=True)
