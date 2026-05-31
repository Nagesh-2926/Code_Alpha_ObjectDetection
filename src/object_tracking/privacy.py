from __future__ import annotations

import cv2


def _ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def load_face_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    return cascade if not cascade.empty() else None


def _blur_region(frame, x1: int, y1: int, x2: int, y2: int, kernel_size: int) -> None:
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (_ensure_odd(kernel_size), _ensure_odd(kernel_size)), 0)


def apply_privacy_masking(frame, detections, privacy_config, face_cascade):
    if not privacy_config.enabled:
        return frame.copy()

    masked = frame.copy()
    target_classes = set(privacy_config.classes or [])
    kernel_size = max(3, _ensure_odd(privacy_config.blur_kernel_size))

    if privacy_config.mode == "person":
        for detection in detections:
            if target_classes and detection.class_name not in target_classes:
                continue
            _blur_region(masked, *detection.bbox, kernel_size=kernel_size)
        return masked

    if privacy_config.mode == "face" and face_cascade is not None:
        grayscale = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        person_detections = [item for item in detections if item.class_name == "person"]

        if person_detections:
            for detection in person_detections:
                x1, y1, x2, y2 = detection.bbox
                roi_gray = grayscale[y1:y2, x1:x2]
                roi_color = masked[y1:y2, x1:x2]
                faces = face_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
                for face_x, face_y, width, height in faces:
                    roi = roi_color[face_y : face_y + height, face_x : face_x + width]
                    roi_color[face_y : face_y + height, face_x : face_x + width] = cv2.GaussianBlur(
                        roi,
                        (kernel_size, kernel_size),
                        0,
                    )
        else:
            faces = face_cascade.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
            for face_x, face_y, width, height in faces:
                _blur_region(masked, face_x, face_y, face_x + width, face_y + height, kernel_size=kernel_size)

    return masked
