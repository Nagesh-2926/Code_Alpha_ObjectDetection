from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from .config import AnalyticsConfig, ZoneConfig
from .geometry import euclidean_distance, point_in_polygon


@dataclass(slots=True)
class ZoneSnapshot:
    name: str
    current_count: int
    cumulative_entries: int
    class_counts: dict[str, int]
    restricted: bool = False


@dataclass(slots=True)
class AlertEvent:
    frame_index: int
    zone_name: str
    track_id: int
    class_name: str
    message: str


@dataclass(slots=True)
class FrameAnalyticsSnapshot:
    zone_stats: list[ZoneSnapshot] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)
    active_alerts: list[str] = field(default_factory=list)
    total_intrusions: int = 0
    average_speed_kph: float = 0.0
    max_speed_kph: float = 0.0


class SpeedEstimator:
    def __init__(self, config) -> None:
        self.config = config
        self.previous_centers: dict[int, tuple[int, int]] = {}
        self.speed_history: dict[int, deque[float]] = {}

    def update(self, detections, fps: float) -> dict[int, float]:
        if not self.config.enabled or fps <= 0 or self.config.pixels_per_meter <= 0:
            return {}

        detections_with_ids = [item for item in detections if item.track_id is not None]
        active_ids = {item.track_id for item in detections_with_ids}
        speed_map: dict[int, float] = {}
        seconds_per_frame = 1.0 / fps

        for detection in detections_with_ids:
            if self.config.classes and detection.class_name not in self.config.classes:
                continue

            previous_center = self.previous_centers.get(detection.track_id)
            self.previous_centers[detection.track_id] = detection.center

            if previous_center is None:
                continue

            distance_pixels = euclidean_distance(previous_center, detection.center)
            meters_per_second = (distance_pixels / self.config.pixels_per_meter) / seconds_per_frame
            speed_kph = meters_per_second * 3.6

            history = self.speed_history.setdefault(
                detection.track_id,
                deque(maxlen=self.config.smoothing_window),
            )
            history.append(speed_kph)
            speed_map[detection.track_id] = sum(history) / len(history)

        stale_ids = set(self.previous_centers) - active_ids
        for track_id in stale_ids:
            self.previous_centers.pop(track_id, None)
            self.speed_history.pop(track_id, None)

        return speed_map


class AnalyticsEngine:
    def __init__(self, config: AnalyticsConfig) -> None:
        self.config = config
        self.zone_previous_members: dict[str, set[int]] = {zone.name: set() for zone in config.zones}
        self.zone_entry_counts: dict[str, int] = {zone.name: 0 for zone in config.zones}
        self.intrusion_last_frame: dict[tuple[str, int], int] = {}
        self.alert_events: list[AlertEvent] = []
        self.total_intrusions = 0
        self.speed_estimator = SpeedEstimator(config.speed)

    def update(self, detections, frame_index: int, fps: float) -> FrameAnalyticsSnapshot:
        class_counts = Counter(item.class_name for item in detections)
        speed_map = self.speed_estimator.update(detections, fps)
        alert_tracks: set[int] = set()
        active_alerts: list[str] = []
        zone_snapshots: list[ZoneSnapshot] = []

        track_zones: dict[int, list[str]] = {}
        for zone in self.config.zones:
            current_members: set[int] = set()
            zone_class_counts: Counter[str] = Counter()

            for detection in detections:
                if not point_in_polygon(detection.center, zone.polygon):
                    continue

                if zone.count_classes and detection.class_name not in zone.count_classes:
                    continue

                zone_class_counts[detection.class_name] += 1
                if detection.track_id is not None:
                    current_members.add(detection.track_id)
                    track_zones.setdefault(detection.track_id, []).append(zone.name)

            entered_ids = current_members - self.zone_previous_members.get(zone.name, set())
            if entered_ids:
                self.zone_entry_counts[zone.name] += len(entered_ids)

            if zone.restricted and self.config.intrusion.enabled:
                for detection in detections:
                    if detection.track_id not in entered_ids:
                        continue
                    if detection.class_name not in self.config.intrusion.classes:
                        continue

                    cooldown_key = (zone.name, detection.track_id)
                    previous_frame = self.intrusion_last_frame.get(cooldown_key, -self.config.intrusion.cooldown_frames)
                    if frame_index - previous_frame < self.config.intrusion.cooldown_frames:
                        continue

                    message = f"Intrusion: {detection.class_name} entered {zone.name}"
                    active_alerts.append(message)
                    alert_tracks.add(detection.track_id)
                    self.total_intrusions += 1
                    self.intrusion_last_frame[cooldown_key] = frame_index
                    self.alert_events.append(
                        AlertEvent(
                            frame_index=frame_index,
                            zone_name=zone.name,
                            track_id=detection.track_id,
                            class_name=detection.class_name,
                            message=message,
                        )
                    )

            self.zone_previous_members[zone.name] = current_members
            zone_snapshots.append(
                ZoneSnapshot(
                    name=zone.name,
                    current_count=len(current_members),
                    cumulative_entries=self.zone_entry_counts[zone.name],
                    class_counts=dict(zone_class_counts),
                    restricted=zone.restricted,
                )
            )

        current_speeds: list[float] = []
        for detection in detections:
            if detection.track_id is not None:
                detection.zone_names = tuple(track_zones.get(detection.track_id, []))
                detection.is_intrusion = detection.track_id in alert_tracks
                detection.speed_kph = speed_map.get(detection.track_id)
                if detection.speed_kph is not None:
                    current_speeds.append(detection.speed_kph)
            else:
                detection.zone_names = ()
                detection.is_intrusion = False
                detection.speed_kph = None

        return FrameAnalyticsSnapshot(
            zone_stats=zone_snapshots,
            class_counts=dict(class_counts),
            active_alerts=active_alerts,
            total_intrusions=self.total_intrusions,
            average_speed_kph=(sum(current_speeds) / len(current_speeds)) if current_speeds else 0.0,
            max_speed_kph=max(current_speeds) if current_speeds else 0.0,
        )

    def build_report(self) -> dict[str, object]:
        return {
            "total_intrusions": self.total_intrusions,
            "zones": [
                {
                    "name": zone.name,
                    "restricted": zone.restricted,
                    "entries": self.zone_entry_counts.get(zone.name, 0),
                }
                for zone in self.config.zones
            ],
            "alert_events": [
                {
                    "frame_index": alert.frame_index,
                    "zone_name": alert.zone_name,
                    "track_id": alert.track_id,
                    "class_name": alert.class_name,
                    "message": alert.message,
                }
                for alert in self.alert_events
            ],
        }
