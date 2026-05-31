from object_tracking.analytics import AnalyticsEngine
from object_tracking.config import AnalyticsConfig, IntrusionConfig, SpeedConfig, ZoneConfig
from object_tracking.pipeline import Detection


def _build_engine() -> AnalyticsEngine:
    return AnalyticsEngine(
        AnalyticsConfig(
            enable_line_counting=False,
            count_line=None,
            enable_zone_counting=True,
            zones=[
                ZoneConfig(
                    name="Restricted Area",
                    polygon=((0, 0), (100, 0), (100, 100), (0, 100)),
                    color=(0, 255, 0),
                    count_classes=["person"],
                    restricted=True,
                )
            ],
            intrusion=IntrusionConfig(enabled=True, classes=["person"], cooldown_frames=5),
            speed=SpeedConfig(enabled=True, pixels_per_meter=10.0, smoothing_window=3, classes=["person"]),
        )
    )


def test_zone_entries_intrusions_and_speed_updates():
    engine = _build_engine()
    first_frame = [
        Detection(
            bbox=(10, 10, 30, 40),
            center=(20, 25),
            class_id=0,
            class_name="person",
            confidence=0.9,
            track_id=1,
        )
    ]
    second_frame = [
        Detection(
            bbox=(20, 10, 40, 40),
            center=(30, 25),
            class_id=0,
            class_name="person",
            confidence=0.88,
            track_id=1,
        )
    ]

    first_snapshot = engine.update(first_frame, frame_index=1, fps=10.0)
    second_snapshot = engine.update(second_frame, frame_index=2, fps=10.0)

    assert first_snapshot.total_intrusions == 1
    assert first_snapshot.zone_stats[0].current_count == 1
    assert first_snapshot.zone_stats[0].cumulative_entries == 1
    assert second_snapshot.average_speed_kph > 0
    assert second_frame[0].speed_kph is not None
