import asyncio
from pathlib import Path

import httpx

from api import app
from object_tracking.pipeline import RunSummary


def _request(method: str, path: str, **kwargs):
    async def _send():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(_send())


def test_health_endpoint_returns_ok():
    response = _request("GET", "/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_process_video_rejects_invalid_tracker():
    video_path = Path("artifacts/demo_input.mp4")
    with video_path.open("rb") as handle:
        response = _request(
            "POST",
            "/process-video",
            data={"tracker": "invalid-tracker"},
            files={"file": ("demo_input.mp4", handle, "video/mp4")},
        )

    assert response.status_code == 400
    assert "tracker" in response.json()["detail"]


def test_process_video_returns_summary(monkeypatch):
    video_path = Path("artifacts/demo_input.mp4")

    def fake_run_tracking(config):
        return RunSummary(
            source=str(config.source),
            frames_processed=12,
            average_fps=2.5,
            output_path="artifacts/runs/fake.mp4",
            analytics_path="artifacts/runs/fake.json",
            tracker=config.model.tracker,
            model_weights=config.model.weights,
            total_intrusions=1,
        )

    monkeypatch.setattr("api.run_tracking", fake_run_tracking)

    with video_path.open("rb") as handle:
        response = _request(
            "POST",
            "/process-video",
            data={"tracker": "botsort.yaml", "pose": "false", "speed": "false", "intrusion": "false"},
            files={"file": ("demo_input.mp4", handle, "video/mp4")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["frames_processed"] == 12
    assert payload["total_intrusions"] == 1
