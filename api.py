from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from object_tracking.config import load_config
from object_tracking.pipeline import run_tracking


app = FastAPI(title="CodeAlpha Object Detection API", version="2.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/process-video")
async def process_video(
    file: UploadFile = File(...),
    tracker: str = Form("botsort.yaml"),
    pose: bool = Form(False),
    privacy_mode: str = Form(""),
    speed: bool = Form(False),
    intrusion: bool = Form(False),
) -> dict[str, object]:
    if tracker not in {"botsort.yaml", "bytetrack.yaml"}:
        raise HTTPException(status_code=400, detail="tracker must be either 'botsort.yaml' or 'bytetrack.yaml'.")

    if privacy_mode and privacy_mode not in {"face", "person"}:
        raise HTTPException(status_code=400, detail="privacy_mode must be empty, 'face', or 'person'.")

    with tempfile.TemporaryDirectory() as temporary_directory:
        temp_dir = Path(temporary_directory)
        input_path = temp_dir / Path(file.filename or "upload.mp4").name
        with input_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)

        overrides: dict[str, object] = {
            "source": str(input_path),
            "output": {"show_window": False},
            "model": {"tracker": tracker},
        }

        if pose:
            overrides.setdefault("features", {})
            overrides["features"]["pose"] = {"enabled": True}
        if speed:
            overrides.setdefault("features", {})
            overrides["features"].setdefault("analytics", {})
            overrides["features"]["analytics"]["speed"] = {"enabled": True}
        if intrusion:
            overrides.setdefault("features", {})
            overrides["features"].setdefault("analytics", {})
            overrides["features"]["analytics"]["intrusion"] = {"enabled": True}
        if privacy_mode:
            overrides.setdefault("features", {})
            overrides["features"]["privacy"] = {"enabled": True, "mode": privacy_mode}

        try:
            config = load_config(str(PROJECT_ROOT / "configs" / "default.yaml"), overrides=overrides)
            summary = run_tracking(config)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Video processing failed: {exc}") from exc
        return {
            "source": summary.source,
            "frames_processed": summary.frames_processed,
            "average_fps": summary.average_fps,
            "output_path": summary.output_path,
            "analytics_path": summary.analytics_path,
            "total_intrusions": summary.total_intrusions,
        }
