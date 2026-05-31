from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from object_tracking.config import load_config
from object_tracking.pipeline import run_tracking


st.set_page_config(page_title="CodeAlpha Object Detection Pro", layout="wide")
st.title("CodeAlpha Object Detection Pro")
st.caption("Upload a video, enable analytics, and generate a processed output with tracking, zones, pose, speed, and privacy controls.")


with st.sidebar:
    st.header("Runtime Controls")
    tracker = st.selectbox("Tracker", ["botsort.yaml", "bytetrack.yaml"], index=0)
    enable_pose = st.checkbox("Enable pose landmarks", value=False)
    enable_pose_labels = st.checkbox("Draw pose labels", value=False)
    enable_speed = st.checkbox("Enable speed estimation", value=False)
    pixels_per_meter = st.number_input("Pixels per meter", min_value=1.0, value=12.0, step=1.0)
    enable_zones = st.checkbox("Enable zone counting", value=False)
    enable_intrusion = st.checkbox("Enable intrusion detection", value=False)
    privacy_mode = st.selectbox("Privacy mode", ["disabled", "face", "person"], index=0)


uploaded_file = st.file_uploader("Upload a reference video", type=["mp4", "avi", "mov", "mkv"])
source_path = st.text_input("Or use an existing file path", value="")


def _materialize_input_file() -> str | None:
    if uploaded_file is None and not source_path.strip():
        return None
    if source_path.strip():
        return source_path.strip()

    upload_dir = PROJECT_ROOT / "artifacts" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    input_path = upload_dir / uploaded_file.name
    with input_path.open("wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return str(input_path)


if st.button("Process Video", type="primary"):
    selected_source = _materialize_input_file()
    if not selected_source:
        st.error("Provide either an uploaded file or a local source path.")
    else:
        overrides: dict[str, object] = {
            "source": selected_source,
            "output": {"show_window": False},
            "model": {"tracker": tracker},
            "features": {
                "analytics": {
                    "enable_zone_counting": enable_zones,
                    "intrusion": {"enabled": enable_intrusion},
                    "speed": {"enabled": enable_speed, "pixels_per_meter": pixels_per_meter},
                },
                "pose": {"enabled": enable_pose, "draw_labels": enable_pose_labels},
            },
        }
        if privacy_mode != "disabled":
            overrides["features"]["privacy"] = {"enabled": True, "mode": privacy_mode}

        with st.spinner("Running detection and tracking pipeline..."):
            config = load_config(str(PROJECT_ROOT / "configs" / "default.yaml"), overrides=overrides)
            summary = run_tracking(config)

        st.success("Processing complete.")
        if summary.output_path:
            st.video(summary.output_path)

        metrics_col_1, metrics_col_2, metrics_col_3 = st.columns(3)
        metrics_col_1.metric("Frames", summary.frames_processed)
        metrics_col_2.metric("Average FPS", f"{summary.average_fps:.2f}")
        metrics_col_3.metric("Intrusions", summary.total_intrusions)

        if summary.analytics_path and Path(summary.analytics_path).exists():
            analytics_payload = json.loads(Path(summary.analytics_path).read_text(encoding="utf-8"))
            st.subheader("Analytics Summary")
            st.json(analytics_payload)
