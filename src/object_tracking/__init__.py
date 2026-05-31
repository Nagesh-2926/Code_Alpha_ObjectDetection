"""Advanced object detection and tracking package."""

from .config import AppConfig
from .pipeline import RunSummary, run_tracking

__all__ = ["AppConfig", "RunSummary", "run_tracking"]
