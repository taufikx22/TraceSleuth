"""TraceSleuth Replay Package.

Exports replay service, configurations, metrics diff, and verdict models.
"""

from src.replay.types import (
    ReplayConfig,
    ReplayMetricsDiff,
    ReplayRun,
    ReplayType,
    ReplayVerdict,
)

__all__ = [
    "ReplayService",
    "ReplayConfig",
    "ReplayMetricsDiff",
    "ReplayRun",
    "ReplayType",
    "ReplayVerdict",
]


def __getattr__(name: str):
    if name == "ReplayService":
        from src.replay.service import ReplayService

        return ReplayService
    raise AttributeError(f"module {__name__} has no attribute {name}")
