"""Traffic-rule monitoring research prototype."""

from .engine import RuleConfig, ViolationEngine
from .types import BoundingBox, Detection, ViolationEvent

__all__ = [
    "BoundingBox",
    "Detection",
    "RuleConfig",
    "ViolationEngine",
    "ViolationEvent",
]

