from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("Bounding box must satisfy x2 >= x1 and y2 >= y1")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def expanded(self, horizontal: float, upward: float, downward: float = 0.0) -> "BoundingBox":
        return BoundingBox(
            self.x1 - self.width * horizontal,
            self.y1 - self.height * upward,
            self.x2 + self.width * horizontal,
            self.y2 + self.height * downward,
        )

    def contains(self, point: tuple[float, float]) -> bool:
        x, y = point
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: BoundingBox
    track_id: int | None = None

    @property
    def canonical_label(self) -> str:
        return self.label.lower().replace("_", "").replace("-", "").replace(" ", "")


@dataclass(frozen=True)
class ViolationEvent:
    frame_index: int
    track_id: int
    rule: str
    confidence: float
    rider_count: int
    signal_state: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

