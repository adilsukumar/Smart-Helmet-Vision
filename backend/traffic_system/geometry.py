from __future__ import annotations

from .types import BoundingBox


def intersection_area(a: BoundingBox, b: BoundingBox) -> float:
    width = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    height = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    return width * height


def intersection_over_area(a: BoundingBox, container: BoundingBox) -> float:
    if a.area <= 0:
        return 0.0
    return intersection_area(a, container) / a.area


def point_side(
    point: tuple[float, float],
    line_start: tuple[float, float],
    line_end: tuple[float, float],
    tolerance: float = 1e-6,
) -> int:
    """Return -1, 0 or 1 for a point relative to a directed line."""
    px, py = point
    ax, ay = line_start
    bx, by = line_end
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) <= tolerance:
        return 0
    return 1 if cross > 0 else -1


def crossed_to_side(
    previous: tuple[float, float],
    current: tuple[float, float],
    line_start: tuple[float, float],
    line_end: tuple[float, float],
    violation_side: int,
) -> bool:
    if violation_side not in (-1, 1):
        raise ValueError("violation_side must be -1 or 1")
    old_side = point_side(previous, line_start, line_end)
    new_side = point_side(current, line_start, line_end)
    return old_side not in (0, violation_side) and new_side == violation_side

