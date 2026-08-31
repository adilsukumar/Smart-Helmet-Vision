from __future__ import annotations

from collections import defaultdict

from .geometry import intersection_over_area
from .types import Detection


BIKE_LABELS = {"bike", "motorbike", "motorcycle"}


def is_bike(detection: Detection) -> bool:
    return detection.canonical_label in BIKE_LABELS


def is_rider(detection: Detection) -> bool:
    label = detection.canonical_label
    return label == "person" or "helmet" in label or label.startswith(("driver", "passenger", "rider"))


def is_known_no_helmet(detection: Detection) -> bool:
    return "nohelmet" in detection.canonical_label


def associate_riders_to_bikes(
    bikes: list[Detection],
    riders: list[Detection],
    minimum_score: float = 0.08,
) -> dict[int, list[Detection]]:
    """Assign every rider to at most one tracked bike.

    The bike box is expanded upward because rider boxes normally sit above the
    motorcycle itself. The score combines overlap with horizontal proximity.
    """
    grouped: dict[int, list[Detection]] = defaultdict(list)
    for rider in riders:
        best_track: int | None = None
        best_score = float("-inf")
        rider_x, _ = rider.box.bottom_center

        for bike in bikes:
            if bike.track_id is None:
                continue
            region = bike.box.expanded(horizontal=0.35, upward=1.8, downward=0.25)
            if not region.contains(rider.box.bottom_center) and intersection_over_area(rider.box, region) == 0:
                continue

            overlap = intersection_over_area(rider.box, region)
            horizontal_distance = abs(rider_x - bike.box.center[0]) / max(region.width, 1.0)
            score = overlap + max(0.0, 0.5 - horizontal_distance)
            if score > best_score:
                best_score = score
                best_track = bike.track_id

        if best_track is not None and best_score >= minimum_score:
            grouped[best_track].append(rider)

    return dict(grouped)

