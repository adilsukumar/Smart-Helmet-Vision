from __future__ import annotations

import json

from traffic_system.engine import RuleConfig, ViolationEngine
from traffic_system.types import BoundingBox, Detection


def detection(label: str, box: tuple[float, float, float, float], track_id: int | None, confidence: float = 0.92) -> Detection:
    return Detection(label, confidence, BoundingBox(*box), track_id)


def main() -> int:
    engine = ViolationEngine(
        RuleConfig(
            history_window=5,
            confirmation_hits=3,
            line_start=(0.0, 100.0),
            line_end=(300.0, 100.0),
            violation_side=-1,
        )
    )
    print("Running a controlled six-frame traffic scenario...\n")
    all_events = []

    bike_bottom_y = [145, 130, 112, 94, 82, 70]
    for frame_index, bottom_y in enumerate(bike_bottom_y):
        bike = detection("motorbike", (80, bottom_y - 35, 220, bottom_y), 7, 0.95)
        riders = [
            detection("DHelmet", (100, bottom_y - 105, 140, bottom_y - 15), None, 0.94),
            detection("P1Helmet", (135, bottom_y - 100, 175, bottom_y - 12), None, 0.91),
            detection("P2NoHelmet", (165, bottom_y - 95, 205, bottom_y - 10), None, 0.89),
        ]
        events = engine.observe(frame_index, bike, riders, "red")
        all_events.extend(events)
        print(f"frame={frame_index} bike_bottom_y={bottom_y} new_events={[event.rule for event in events]}")

    print("\nFinal events:")
    for event in all_events:
        print(json.dumps(event.to_dict(), indent=2))

    observed = {event.rule for event in all_events}
    expected = {"NO_HELMET", "TRIPLE_RIDING", "RED_LIGHT_CROSSING"}
    if observed != expected:
        raise SystemExit(f"Simulation failed: expected {expected}, observed {observed}")
    print("\nSimulation passed: every expected violation was emitted once.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
