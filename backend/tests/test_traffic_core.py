from __future__ import annotations

import unittest

from traffic_system.association import associate_riders_to_bikes
from traffic_system.engine import RuleConfig, ViolationEngine
from traffic_system.geometry import crossed_to_side, point_side
from traffic_system.types import BoundingBox, Detection


def det(label: str, box: tuple[float, float, float, float], track_id: int | None = None) -> Detection:
    return Detection(label, 0.9, BoundingBox(*box), track_id)


class GeometryTests(unittest.TestCase):
    def test_point_side(self) -> None:
        self.assertEqual(point_side((5, 5), (0, 0), (10, 0)), 1)
        self.assertEqual(point_side((5, -5), (0, 0), (10, 0)), -1)

    def test_crossing_in_required_direction(self) -> None:
        self.assertTrue(crossed_to_side((5, 5), (5, -2), (0, 0), (10, 0), -1))
        self.assertFalse(crossed_to_side((5, -2), (5, 5), (0, 0), (10, 0), -1))


class AssociationTests(unittest.TestCase):
    def test_rider_is_assigned_to_only_one_bike(self) -> None:
        bikes = [det("motorbike", (20, 100, 120, 160), 1), det("motorbike", (180, 100, 280, 160), 2)]
        riders = [det("DHelmet", (45, 40, 95, 145)), det("DNoHelmet", (205, 40, 255, 145))]
        groups = associate_riders_to_bikes(bikes, riders)
        self.assertEqual(groups[1][0].label, "DHelmet")
        self.assertEqual(groups[2][0].label, "DNoHelmet")


class ViolationEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ViolationEngine(
            RuleConfig(
                history_window=3,
                confirmation_hits=2,
                line_start=(0, 100),
                line_end=(300, 100),
                violation_side=-1,
            )
        )
        self.riders = [
            det("DHelmet", (100, 10, 130, 100)),
            det("P1Helmet", (130, 10, 160, 100)),
            det("P2NoHelmet", (160, 10, 190, 100)),
        ]

    def test_persistence_and_duplicate_suppression(self) -> None:
        bike = det("motorbike", (80, 100, 210, 150), 8)
        self.assertEqual(self.engine.observe(0, bike, self.riders, "green"), [])
        events = self.engine.observe(1, bike, self.riders, "green")
        self.assertEqual({event.rule for event in events}, {"NO_HELMET", "TRIPLE_RIDING"})
        self.assertEqual(self.engine.observe(2, bike, self.riders, "green"), [])

    def test_red_crossing_only_when_red(self) -> None:
        before = det("motorbike", (80, 90, 210, 130), 9)
        after = det("motorbike", (80, 40, 210, 90), 9)
        self.engine.observe(0, before, [], "red")
        events = self.engine.observe(1, after, [], "red")
        self.assertEqual([event.rule for event in events], ["RED_LIGHT_CROSSING"])

    def test_crossing_on_green_does_not_trigger(self) -> None:
        before = det("motorbike", (80, 90, 210, 130), 10)
        after = det("motorbike", (80, 40, 210, 90), 10)
        self.engine.observe(0, before, [], "green")
        self.assertEqual(self.engine.observe(1, after, [], "green"), [])


if __name__ == "__main__":
    unittest.main()

