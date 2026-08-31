from __future__ import annotations

from typing import Any


class SignalStateEstimator:
    """Estimate red/amber/green from a fixed normalized camera region."""

    def __init__(self, roi: tuple[float, float, float, float], minimum_ratio: float = 0.02) -> None:
        self.roi = roi
        self.minimum_ratio = minimum_ratio

    def estimate(self, frame: Any) -> str:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("OpenCV and NumPy are required for automatic signal detection") from exc

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self.roi
        crop = frame[int(y1 * height) : int(y2 * height), int(x1 * width) : int(x2 * width)]
        if crop.size == 0:
            return "unknown"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        red_a = cv2.inRange(hsv, np.array([0, 90, 100]), np.array([10, 255, 255]))
        red_b = cv2.inRange(hsv, np.array([170, 90, 100]), np.array([180, 255, 255]))
        amber = cv2.inRange(hsv, np.array([12, 90, 100]), np.array([35, 255, 255]))
        green = cv2.inRange(hsv, np.array([36, 70, 70]), np.array([95, 255, 255]))
        ratios = {
            "red": float(((red_a > 0) | (red_b > 0)).mean()),
            "amber": float((amber > 0).mean()),
            "green": float((green > 0).mean()),
        }
        state = max(ratios, key=ratios.get)
        return state if ratios[state] >= self.minimum_ratio else "unknown"

