from __future__ import annotations

from typing import Any

from .types import BoundingBox, Detection


class UltralyticsTracker:
    """Small adapter around Ultralytics persistent tracking.

    Imports are intentionally lazy so the rule-engine tests do not require the
    large ML dependency set.
    """

    def __init__(self, model_path: str, confidence: float = 0.25, device: str | None = None) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install requirements-ml.txt to run video inference") from exc

        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device

    def track(self, frame: Any) -> list[Detection]:
        result = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None:
            return []

        xyxy = boxes.xyxy.cpu().tolist()
        classes = boxes.cls.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
        names = result.names

        return [
            Detection(
                label=str(names[int(class_id)]),
                confidence=float(score),
                box=BoundingBox(*map(float, coordinates)),
                track_id=int(track_id) if track_id is not None else None,
            )
            for coordinates, class_id, score, track_id in zip(xyxy, classes, confidences, ids)
        ]

