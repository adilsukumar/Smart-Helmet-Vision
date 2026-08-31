from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import ViolationEvent


class EvidenceStore:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "events.jsonl"

    def write(self, event: ViolationEvent, frame: Any | None = None) -> dict[str, object]:
        payload = event.to_dict()
        if frame is not None:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("OpenCV is required to save evidence images") from exc
            image_name = f"frame_{event.frame_index:06d}_track_{event.track_id}_{event.rule.lower()}.jpg"
            image_path = self.output_dir / image_name
            if not cv2.imwrite(str(image_path), frame):
                raise RuntimeError(f"Could not write evidence image: {image_path}")
            payload["evidence_image"] = image_name

        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        return payload

