from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .association import associate_riders_to_bikes, is_bike, is_rider
from .detector import UltralyticsTracker
from .engine import RuleConfig, ViolationEngine
from .evidence import EvidenceStore
from .signal import SignalStateEstimator


def _pixel_point(point: list[float], width: int, height: int) -> tuple[float, float]:
    return (point[0] * width, point[1] * height)


def _draw(frame: Any, detections: list[Any], events: list[Any], signal: str, line: tuple[Any, Any]) -> None:
    import cv2

    colors = {"red": (0, 0, 255), "green": (0, 180, 0), "amber": (0, 180, 255)}
    cv2.line(frame, tuple(map(int, line[0])), tuple(map(int, line[1])), (255, 180, 0), 2)
    for detection in detections:
        box = detection.box
        color = (0, 180, 0) if "nohelmet" not in detection.canonical_label else (0, 0, 255)
        cv2.rectangle(frame, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, 2)
        text = f"{detection.label} {detection.confidence:.2f} id={detection.track_id}"
        cv2.putText(frame, text, (int(box.x1), max(18, int(box.y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(frame, f"Signal: {signal}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors.get(signal, (180, 180, 180)), 2)
    y = 60
    for event in events:
        cv2.putText(frame, f"VIOLATION: {event.rule} / track {event.track_id}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y += 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traffic-rule monitoring research prototype")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--model", required=True, help="Custom Ultralytics model path")
    parser.add_argument("--config", default="configs/traffic_demo.json")
    parser.add_argument("--signal", choices=["auto", "red", "amber", "green"], default="auto")
    parser.add_argument("--output", default="runs/traffic_events")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", default=None)
    parser.add_argument("--display", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Install requirements-ml.txt before running the video application") from exc

    args = parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    source: int | str = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    line_start = _pixel_point(config["stop_line"][0], width, height)
    line_end = _pixel_point(config["stop_line"][1], width, height)
    rules = RuleConfig(
        history_window=int(config["history_window"]),
        confirmation_hits=int(config["confirmation_hits"]),
        triple_rider_count=int(config["triple_rider_count"]),
        line_start=line_start,
        line_end=line_end,
        violation_side=int(config["violation_side"]),
    )
    detector = UltralyticsTracker(args.model, confidence=args.confidence, device=args.device)
    engine = ViolationEngine(rules)
    signal_estimator = SignalStateEstimator(tuple(config["signal_roi"]))
    evidence = EvidenceStore(args.output)
    frame_index = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        detections = detector.track(frame)
        bikes = [d for d in detections if is_bike(d)]
        riders = [d for d in detections if is_rider(d) and not is_bike(d)]
        grouped = associate_riders_to_bikes(bikes, riders)
        signal_state = args.signal if args.signal != "auto" else signal_estimator.estimate(frame)
        frame_events = []
        for bike in bikes:
            if bike.track_id is None:
                continue
            events = engine.observe(frame_index, bike, grouped.get(bike.track_id, []), signal_state)
            frame_events.extend(events)

        _draw(frame, detections, frame_events, signal_state, (line_start, line_end))
        for event in frame_events:
            print(json.dumps(evidence.write(event, frame)))

        if args.display:
            cv2.imshow("Traffic rule prototype - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_index += 1

    capture.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

