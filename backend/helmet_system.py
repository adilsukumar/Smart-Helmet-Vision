from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
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
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, self.y2)

    def expanded(self, horizontal: float, upward: float, downward: float = 0) -> "BoundingBox":
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


def intersection_over_area(box: BoundingBox, region: BoundingBox) -> float:
    width = max(0.0, min(box.x2, region.x2) - max(box.x1, region.x1))
    height = max(0.0, min(box.y2, region.y2) - max(box.y1, region.y1))
    return width * height / box.area if box.area > 0 else 0.0


def point_side(point, line_start, line_end, tolerance: float = 1e-6) -> int:
    px, py = point
    ax, ay = line_start
    bx, by = line_end
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) <= tolerance:
        return 0
    return 1 if cross > 0 else -1


def crossed_to_side(previous, current, line_start, line_end, violation_side: int) -> bool:
    if violation_side not in (-1, 1):
        raise ValueError("violation_side must be -1 or 1")
    old_side = point_side(previous, line_start, line_end)
    new_side = point_side(current, line_start, line_end)
    return old_side not in (0, violation_side) and new_side == violation_side


def is_bike(detection: Detection) -> bool:
    return detection.canonical_label in {"bike", "motorbike", "motorcycle"}


def is_rider(detection: Detection) -> bool:
    label = detection.canonical_label
    return label == "person" or "helmet" in label or label.startswith(("driver", "passenger", "rider"))


def is_no_helmet(detection: Detection) -> bool:
    return detection.canonical_label in {"nohelmet", "withouthelmet", "ridernohelmet"} or "nohelmet" in detection.canonical_label


def attach_helmet_state(people, helmet_marks):
    riders = []
    for person in people:
        best = None
        best_score = 0.0
        upper_body = BoundingBox(
            person.box.x1 - person.box.width * 0.12,
            person.box.y1 - person.box.height * 0.08,
            person.box.x2 + person.box.width * 0.12,
            person.box.y1 + person.box.height * 0.68,
        )
        for mark in helmet_marks:
            if upper_body.contains(mark.box.center) and mark.confidence > best_score:
                best = mark
                best_score = mark.confidence
        label = "rider"
        confidence = person.confidence
        if best is not None:
            label = "rider_no_helmet" if is_no_helmet(best) else "rider_helmet"
            confidence = min(person.confidence, best.confidence)
        riders.append(Detection(label, confidence, person.box, person.track_id))
    return riders


def associate_riders_to_bikes(bikes, riders, minimum_score: float = 0.08):
    grouped = defaultdict(list)
    for rider in riders:
        best_track = None
        best_score = float("-inf")
        rider_x, _ = rider.box.bottom_center

        for bike in bikes:
            if bike.track_id is None:
                continue
            region = bike.box.expanded(horizontal=0.35, upward=1.8, downward=0.25)
            overlap = intersection_over_area(rider.box, region)
            if not region.contains(rider.box.bottom_center) and overlap == 0:
                continue
            distance = abs(rider_x - bike.box.center[0]) / max(region.width, 1)
            score = overlap + max(0.0, 0.5 - distance)
            if score > best_score:
                best_track, best_score = bike.track_id, score

        if best_track is not None and best_score >= minimum_score:
            grouped[best_track].append(rider)
    return dict(grouped)


@dataclass(frozen=True)
class RuleConfig:
    history_window: int = 10
    confirmation_hits: int = 6
    triple_rider_count: int = 3
    line_start: tuple[float, float] = (100.0, 500.0)
    line_end: tuple[float, float] = (1100.0, 500.0)
    violation_side: int = -1

    def __post_init__(self) -> None:
        if not 1 <= self.confirmation_hits <= self.history_window:
            raise ValueError("confirmation_hits must be between 1 and history_window")
        if self.triple_rider_count < 3:
            raise ValueError("triple_rider_count should be at least 3")
        if self.violation_side not in (-1, 1):
            raise ValueError("violation_side must be -1 or 1")


@dataclass
class TrackState:
    no_helmet: deque[bool]
    triple: deque[bool]
    previous_point: tuple[float, float] | None = None
    emitted: set[str] = field(default_factory=set)


class ViolationEngine:
    def __init__(self, config: RuleConfig) -> None:
        self.config = config
        self.tracks: dict[int, TrackState] = {}

    def state_for(self, track_id: int) -> TrackState:
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackState(
                deque(maxlen=self.config.history_window),
                deque(maxlen=self.config.history_window),
            )
        return self.tracks[track_id]

    def observe(self, frame_index, bike, riders, signal_state) -> list[ViolationEvent]:
        if bike.track_id is None:
            return []
        state = self.state_for(bike.track_id)
        state.no_helmet.append(any(is_no_helmet(rider) for rider in riders))
        state.triple.append(len(riders) >= self.config.triple_rider_count)
        events = []

        def emit(rule, confidence):
            if rule in state.emitted:
                return
            state.emitted.add(rule)
            events.append(ViolationEvent(
                frame_index, bike.track_id, rule, round(float(confidence), 4), len(riders), signal_state
            ))

        if sum(state.no_helmet) >= self.config.confirmation_hits:
            scores = [r.confidence for r in riders if is_no_helmet(r)]
            emit("NO_HELMET", min(scores, default=bike.confidence))
        if sum(state.triple) >= self.config.confirmation_hits:
            emit("TRIPLE_RIDING", min([bike.confidence, *[r.confidence for r in riders]]))

        current_point = bike.box.bottom_center
        if signal_state.lower() == "red" and state.previous_point is not None:
            if crossed_to_side(state.previous_point, current_point, self.config.line_start,
                               self.config.line_end, self.config.violation_side):
                emit("RED_LIGHT_CROSSING", bike.confidence)
        state.previous_point = current_point
        return events


class SignalStateEstimator:
    def __init__(self, roi, minimum_ratio: float = 0.02) -> None:
        self.roi = roi
        self.minimum_ratio = minimum_ratio

    def estimate(self, frame: Any) -> str:
        import cv2
        import numpy as np

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = self.roi
        crop = frame[int(y1 * height):int(y2 * height), int(x1 * width):int(x2 * width)]
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


class UltralyticsTracker:
    def __init__(self, model_path: str, confidence: float = 0.25, device: str | None = None) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install backend/requirements.txt to run video inference") from exc
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device

    def track(self, frame: Any) -> list[Detection]:
        result = self.model.track(frame, persist=True, tracker="bytetrack.yaml",
                                  conf=self.confidence, device=self.device, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None:
            return []
        xyxy = boxes.xyxy.cpu().tolist()
        classes = boxes.cls.cpu().tolist()
        scores = boxes.conf.cpu().tolist()
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(xyxy)
        return [
            Detection(str(result.names[int(class_id)]), float(score),
                      BoundingBox(*map(float, coordinates)), int(track_id) if track_id is not None else None)
            for coordinates, class_id, score, track_id in zip(xyxy, classes, scores, ids)
        ]


class UltralyticsDetector:
    def __init__(self, model_path: str, confidence: float = 0.2, device: str | None = None) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install backend/ml-requirements.txt to run video inference") from exc
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device

    def detect(self, frame: Any) -> list[Detection]:
        result = self.model.predict(frame, conf=self.confidence, device=self.device,
                                    imgsz=640, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None:
            return []
        return [
            Detection(str(result.names[int(class_id)]), float(score),
                      BoundingBox(*map(float, coordinates)))
            for coordinates, class_id, score in zip(
                boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist()
            )
        ]


class EvidenceStore:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "events.jsonl"

    def write(self, event: ViolationEvent, frame=None):
        payload = event.to_dict()
        if frame is not None:
            import cv2
            name = f"frame_{event.frame_index:06d}_track_{event.track_id}_{event.rule.lower()}.jpg"
            if not cv2.imwrite(str(self.output_dir / name), frame):
                raise RuntimeError("Could not save evidence image")
            payload["evidence_image"] = name
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        return payload


def run_demo() -> int:
    engine = ViolationEngine(RuleConfig(
        history_window=5, confirmation_hits=3, line_start=(0, 100),
        line_end=(300, 100), violation_side=-1,
    ))
    all_events = []
    print("Running a controlled six-frame traffic scenario...\n")
    for frame_index, bottom_y in enumerate([145, 130, 112, 94, 82, 70]):
        bike = Detection("motorbike", 0.95, BoundingBox(80, bottom_y - 35, 220, bottom_y), 7)
        riders = [
            Detection("DHelmet", 0.94, BoundingBox(100, bottom_y - 105, 140, bottom_y - 15)),
            Detection("P1Helmet", 0.91, BoundingBox(135, bottom_y - 100, 175, bottom_y - 12)),
            Detection("P2NoHelmet", 0.89, BoundingBox(165, bottom_y - 95, 205, bottom_y - 10)),
        ]
        events = engine.observe(frame_index, bike, riders, "red")
        all_events.extend(events)
        print(f"frame={frame_index} events={[event.rule for event in events]}")

    observed = {event.rule for event in all_events}
    expected = {"NO_HELMET", "TRIPLE_RIDING", "RED_LIGHT_CROSSING"}
    if observed != expected:
        raise SystemExit(f"Expected {expected}, observed {observed}")
    print("\nSimulation passed:", ", ".join(sorted(observed)))
    return 0


def draw(frame, detections, events, signal, line=None) -> None:
    import cv2
    if line is not None:
        cv2.line(frame, tuple(map(int, line[0])), tuple(map(int, line[1])), (255, 180, 0), 2)
    for detection in detections:
        box = detection.box
        if is_no_helmet(detection):
            color = (50, 50, 235)
        elif "helmet" in detection.canonical_label:
            color = (70, 205, 110)
        else:
            color = (235, 190, 65)
        cv2.rectangle(frame, (int(box.x1), int(box.y1)), (int(box.x2), int(box.y2)), color, 2)
        track = f" #{detection.track_id}" if detection.track_id is not None else ""
        cv2.putText(frame, f"{detection.label}{track} {detection.confidence:.2f}",
                    (int(box.x1), max(18, int(box.y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    overlay = frame.copy()
    cv2.rectangle(overlay, (12, 12), (650, 92), (12, 22, 28), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    cv2.putText(frame, "REAL TRAFFIC FOOTAGE / TWO-MODEL PROTOTYPE", (26, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.64, (245, 245, 245), 2)
    signal_text = f"Signal state: {signal}" if signal != "unknown" else "Signal jumping: not evaluated in this clip"
    cv2.putText(frame, signal_text, (26, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 205, 215), 1)
    for index, event in enumerate(events):
        cv2.putText(frame, f"CANDIDATE EVENT: {event.rule}", (20, 118 + index * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def run_video(args) -> int:
    import cv2

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    source = int(args.source) if args.source.isdigit() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {args.source}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = min(args.width, source_width)
    height = round(width * source_height / source_width)
    if height % 2:
        height += 1
    scale = lambda p: (p[0] * width, p[1] * height)
    line_start, line_end = scale(config["stop_line"][0]), scale(config["stop_line"][1])
    rules = RuleConfig(config["history_window"], config["confirmation_hits"],
                       config["triple_rider_count"], line_start, line_end, config["violation_side"])
    detector = UltralyticsTracker(args.model, args.confidence, args.device)
    helmet_detector = UltralyticsDetector(args.helmet_model, args.helmet_confidence, args.device) if args.helmet_model else None
    engine = ViolationEngine(rules)
    signal_reader = SignalStateEstimator(config["signal_roi"])
    evidence = EvidenceStore(args.output)
    frame_index = 0
    read_index = 0
    stride = max(1, round(source_fps / args.fps))
    writer = None
    if args.save_video:
        output_path = Path(args.save_video)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 source_fps / stride, (width, height))
        if not writer.isOpened():
            raise SystemExit(f"Could not create video: {output_path}")

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if read_index % stride:
            read_index += 1
            continue
        read_index += 1
        if frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        detections = detector.track(frame)
        bikes = [item for item in detections if is_bike(item)]
        people = [item for item in detections if item.canonical_label == "person"]
        helmet_marks = helmet_detector.detect(frame) if helmet_detector else []
        riders = attach_helmet_state(people, helmet_marks) if helmet_detector else people
        groups = associate_riders_to_bikes(bikes, riders)
        signal = args.signal if args.signal != "auto" else signal_reader.estimate(frame)
        events = []
        for bike in bikes:
            events.extend(engine.observe(frame_index, bike, groups.get(bike.track_id, []), signal))
        grouped_riders = {
            rider.track_id: rider
            for bike_riders in groups.values()
            for rider in bike_riders
            if rider.track_id is not None
        }
        visible = [*bikes, *grouped_riders.values()]
        line = (line_start, line_end) if signal != "unknown" else None
        draw(frame, visible, events, signal, line)
        for event in events:
            print(json.dumps(evidence.write(event, frame)))
        if args.display:
            cv2.imshow("Smart Helmet Vision - press q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if writer is not None:
            writer.write(frame)
        frame_index += 1
    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Smart Helmet Vision")
    parser.add_argument("--demo", action="store_true", help="Run without a model or camera")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model")
    parser.add_argument("--config", default=str(Path(__file__).with_name("traffic_demo.json")))
    parser.add_argument("--helmet-model")
    parser.add_argument("--signal", choices=["auto", "red", "amber", "green", "unknown"], default="auto")
    parser.add_argument("--output", default="runs/traffic_events")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--helmet-confidence", type=float, default=0.15)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-video")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()
    if args.demo:
        return run_demo()
    if not args.model:
        parser.error("--model is required unless --demo is used")
    return run_video(args)


if __name__ == "__main__":
    raise SystemExit(main())
