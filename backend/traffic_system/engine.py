from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .association import is_known_no_helmet
from .geometry import crossed_to_side
from .types import Detection, ViolationEvent


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
class _TrackState:
    no_helmet: deque[bool]
    triple: deque[bool]
    previous_point: tuple[float, float] | None = None
    emitted: set[str] = field(default_factory=set)


class ViolationEngine:
    def __init__(self, config: RuleConfig) -> None:
        self.config = config
        self._tracks: dict[int, _TrackState] = {}

    def _state_for(self, track_id: int) -> _TrackState:
        if track_id not in self._tracks:
            self._tracks[track_id] = _TrackState(
                no_helmet=deque(maxlen=self.config.history_window),
                triple=deque(maxlen=self.config.history_window),
            )
        return self._tracks[track_id]

    def observe(
        self,
        frame_index: int,
        bike: Detection,
        riders: list[Detection],
        signal_state: str,
    ) -> list[ViolationEvent]:
        if bike.track_id is None:
            return []

        state = self._state_for(bike.track_id)
        state.no_helmet.append(any(is_known_no_helmet(rider) for rider in riders))
        state.triple.append(len(riders) >= self.config.triple_rider_count)
        events: list[ViolationEvent] = []

        def emit(rule: str, confidence: float) -> None:
            if rule in state.emitted:
                return
            state.emitted.add(rule)
            events.append(
                ViolationEvent(
                    frame_index=frame_index,
                    track_id=bike.track_id or 0,
                    rule=rule,
                    confidence=round(float(confidence), 4),
                    rider_count=len(riders),
                    signal_state=signal_state,
                )
            )

        if sum(state.no_helmet) >= self.config.confirmation_hits:
            no_helmet_scores = [r.confidence for r in riders if is_known_no_helmet(r)]
            emit("NO_HELMET", min(no_helmet_scores, default=bike.confidence))

        if sum(state.triple) >= self.config.confirmation_hits:
            emit("TRIPLE_RIDING", min([bike.confidence, *[r.confidence for r in riders]]))

        current_point = bike.box.bottom_center
        if (
            signal_state.lower() == "red"
            and state.previous_point is not None
            and crossed_to_side(
                state.previous_point,
                current_point,
                self.config.line_start,
                self.config.line_end,
                self.config.violation_side,
            )
        ):
            emit("RED_LIGHT_CROSSING", bike.confidence)

        state.previous_point = current_point
        return events

    def reset(self) -> None:
        self._tracks.clear()

