"""
Track-level memory for expensive inference results.

Stores helmet / seatbelt / plate results per track ID so that the pipeline
does not re-run expensive models on the same vehicle every frame.

Recheck logic:
  - Every refresh_interval frames (periodic scheduled recheck).
  - Immediately when last result had low confidence for ANY status:
      "ok" with low conf  → model wasn't sure the helmet was present
                            (vehicle may have been occluded / in shadow)
      "no_helmet" with low conf → recheck to confirm before emitting again
    This handles the occlusion-then-clear scenario: a bike in shadow at
    frame 0 might get an uncertain "ok". By frame 5 it's in clear light
    and should be rechecked rather than waiting the full 30-frame interval.

Entries are evicted when the track disappears (synced with tracker eviction).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrackState:
    vehicle_type: str
    # Helmet
    helmet_checked: bool = False
    helmet_status: Optional[str] = None       # "no_helmet" | "ok" | None
    helmet_confidence: float = 0.0
    last_helmet_frame: int = -1
    helmet_violation_emitted: bool = False
    # Seatbelt
    seatbelt_checked: bool = False
    seatbelt_status: Optional[str] = None     # "no_seatbelt" | "seatbelt" | "indeterminate"
    seatbelt_confidence: float = 0.0
    last_seatbelt_frame: int = -1
    seatbelt_violation_emitted: bool = False
    # Plate / OCR
    plate_number: Optional[str] = None
    plate_confidence: float = 0.0


class TrackMemory:
    def __init__(
        self,
        helmet_refresh_interval: int = 30,
        seatbelt_refresh_interval: int = 60,
        low_conf_recheck_threshold: float = 0.70,
    ):
        self._states: dict[int, TrackState] = {}
        self._helmet_interval = helmet_refresh_interval
        self._seatbelt_interval = seatbelt_refresh_interval
        # Any result (ok OR violation) below this confidence triggers an early
        # recheck on the next frame, regardless of the interval. This catches
        # the occlusion / poor visibility scenario.
        self._low_conf = low_conf_recheck_threshold

    def get_or_create(self, track_id: int, vehicle_type: str) -> TrackState:
        if track_id not in self._states:
            self._states[track_id] = TrackState(vehicle_type=vehicle_type)
        return self._states[track_id]

    def get(self, track_id: int) -> Optional[TrackState]:
        return self._states.get(track_id)

    def needs_helmet_check(self, track_id: int, frame_id: int) -> bool:
        state = self._states.get(track_id)
        if state is None:
            return True
        if not state.helmet_checked:
            return True
        # Low confidence on ANY result means the model wasn't sure:
        #   "ok" + low conf  → bike may have been occluded; recheck next frame
        #   "no_helmet" + low conf → recheck to confirm before treating as definitive
        if state.helmet_confidence < self._low_conf:
            return True
        return (frame_id - state.last_helmet_frame) >= self._helmet_interval

    def needs_seatbelt_check(self, track_id: int, frame_id: int) -> bool:
        state = self._states.get(track_id)
        if state is None:
            return True
        if not state.seatbelt_checked:
            return True
        # Same low-confidence early-recheck for seatbelt.
        if state.seatbelt_confidence < self._low_conf and state.seatbelt_status in ("seatbelt", "no_seatbelt"):
            return True
        return (frame_id - state.last_seatbelt_frame) >= self._seatbelt_interval

    def has_plate(self, track_id: int) -> bool:
        state = self._states.get(track_id)
        return state is not None and state.plate_number is not None

    def evict_stale(self, active_track_ids: set[int]):
        stale = [tid for tid in self._states if tid not in active_track_ids]
        for tid in stale:
            del self._states[tid]
