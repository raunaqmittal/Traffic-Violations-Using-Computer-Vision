"""
Lightweight multi-object tracker (greedy IoU association).

We deliberately do NOT call Ultralytics' internal BYTETracker: it is a private
API whose constructor and update signatures change between releases (and broke
on ultralytics 8.4.x). For image-based / short-clip traffic enforcement this
self-contained IoU tracker is robust, deterministic, dependency-free, and gives
exactly what the violation modules need: stable track IDs plus per-track
centroid history.

Association: greedy IoU matching of current detections to existing tracks; an
unmatched detection starts a new track; a track unseen for `track_buffer`
frames is dropped.
"""

from collections import defaultdict, deque

import numpy as np

from src.models import Detection, TrackedObject

_HISTORY_LEN = 60   # frames of centroid history to keep per track


class Tracker:
    def __init__(
        self,
        track_thresh: float = 0.50,
        track_buffer: int = 30,
        match_thresh: float = 0.80,
        iou_gate: float = 0.3,
    ):
        # track_thresh: minimum detection confidence to track.
        # track_buffer: frames to keep a lost track alive before dropping it.
        # match_thresh: retained for config compatibility (ByteTrack-era param).
        # iou_gate: minimum IoU for a detection to match an existing track.
        self._track_thresh = track_thresh
        self._track_buffer = track_buffer
        self._iou_gate = iou_gate

        self._next_id = 1
        # tid -> {"bbox", "class_name", "score", "last_frame"}
        self._tracks: dict[int, dict] = {}
        self._centroid_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=_HISTORY_LEN))

    def update(self, detections: list[Detection], frame: np.ndarray, frame_id: int) -> list[TrackedObject]:
        dets = [d for d in detections if d.confidence >= self._track_thresh]

        # Build candidate (iou, tid, det_index) matches, then assign greedily.
        candidates = []
        for tid, info in self._tracks.items():
            for di, d in enumerate(dets):
                iou = _iou(info["bbox"], d.bbox)
                if iou >= self._iou_gate:
                    candidates.append((iou, tid, di))
        candidates.sort(reverse=True)

        matched_tids: set[int] = set()
        matched_dets: set[int] = set()
        det_to_tid: dict[int, int] = {}
        for iou, tid, di in candidates:
            if tid in matched_tids or di in matched_dets:
                continue
            matched_tids.add(tid)
            matched_dets.add(di)
            det_to_tid[di] = tid

        active: list[TrackedObject] = []
        for di, d in enumerate(dets):
            tid = det_to_tid.get(di)
            if tid is None:
                tid = self._next_id
                self._next_id += 1

            x1, y1, x2, y2 = d.bbox
            self._tracks[tid] = {
                "bbox": d.bbox,
                "class_name": d.class_name,
                "score": d.confidence,
                "last_frame": frame_id,
            }
            self._centroid_history[tid].append(((x1 + x2) // 2, (y1 + y2) // 2))

            active.append(TrackedObject(
                track_id=tid,
                class_name=d.class_name,
                bbox=d.bbox,
                confidence=d.confidence,
                frame_id=frame_id,
                centroid_history=list(self._centroid_history[tid]),
            ))

        self._evict_stale(frame_id)
        return active

    def get_history(self, track_id: int) -> list[tuple[int, int]]:
        return list(self._centroid_history.get(track_id, []))

    def _evict_stale(self, frame_id: int) -> None:
        stale = [tid for tid, info in self._tracks.items()
                 if frame_id - info["last_frame"] > self._track_buffer]
        for tid in stale:
            self._tracks.pop(tid, None)
            self._centroid_history.pop(tid, None)


def _iou(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)
