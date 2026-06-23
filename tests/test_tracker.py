"""Tests for the self-contained IoU tracker, including the centroid fallback
that keeps IDs stable when frame subsampling makes boxes stop overlapping."""

from src.models import Detection
from src.components.tracking.tracker import Tracker


def _det(class_name, bbox, conf=0.9):
    return Detection(class_name=class_name, confidence=conf, bbox=bbox, frame_id=0)


class TestTracker:
    def test_same_box_keeps_id(self):
        tr = Tracker()
        a = tr.update([_det("car", (0, 0, 100, 100))], None, 1)
        b = tr.update([_det("car", (0, 0, 100, 100))], None, 2)
        assert a[0].track_id == b[0].track_id

    def test_large_jump_no_iou_keeps_id_via_centroid(self):
        # Displacement of 120px on a 100px box => IoU == 0, but within the
        # centroid gate (1.5 * 100 = 150), so the ID must be preserved.
        tr = Tracker()
        a = tr.update([_det("car", (0, 0, 100, 100))], None, 1)
        b = tr.update([_det("car", (120, 0, 220, 100))], None, 2)
        assert a[0].track_id == b[0].track_id

    def test_huge_jump_beyond_gate_gets_new_id(self):
        tr = Tracker()
        a = tr.update([_det("car", (0, 0, 100, 100))], None, 1)
        b = tr.update([_det("car", (400, 400, 500, 500))], None, 2)
        assert a[0].track_id != b[0].track_id

    def test_centroid_fallback_respects_class(self):
        # A near-overlapping detection of a different class must not steal the ID.
        tr = Tracker()
        a = tr.update([_det("car", (0, 0, 100, 100))], None, 1)
        b = tr.update([_det("truck", (120, 0, 220, 100))], None, 2)
        assert a[0].track_id != b[0].track_id
