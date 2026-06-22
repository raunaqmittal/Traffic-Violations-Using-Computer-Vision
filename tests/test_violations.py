"""Tests for rule-based violation modules."""

import pytest
from unittest.mock import patch
from src.models import TrackedObject, ViolationRecord


def _make_track(track_id: int, class_name: str, bbox: tuple, history: list | None = None) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        class_name=class_name,
        bbox=bbox,
        confidence=0.9,
        frame_id=1,
        centroid_history=history or [],
    )


# ── Triple Riding ───────────────────────────────────────────────────────────

class TestTripleRiding:
    def test_three_riders_flagged(self):
        from src.components.violations import triple_riding
        moto = _make_track(1, "motorcycle", (100, 100, 300, 300))
        p1 = _make_track(2, "person", (110, 110, 200, 200))
        p2 = _make_track(3, "person", (150, 150, 220, 220))
        p3 = _make_track(4, "person", (120, 120, 180, 280))
        result = triple_riding.check([moto, p1, p2, p3], frame_id=1)
        assert len(result) == 1
        assert result[0].violation_type == "triple_riding"

    def test_two_riders_not_flagged(self):
        from src.components.violations import triple_riding
        moto = _make_track(1, "motorcycle", (100, 100, 300, 300))
        p1 = _make_track(2, "person", (110, 110, 200, 200))
        p2 = _make_track(3, "person", (150, 150, 220, 220))
        result = triple_riding.check([moto, p1, p2], frame_id=1)
        assert len(result) == 0

    def test_no_motorcycle_no_violation(self):
        from src.components.violations import triple_riding
        tracks = [
            _make_track(i, "person", (10 * i, 10 * i, 100 + 10 * i, 100 + 10 * i))
            for i in range(4)
        ]
        result = triple_riding.check(tracks, frame_id=1)
        assert len(result) == 0


# ── Wrong Side ──────────────────────────────────────────────────────────────

class TestWrongSide:
    def test_wrong_direction_flagged(self):
        from src.components.violations import wrong_side
        # allowed = 90 (up), vehicle moving right = 0 deg
        history = [(100 + i * 5, 300) for i in range(15)]
        track = _make_track(1, "car", (100, 280, 200, 320), history=history)
        result = wrong_side.check(
            [track], frame_id=50,
            allowed_direction_deg=90,
            direction_tolerance_deg=30,
            min_track_frames=8,
            consecutive_wrong_frames=5,
        )
        assert len(result) == 1

    def test_correct_direction_not_flagged(self):
        from src.components.violations import wrong_side
        # allowed = 90 (up, i.e. y decreasing in image coords)
        history = [(200, 300 - i * 5) for i in range(15)]
        track = _make_track(1, "car", (180, 250, 220, 270), history=history)
        result = wrong_side.check(
            [track], frame_id=50,
            allowed_direction_deg=90,
            direction_tolerance_deg=30,
            min_track_frames=8,
            consecutive_wrong_frames=5,
        )
        assert len(result) == 0


# ── Classifier routing ──────────────────────────────────────────────────────

class TestClassifierRouting:
    def test_high_confidence_auto_flagged(self):
        from src.components.violations.classifier import route
        record = ViolationRecord(
            violation_type="triple_riding",
            confidence=1.0,
            vehicle_id=1,
            bbox=(0, 0, 100, 100),
            timestamp="2026-01-01T00:00:00",
            frame_id=1,
        )
        result = route(record)
        assert result.status == "auto_flagged"

    def test_indeterminate_passes_through(self):
        from src.components.violations.classifier import route
        record = ViolationRecord(
            violation_type="seatbelt",
            confidence=0.0,
            vehicle_id=1,
            bbox=(0, 0, 100, 100),
            timestamp="2026-01-01T00:00:00",
            frame_id=1,
            status="indeterminate",
        )
        result = route(record)
        assert result.status == "indeterminate"
