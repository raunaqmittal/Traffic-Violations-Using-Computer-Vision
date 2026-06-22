"""Tests for TrackMemory inference caching."""

import pytest
from src.components.tracking.track_memory import TrackMemory, TrackState


class TestTrackMemory:
    def test_get_or_create_new_track(self):
        mem = TrackMemory()
        state = mem.get_or_create(1, "motorcycle")
        assert isinstance(state, TrackState)
        assert state.vehicle_type == "motorcycle"
        assert state.helmet_checked is False

    def test_get_or_create_existing_returns_same(self):
        mem = TrackMemory()
        s1 = mem.get_or_create(1, "motorcycle")
        s1.helmet_checked = True
        s2 = mem.get_or_create(1, "motorcycle")
        assert s2.helmet_checked is True  # same object

    def test_needs_helmet_check_unchecked(self):
        mem = TrackMemory(helmet_refresh_interval=30)
        mem.get_or_create(1, "motorcycle")
        assert mem.needs_helmet_check(1, frame_id=0) is True

    def test_needs_helmet_check_cached(self):
        mem = TrackMemory(helmet_refresh_interval=30)
        state = mem.get_or_create(1, "motorcycle")
        state.helmet_checked = True
        state.helmet_status = "ok"
        state.helmet_confidence = 0.92   # high confidence → trust it
        state.last_helmet_frame = 10
        assert mem.needs_helmet_check(1, frame_id=15) is False  # only 5 frames later

    def test_needs_helmet_check_ok_low_conf_occlusion(self):
        """Low confidence 'ok' means model wasn't sure — recheck immediately."""
        mem = TrackMemory(helmet_refresh_interval=30, low_conf_recheck_threshold=0.70)
        state = mem.get_or_create(1, "motorcycle")
        state.helmet_checked = True
        state.helmet_status = "ok"       # claimed ok ...
        state.helmet_confidence = 0.55   # ... but wasn't confident → occlusion risk
        state.last_helmet_frame = 10
        assert mem.needs_helmet_check(1, frame_id=12) is True   # only 2 frames later

    def test_needs_helmet_check_no_helmet_low_conf(self):
        """Low confidence 'no_helmet' also rechecks (reconfirm before treating as definitive)."""
        mem = TrackMemory(helmet_refresh_interval=30, low_conf_recheck_threshold=0.70)
        state = mem.get_or_create(1, "motorcycle")
        state.helmet_checked = True
        state.helmet_status = "no_helmet"
        state.helmet_confidence = 0.62   # below threshold
        state.last_helmet_frame = 10
        assert mem.needs_helmet_check(1, frame_id=12) is True

    def test_needs_helmet_check_expired(self):
        mem = TrackMemory(helmet_refresh_interval=30)
        state = mem.get_or_create(1, "motorcycle")
        state.helmet_checked = True
        state.helmet_status = "ok"
        state.last_helmet_frame = 10
        assert mem.needs_helmet_check(1, frame_id=41) is True  # 31 frames later

    def test_needs_seatbelt_check_unchecked(self):
        mem = TrackMemory(seatbelt_refresh_interval=60)
        mem.get_or_create(1, "car")
        assert mem.needs_seatbelt_check(1, frame_id=0) is True

    def test_needs_seatbelt_check_cached(self):
        mem = TrackMemory(seatbelt_refresh_interval=60)
        state = mem.get_or_create(1, "car")
        state.seatbelt_checked = True
        state.last_seatbelt_frame = 10
        assert mem.needs_seatbelt_check(1, frame_id=50) is False  # only 40 frames

    def test_needs_seatbelt_check_expired(self):
        mem = TrackMemory(seatbelt_refresh_interval=60)
        state = mem.get_or_create(1, "car")
        state.seatbelt_checked = True
        state.last_seatbelt_frame = 10
        assert mem.needs_seatbelt_check(1, frame_id=71) is True  # 61 frames later

    def test_plate_caching(self):
        mem = TrackMemory()
        state = mem.get_or_create(1, "car")
        assert mem.has_plate(1) is False
        state.plate_number = "MH12AB1234"
        state.plate_confidence = 0.92
        assert mem.has_plate(1) is True

    def test_evict_stale(self):
        mem = TrackMemory()
        mem.get_or_create(1, "car")
        mem.get_or_create(2, "motorcycle")
        mem.get_or_create(3, "bus")
        # Only track 2 is still active
        mem.evict_stale(active_track_ids={2})
        assert mem.get(1) is None
        assert mem.get(2) is not None
        assert mem.get(3) is None

    def test_unknown_track_needs_check(self):
        mem = TrackMemory()
        # Track never seen — both should return True
        assert mem.needs_helmet_check(999, frame_id=0) is True
        assert mem.needs_seatbelt_check(999, frame_id=0) is True
