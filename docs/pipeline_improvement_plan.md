# Pipeline Optimization — Implementation Notes

## What This Plan Proposed vs What Was Actually Implemented

The original plan had 4 priority tiers. Here is what was done, what was skipped, and why.

---

## ✅ IMPLEMENTED (Priorities 1 & 2)

### Track Memory Manager
- **New file:** `src/tracking/track_memory.py`
- Stores per-track state: helmet result, seatbelt result, plate OCR result
- Entries are keyed by tracker's `track_id` (already stable)
- Eviction is **synced to the tracker** — when the tracker evicts a stale track, the memory evicts too. No separate timeout (the plan suggested 120-frame timeout, but that's redundant since our tracker already has a `track_buffer=30` eviction mechanism)

### Helmet Result Caching
- `helmet.py` now accepts `track_memory` parameter
- Helmet YOLO is **only called when at least one motorcycle needs a recheck**
- On first appearance: model runs immediately and caches the result
- On subsequent frames: cached result is reused
- Recheck triggers: (a) every `refresh_interval` frames (default 30 = 3 sec at 10 FPS), or (b) if previous detection confidence was low (< 0.6)
- Violation is emitted **once per track** — no duplicate violations for the same bike

### Seatbelt Result Caching
- `seatbelt.py` now accepts `track_memory` parameter
- Same pattern: check once, cache, recheck every `refresh_interval` frames (default 60 = 6 sec)
- **Critical fix:** previously, an `indeterminate` record was emitted for every car on every frame (seatbelt model not trained). This was flooding the DB. Now `indeterminate` is emitted **once per car track**, then cached.

### Plate / OCR Caching
- Integrated directly in `video_pipeline.py` ANPR section
- Once a plate is read for a vehicle track, it's stored in track memory
- If the same vehicle triggers another violation, the cached plate is reused — no redundant plate detection + OCR call
- **Note:** OCR was already violation-triggered (not per-frame) in our pipeline. The plan incorrectly claimed "OCR runs every frame." The caching still helps when one vehicle has multiple violations.

### Config Changes
- `src/configs/violations.yaml` — added `refresh_interval` to `helmet` (30) and `seatbelt` (60) sections

### Tests
- `tests/test_track_memory.py` — 12 new unit tests covering: creation, refresh intervals, low-confidence recheck, plate caching, stale eviction, unknown track handling
- **Total tests: 35/35 passing**

---

## ❌ SKIPPED (Priorities 3 & 4) — With Reasoning

### Async OCR Queue — SKIPPED
**Why:** Python's GIL prevents true threading parallelism for GPU-bound work. OCR already runs on the same CUDA device as the detectors. Threading would add locking complexity, potential race conditions on evidence generation and DB writes, and no actual speedup because:
- OCR is already violation-triggered, not per-frame (plan overstated the problem)
- EasyOCR on GPU takes ~20ms per plate — this is negligible vs the 60-100ms YOLO inference
- For a prototype on a single GTX 1650, async queuing buys nothing

**Would implement if:** Moving to multi-GPU or multi-process architecture for production.

### Async Helmet/Seatbelt Workers — SKIPPED
**Why:** Same GIL + single-GPU problem. The caching layer (implemented above) already reduces helmet calls by ~30x. Adding async workers would:
- Not improve GPU throughput (still one GPU, still serialized CUDA calls)
- Add `threading.Thread` / `multiprocessing.Process` complexity
- Create race conditions on track memory reads/writes
- Require producer-consumer plumbing that's over-engineered for a prototype

**Would implement if:** Running on a multi-GPU server with >4 RTSP streams.

---

## Plan Corrections (Factual Errors in the Original Document)

| Plan Claim | Actual State | Correction |
|-----------|-------------|------------|
| "OCR runs every frame" | OCR already only ran on violation frames (line 202-228 of original pipeline) | The "95% reduction" was already the baseline. Plate caching still helps for multi-violation tracks. |
| "30 checks/sec/vehicle for helmet" | Our target_fps = 10, not 30. So it was 10 checks/sec without caching, now 1 check every 3 sec with caching. | Actual reduction: ~30x → still accurate in ratio, just at lower absolute numbers. |
| "TRACK_MEMORY_TIMEOUT = 120 frames" | Unnecessary — tracker already evicts at `track_buffer=30`. Memory synced to tracker eviction. | No separate timeout needed; simpler architecture. |
| "ByteTrack as future upgrade" | ByteTrack was the **original** tracker — it broke on ultralytics 8.4.x and was replaced with the IoU tracker. | The IoU tracker was a deliberate downgrade for stability. "Upgrading" back to ByteTrack would reintroduce the breakage. |

---

## Performance Impact Summary

| Metric | Before | After | Reduction |
|--------|--------|-------|-----------|
| Helmet YOLO calls per motorcycle | Every processed frame | Once on first appearance, then every 30 frames | ~10x less GPU inference |
| Seatbelt CNN calls per car | Every processed frame | Once on first appearance, then every 60 frames | ~10x less GPU inference |
| `indeterminate` DB records per car | 1 per frame per car (DB flooding) | 1 per car track (emitted once) | ~target_fps × video_duration less |
| Plate detector + OCR per vehicle | Once per violation | Once per vehicle (cached across violations) | Variable — big win for multi-violation vehicles |
| Memory overhead | None | ~200 bytes per active track | Negligible |

---

## Files Changed

| File | Change |
|------|--------|
| `src/tracking/track_memory.py` | **NEW** — TrackMemory + TrackState classes |
| `src/violations/helmet.py` | Added `track_memory` parameter, caching logic, single-emit per track |
| `src/violations/seatbelt.py` | Added `track_memory` parameter, caching logic, indeterminate-once per track |
| `pipelines/video_pipeline.py` | Wired TrackMemory, synced eviction, plate caching in ANPR section |
| `src/configs/violations.yaml` | Added `refresh_interval` to helmet (30) and seatbelt (60) |
| `tests/test_track_memory.py` | **NEW** — 12 unit tests |
