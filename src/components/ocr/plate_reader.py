"""
License plate OCR using EasyOCR.

We use EasyOCR (torch-based) rather than PaddleOCR: it runs on the same CUDA
GPU as the detectors, installs cleanly on Windows/Python 3.11, and avoids the
PaddlePaddle 3.x oneDNN runtime bug. Accepts a plate crop (numpy BGR array)
and returns the recognized text and confidence.
"""

import re
import numpy as np
from dataclasses import dataclass


@dataclass
class PlateReadResult:
    text: str
    confidence: float
    is_partial: bool
    is_valid: bool = False   # matches the Indian plate format after normalisation


# Indian registration plate: 2-letter state, 1-2 digit RTO, 1-3 letter series,
# 1-4 digit number. e.g. MH12AB1234, KA01A1234, DL3CAB1234.
_PLATE_RE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$")


def normalize_plate(raw: str) -> tuple[str, bool]:
    """
    Uppercase, strip non-alphanumerics, and test against the Indian plate
    format. Returns (cleaned_text, is_valid).
    """
    cleaned = re.sub(r"[^A-Z0-9]", "", raw.upper())
    return cleaned, bool(_PLATE_RE.match(cleaned))


class PlateReader:
    def __init__(self, lang: str = "en", use_gpu: bool = False):
        import easyocr
        # allowlist keeps recognition to plate-relevant characters.
        self._reader = easyocr.Reader([lang], gpu=use_gpu)

    def read(self, plate_crop: np.ndarray) -> PlateReadResult | None:
        if plate_crop is None or plate_crop.size == 0:
            return None

        # detail=1 -> list of (bbox, text, confidence)
        results = self._reader.readtext(
            plate_crop,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        )
        if not results:
            return None

        texts, confidences = [], []
        for _bbox, text, conf in results:
            cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
            if cleaned:
                texts.append(cleaned)
                confidences.append(float(conf))

        if not texts:
            return None

        combined_text = "".join(texts)
        avg_conf = sum(confidences) / len(confidences)
        normalized, is_valid = normalize_plate(combined_text)
        is_partial = len(normalized) < 4

        return PlateReadResult(
            text=normalized,
            confidence=avg_conf,
            is_partial=is_partial,
            is_valid=is_valid,
        )
