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
        is_partial = len(combined_text) < 4

        return PlateReadResult(
            text=combined_text,
            confidence=avg_conf,
            is_partial=is_partial,
        )
