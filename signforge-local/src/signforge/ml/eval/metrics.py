"""
Metrics calculation for SignForge evaluation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from PIL import Image
from signforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ImageMetrics:
    """Metrics for a single generated image."""
    sharpness: float = 0.0
    contrast: float = 0.0
    brightness: float = 0.0
    saturation: float = 0.0
    text_clarity: float = 0.0
    composition: float = 0.0
    technical_score: float = 0.0
    aesthetic_score: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class MetricsCalculator:
    """Calculates quality metrics for generated images."""

    def calculate(self, image: Image.Image) -> ImageMetrics:
        if image.mode != "RGB":
            image = image.convert("RGB")
        arr = np.array(image).astype(float)
        
        sharpness = self._sharpness(arr)
        contrast = self._contrast(arr)
        brightness = self._brightness(arr)
        saturation = self._saturation(arr)
        composition = self._composition(arr)
        
        return ImageMetrics(
            sharpness=sharpness, contrast=contrast,
            brightness=brightness, saturation=saturation,
            composition=composition,
            technical_score=(sharpness + contrast + brightness) / 3,
            aesthetic_score=(composition + saturation) / 2,
        )

    def _sharpness(self, arr: np.ndarray) -> float:
        gray = np.mean(arr, axis=2)
        lap = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        h, w = gray.shape
        result = np.zeros((h-2, w-2))
        for i in range(1, h-1):
            for j in range(1, w-1):
                result[i-1, j-1] = np.sum(gray[i-1:i+2, j-1:j+2] * lap)
        return float(min(np.var(result) / 1000, 1.0))

    def _contrast(self, arr: np.ndarray) -> float:
        return float(min(np.std(np.mean(arr, axis=2)) / 80, 1.0))

    def _brightness(self, arr: np.ndarray) -> float:
        mean_b = np.mean(np.mean(arr, axis=2))
        return float(1 - abs(mean_b - 128) / 128)

    def _saturation(self, arr: np.ndarray) -> float:
        r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        with np.errstate(divide='ignore', invalid='ignore'):
            sat = np.where(max_c > 0, (max_c - min_c) / max_c, 0)
        return float(min(np.mean(sat) + 0.5, 1.0))

    def _composition(self, arr: np.ndarray) -> float:
        h, w = arr.shape[:2]
        gray = np.mean(arr, axis=2)
        mid_h, mid_w = h // 2, w // 2
        quads = [gray[:mid_h, :mid_w], gray[:mid_h, mid_w:],
                 gray[mid_h:, :mid_w], gray[mid_h:, mid_w:]]
        var = np.std([np.mean(q) for q in quads])
        if var < 10:
            return 0.5
        elif var > 50:
            return max(0.3, 1 - var / 100)
        return float(min(0.7 + (var - 10) / 120, 1.0))

    def batch_calculate(self, images: list[Image.Image]) -> dict[str, Any]:
        if not images:
            return {"count": 0}
        metrics = [self.calculate(img) for img in images]
        def summarize(vals):
            return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        return {
            "count": len(images),
            "sharpness": summarize([m.sharpness for m in metrics]),
            "technical_score": summarize([m.technical_score for m in metrics]),
        }
