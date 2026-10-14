from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    xyxy: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class YOLODetector:
    """YOLO object detector wrapper."""

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.40):
        if not 0 < confidence <= 1:
            raise ValueError("confidence must be in (0, 1].")
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.names = self.model.names

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if frame is None or frame.size == 0:
            raise ValueError("Cannot detect objects in an empty frame.")
        results = self.model.predict(source=frame, conf=self.confidence, verbose=False)
        if not results or results[0].boxes is None:
            return []

        detections: List[Detection] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().tolist())
            score = float(box.conf[0].cpu().item())
            class_id = int(box.cls[0].cpu().item())
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2, score)):
                continue
            if x2 <= x1 or y2 <= y1 or not 0 <= score <= 1:
                continue
            if isinstance(self.names, dict):
                class_name = str(self.names.get(class_id, class_id))
            else:
                class_name = str(self.names[class_id]) if 0 <= class_id < len(self.names) else str(class_id)
            detections.append(
                Detection(
                    xyxy=(x1, y1, x2, y2),
                    confidence=score,
                    class_id=class_id,
                    class_name=class_name,
                )
            )
        return detections
