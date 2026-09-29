from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

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
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.names = self.model.names

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False,
        )

        detections: List[Detection] = []

        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            confidence = float(box.conf[0].cpu().item())
            class_id = int(box.cls[0].cpu().item())
            class_name = str(self.names[class_id])

            detections.append(
                Detection(
                    xyxy=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                )
            )

        return detections
