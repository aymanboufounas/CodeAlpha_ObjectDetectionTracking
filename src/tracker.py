from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from .detector import Detection


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    confidence: float
    xyxy: Tuple[int, int, int, int]


class ObjectTracker:
    """Deep SORT tracker wrapper."""

    def __init__(self, max_age: int = 30, n_init: int = 3):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=1.0,
            embedder="mobilenet",
            half=True,
            bgr=True,
        )

    @staticmethod
    def _to_deepsort(detections: Iterable[Detection]):
        output = []
        for det in detections:
            x1, y1, x2, y2 = det.xyxy
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)

            output.append(
                ([x1, y1, width, height], det.confidence, det.class_name)
            )
        return output

    def update(
        self,
        frame: np.ndarray,
        detections: Iterable[Detection],
    ) -> List[TrackedObject]:
        tracks = self.tracker.update_tracks(
            self._to_deepsort(detections),
            frame=frame,
        )

        tracked_objects: List[TrackedObject] = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            left, top, right, bottom = track.to_ltrb()
            class_name = str(track.get_det_class() or "object")

            confidence = 0.0
            det_conf = getattr(track, "det_conf", None)
            if det_conf is not None:
                confidence = float(det_conf)

            tracked_objects.append(
                TrackedObject(
                    track_id=int(track.track_id),
                    class_name=class_name,
                    confidence=confidence,
                    xyxy=(
                        int(left),
                        int(top),
                        int(right),
                        int(bottom),
                    ),
                )
            )

        return tracked_objects
