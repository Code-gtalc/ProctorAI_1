from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class FaceOcclusionState:
    face_visibility_ratio: float
    occlusion_counter: int
    cooldown_active: bool
    cooldown_remaining: float
    model_type: str


class FaceOcclusionDetector:
    """Lightweight, real-time face occlusion detector."""

    def __init__(
        self,
        model_type: str,
        visibility_threshold: float = 0.6,
        consecutive_frames: int = 3,
        cooldown_s: float = 6.0,
    ) -> None:
        self.model_type = str(model_type)
        self.visibility_threshold = float(max(0.0, min(1.0, visibility_threshold)))
        self.consecutive_frames = int(max(1, consecutive_frames))
        self.cooldown_s = float(max(0.0, cooldown_s))

        self._occlusion_counter = 0
        self._cooldown_until = 0.0
        self._last_ratio = 1.0
        self._prev_bbox: Optional[tuple[int, int, int, int]] = None
        self._lock = threading.Lock()

    def _landmark_ratio(self, landmarks: Any) -> float:
        points = getattr(landmarks, "landmark", None)
        if not points:
            return 0.0
        total = len(points)
        visible = 0
        for lm in points:
            x = float(getattr(lm, "x", 0.0))
            y = float(getattr(lm, "y", 0.0))
            z = float(getattr(lm, "z", 0.0))
            vis = getattr(lm, "visibility", None)
            pres = getattr(lm, "presence", None)
            if vis is not None or pres is not None:
                v = float(vis) if vis is not None else 1.0
                p = float(pres) if pres is not None else 1.0
                if v >= 0.5 and p >= 0.5 and math.isfinite(x) and math.isfinite(y):
                    visible += 1
            else:
                if (0.0 <= x <= 1.0) and (0.0 <= y <= 1.0) and math.isfinite(z):
                    visible += 1
        return float(visible / max(1, total))

    def _bbox_ratio(self, face_bbox: Optional[tuple[int, int, int, int]], frame: np.ndarray) -> float:
        if face_bbox is None:
            self._prev_bbox = None
            return 0.0
        x0, y0, x1, y1 = face_bbox
        bw = max(1.0, float(x1 - x0))
        bh = max(1.0, float(y1 - y0))
        if self._prev_bbox is None:
            self._prev_bbox = face_bbox
            return 1.0

        px0, py0, px1, py1 = self._prev_bbox
        pbw = max(1.0, float(px1 - px0))
        pbh = max(1.0, float(py1 - py0))
        cx = (x0 + x1) * 0.5
        cy = (y0 + y1) * 0.5
        pcx = (px0 + px1) * 0.5
        pcy = (py0 + py1) * 0.5

        center_shift = math.hypot(cx - pcx, cy - pcy) / max(1.0, (bw + bh) * 0.5)
        area = bw * bh
        parea = pbw * pbh
        area_change = abs(area - parea) / max(1.0, parea)
        self._prev_bbox = face_bbox

        instability = min(1.0, (0.7 * center_shift) + (0.3 * area_change))
        return float(max(0.0, 1.0 - instability))

    def _snapshot(self, now: float) -> FaceOcclusionState:
        return FaceOcclusionState(
            face_visibility_ratio=float(self._last_ratio),
            occlusion_counter=int(self._occlusion_counter),
            cooldown_active=bool(now < self._cooldown_until),
            cooldown_remaining=float(max(0.0, self._cooldown_until - now)),
            model_type=self.model_type,
        )

    def state(self) -> FaceOcclusionState:
        with self._lock:
            return self._snapshot(time.time())

    def update(
        self,
        frame: np.ndarray,
        landmarks: Any = None,
        face_bbox: Optional[tuple[int, int, int, int]] = None,
        now: Optional[float] = None,
    ) -> str | None:
        t_now = float(now if now is not None else time.time())
        with self._lock:
            if landmarks is not None:
                self._last_ratio = self._landmark_ratio(landmarks)
            else:
                self._last_ratio = self._bbox_ratio(face_bbox, frame)

            if t_now < self._cooldown_until:
                self._occlusion_counter = 0
                return None

            if self._last_ratio < self.visibility_threshold:
                self._occlusion_counter += 1
            else:
                self._occlusion_counter = 0

            if self._occlusion_counter >= self.consecutive_frames:
                self._occlusion_counter = 0
                self._cooldown_until = t_now + self.cooldown_s
                print("Face Occlusion Detected")
                return "FACE_OCCLUDED"
            return None
