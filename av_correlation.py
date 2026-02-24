from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AVHeuristicResult:
    status: str
    audio_present: bool
    mouth_moving: bool
    mar_delta: float
    corr_score: Optional[float]


class AVCorrelationEngine:
    def __init__(
        self,
        mar_delta_threshold: float = 0.012,
        corr_window: int = 24,
        corr_threshold: float = 0.25,
    ) -> None:
        self.mar_delta_threshold = mar_delta_threshold
        self.corr_threshold = corr_threshold
        self.audio_window: deque[float] = deque(maxlen=corr_window)
        self.mouth_window: deque[float] = deque(maxlen=corr_window)
        self._last_mar = 0.0

    def _corr(self) -> Optional[float]:
        if len(self.audio_window) < 6 or len(self.audio_window) != len(self.mouth_window):
            return None
        a = np.array(self.audio_window, dtype=np.float32)
        m = np.array(self.mouth_window, dtype=np.float32)
        if np.std(a) < 1e-6 or np.std(m) < 1e-6:
            return None
        c = np.corrcoef(a, m)[0, 1]
        return None if np.isnan(c) else float(c)

    def update(self, audio_present: bool, audio_energy: float, mar_value: float) -> AVHeuristicResult:
        mar_delta = abs(mar_value - self._last_mar)
        self._last_mar = mar_value
        mouth_moving = mar_delta >= self.mar_delta_threshold

        self.audio_window.append(float(audio_energy))
        self.mouth_window.append(float(mar_delta))
        corr = self._corr()

        if audio_present and mouth_moving:
            status = "SYNC_OK"
        elif audio_present and (not mouth_moving):
            status = "AUDIO_ONLY"
        elif (not audio_present) and mouth_moving:
            status = "SILENT_SPEECH"
        else:
            status = "IDLE"

        if corr is not None and corr < self.corr_threshold and status == "SYNC_OK":
            status = "WEAK_SYNC"

        return AVHeuristicResult(
            status=status,
            audio_present=audio_present,
            mouth_moving=mouth_moving,
            mar_delta=mar_delta,
            corr_score=corr,
        )
