from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LipSyncVerificationResult:
    score: float
    passed: bool
    model_name: str


class LipSyncVerifier:
    """
    Verification-only layer.
    This class is intentionally lightweight and runs only on flagged segments.
    You can replace `_fallback_score` with SyncNet/Wav2Lip discriminator inference.
    """

    def __init__(self, threshold: float = 0.45, model_name: str = "fallback_av_sync") -> None:
        self.threshold = threshold
        self.model_name = model_name

    def _fallback_score(self, mouth_motion_series: np.ndarray, audio_energy_series: np.ndarray) -> float:
        if mouth_motion_series.size < 6 or audio_energy_series.size < 6:
            return 0.0
        mm = mouth_motion_series.astype(np.float32)
        ae = audio_energy_series.astype(np.float32)
        if np.std(mm) < 1e-6 or np.std(ae) < 1e-6:
            return 0.0
        corr = np.corrcoef(mm, ae)[0, 1]
        if np.isnan(corr):
            return 0.0
        return float((corr + 1.0) * 0.5)

    def verify_segment(
        self,
        mouth_motion_series: np.ndarray,
        audio_energy_series: np.ndarray,
        external_score: Optional[float] = None,
    ) -> LipSyncVerificationResult:
        score = float(external_score) if external_score is not None else self._fallback_score(mouth_motion_series, audio_energy_series)
        return LipSyncVerificationResult(score=score, passed=score >= self.threshold, model_name=self.model_name)
