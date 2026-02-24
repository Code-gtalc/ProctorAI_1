from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))


def _log_energy(x: np.ndarray) -> float:
    return float(np.log1p(_rms(x)))


def _zcr(x: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    s = np.sign(x)
    return float(np.mean(np.abs(np.diff(s)) > 0))


def _spectral_features(x: np.ndarray, sr: int) -> tuple[float, float, np.ndarray]:
    if x.size < 8:
        return 0.0, 0.0, np.zeros((16,), dtype=np.float32)
    win = x.astype(np.float32) * np.hanning(x.size)
    mag = np.abs(np.fft.rfft(win))
    if mag.size < 2:
        return 0.0, 0.0, np.zeros((16,), dtype=np.float32)
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sr)
    centroid = float(np.sum(freqs * mag) / (np.sum(mag) + 1e-8))
    flatness = float(np.exp(np.mean(np.log(mag + 1e-8))) / (np.mean(mag) + 1e-8))
    bands = np.array_split(np.log1p(mag), 16)
    band_vec = np.array([float(np.mean(b)) for b in bands], dtype=np.float32)
    n = np.linalg.norm(band_vec)
    return centroid, flatness, (band_vec / n if n > 1e-8 else band_vec)


def _corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size < 6 or b.size < 6 or a.size != b.size:
        return None
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return None
    c = np.corrcoef(a, b)[0, 1]
    if np.isnan(c):
        return None
    return float(c)


@dataclass
class AudioSyncResult:
    score: float
    flags: list[str]
    energy_mar_corr: float
    offset_ms: Optional[float]
    whisper: bool
    viseme_mismatch_count: int
    playback_suspected: bool


class AudioSyncVerifier:
    """
    Real-time, explainable audio-sync heuristics.
    Heavy model verification must be triggered externally when repeated failures occur.
    """

    def __init__(
        self,
        sample_rate: int,
        corr_window_s: float = 0.8,
        offset_desync_ms: float = 200.0,
        offset_events_for_flag: int = 3,
        audio_high_energy: float = 0.020,
        low_motion_thr: float = 0.008,
        whisper_energy_thr: float = 0.010,
        whisper_mar_var_thr: float = 0.005,
        viseme_mismatch_thr: int = 8,
        playback_repeat_sim_thr: float = 0.985,
    ) -> None:
        self.sample_rate = sample_rate
        self.corr_window_s = corr_window_s
        self.offset_desync_ms = offset_desync_ms
        self.offset_events_for_flag = offset_events_for_flag
        self.audio_high_energy = audio_high_energy
        self.low_motion_thr = low_motion_thr
        self.whisper_energy_thr = whisper_energy_thr
        self.whisper_mar_var_thr = whisper_mar_var_thr
        self.viseme_mismatch_thr = viseme_mismatch_thr
        self.playback_repeat_sim_thr = playback_repeat_sim_thr

        self.energy_hist: deque[float] = deque(maxlen=90)
        self.motion_hist: deque[float] = deque(maxlen=90)
        self.vad_hist: deque[bool] = deque(maxlen=90)
        self.timestamp_hist: deque[float] = deque(maxlen=90)
        self.mar_hist: deque[float] = deque(maxlen=90)
        self.spectrum_hist: deque[np.ndarray] = deque(maxlen=18)
        self.zcr_hist: deque[float] = deque(maxlen=30)

        self._last_mar = 0.0
        self._last_audio_onset_t: Optional[float] = None
        self._last_mouth_onset_t: Optional[float] = None
        self._desync_event_count = 0
        self._viseme_mismatch_count = 0

    def _expected_viseme(self, centroid: float, zcr: float, energy: float) -> str:
        # Coarse, explainable mapping (phoneme-like buckets).
        if energy < self.whisper_energy_thr:
            return "quiet"
        if centroid < 1400 and zcr < 0.10:
            return "labial_close"   # /p,b,m/ tendency
        if centroid > 2600 and zcr > 0.15:
            return "fricative"       # /f,v,s/ tendency
        return "vowel_open"

    def _observed_viseme(self, mar: float) -> str:
        if mar < 0.06:
            return "labial_close"
        if mar > 0.17:
            return "vowel_open"
        return "fricative"

    def update(
        self,
        timestamp_s: float,
        audio_chunk: np.ndarray,
        audio_present: bool,
        mar_value: float,
        mouth_occluded: bool,
    ) -> AudioSyncResult:
        energy = _rms(audio_chunk)
        log_energy = _log_energy(audio_chunk)
        mouth_motion = abs(mar_value - self._last_mar)
        self._last_mar = mar_value

        centroid, flatness, spec = _spectral_features(audio_chunk, self.sample_rate)
        zcr = _zcr(audio_chunk)

        self.energy_hist.append(log_energy)
        self.motion_hist.append(mouth_motion)
        self.vad_hist.append(audio_present)
        self.timestamp_hist.append(timestamp_s)
        self.mar_hist.append(mar_value)
        self.spectrum_hist.append(spec)
        self.zcr_hist.append(zcr)

        # Rolling correlation over ~0.5-1.0s, bounded by available samples.
        corr_n = max(8, min(len(self.energy_hist), int(self.corr_window_s * 30)))
        corr_val = _corr(
            np.array(list(self.energy_hist)[-corr_n:], dtype=np.float32),
            np.array(list(self.motion_hist)[-corr_n:], dtype=np.float32),
        )
        energy_mar_corr = 0.0 if corr_val is None else float((corr_val + 1.0) * 0.5)

        # Onset-based AV delay checks.
        if audio_present and energy > self.audio_high_energy and (len(self.energy_hist) >= 2 and self.energy_hist[-2] < self.energy_hist[-1]):
            self._last_audio_onset_t = timestamp_s
        if mouth_motion > max(self.low_motion_thr * 1.8, 0.012):
            self._last_mouth_onset_t = timestamp_s

        offset_ms: Optional[float] = None
        if self._last_audio_onset_t is not None and self._last_mouth_onset_t is not None:
            offset_ms = (self._last_mouth_onset_t - self._last_audio_onset_t) * 1000.0
            if abs(offset_ms) > self.offset_desync_ms:
                self._desync_event_count += 1
            else:
                self._desync_event_count = max(0, self._desync_event_count - 1)

        # Whisper heuristic.
        voiced_ratio = float(np.mean(self.vad_hist)) if self.vad_hist else 0.0
        mar_var = float(np.std(np.array(self.mar_hist, dtype=np.float32))) if self.mar_hist else 0.0
        whisper = (voiced_ratio > 0.65) and (energy < self.whisper_energy_thr) and (mar_var < self.whisper_mar_var_thr)

        # Phoneme-viseme consistency (rule-based coarse classes).
        exp_viseme = self._expected_viseme(centroid, zcr, energy)
        obs_viseme = self._observed_viseme(mar_value)
        if exp_viseme not in ("quiet",) and exp_viseme != obs_viseme:
            self._viseme_mismatch_count += 1
        else:
            self._viseme_mismatch_count = max(0, self._viseme_mismatch_count - 1)

        # Playback detection: repeated spectra + low spectral variability + low jitter proxy.
        playback_suspected = False
        if len(self.spectrum_hist) >= 6:
            cur = self.spectrum_hist[-1]
            sims = [float(np.dot(cur, prev)) for prev in list(self.spectrum_hist)[-6:-1]]
            repeated = max(sims) > self.playback_repeat_sim_thr
            spectral_variance = float(np.var(np.stack(list(self.spectrum_hist)[-6:], axis=0)))
            low_zcr_jitter = float(np.std(np.array(list(self.zcr_hist)[-12:], dtype=np.float32))) < 0.01 if len(self.zcr_hist) >= 12 else False
            playback_suspected = repeated and (spectral_variance < 0.010) and low_zcr_jitter and (flatness < 0.25)

        flags: list[str] = []

        # A/F: audio with little/no lip motion.
        recent_motion = float(np.mean(np.array(list(self.motion_hist)[-8:], dtype=np.float32))) if len(self.motion_hist) >= 4 else mouth_motion
        if audio_present and (energy > self.audio_high_energy) and (recent_motion < self.low_motion_thr):
            flags.append("AUDIO_WITHOUT_LIP_MOTION")
        if audio_present and (recent_motion < self.low_motion_thr) and (not mouth_occluded):
            flags.append("AUDIO_ONLY_SPEECH")

        # B: sustained AV desync.
        if self._desync_event_count >= self.offset_events_for_flag:
            flags.append("AV_DESYNC")

        # C: whisper.
        if whisper:
            flags.append("WHISPER_DETECTED")

        # D: viseme mismatch.
        if self._viseme_mismatch_count > self.viseme_mismatch_thr:
            flags.append("PHONEME_VISEME_MISMATCH")

        # E: playback suspicion.
        if playback_suspected:
            flags.append("POSSIBLE_AUDIO_PLAYBACK")

        # Confidence score.
        offset_penalty = min(1.0, (abs(offset_ms) / self.offset_desync_ms) - 1.0) if offset_ms is not None and abs(offset_ms) > self.offset_desync_ms else 0.0
        whisper_penalty = 0.25 if whisper else 0.0
        viseme_penalty = min(0.45, self._viseme_mismatch_count / max(1.0, float(self.viseme_mismatch_thr * 2)))
        score = energy_mar_corr - (0.35 * offset_penalty) - whisper_penalty - viseme_penalty
        score = float(np.clip(score, 0.0, 1.0))

        return AudioSyncResult(
            score=score,
            flags=sorted(set(flags)),
            energy_mar_corr=energy_mar_corr,
            offset_ms=offset_ms,
            whisper=whisper,
            viseme_mismatch_count=self._viseme_mismatch_count,
            playback_suspected=playback_suspected,
        )
