from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def _frame_signal(signal: np.ndarray, frame_size: int, hop: int) -> list[np.ndarray]:
    if len(signal) < frame_size:
        return []
    return [signal[i : i + frame_size] for i in range(0, len(signal) - frame_size + 1, hop)]


def simple_speaker_embedding(audio: np.ndarray, sr: int) -> np.ndarray:
    # Lightweight fallback embedding: log-band spectral profile + temporal stats.
    audio = audio.astype(np.float32)
    if audio.size == 0:
        return np.zeros(64, dtype=np.float32)
    audio = audio - np.mean(audio)
    audio = audio / (np.std(audio) + 1e-8)

    frames = _frame_signal(audio, frame_size=int(0.03 * sr), hop=int(0.01 * sr))
    if not frames:
        return np.zeros(64, dtype=np.float32)

    specs = []
    for frame in frames:
        win = frame * np.hanning(len(frame))
        spectrum = np.abs(np.fft.rfft(win))
        specs.append(np.log1p(spectrum))
    spec = np.stack(specs, axis=0)

    # 32 coarse frequency bins.
    bins = np.array_split(spec.mean(axis=0), 32)
    band_means = np.array([float(np.mean(b)) for b in bins], dtype=np.float32)
    band_stds = np.array([float(np.std(b)) for b in bins], dtype=np.float32)
    emb = np.concatenate([band_means, band_stds], axis=0)
    return _l2_normalize(emb.astype(np.float32))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = _l2_normalize(a)
    b_n = _l2_normalize(b)
    return float(np.dot(a_n, b_n))


@dataclass
class SpeakerVerificationResult:
    similarity: Optional[float]
    is_mismatch: bool
    has_reference: bool


class SpeakerVerifier:
    def __init__(self, sample_rate: int, similarity_threshold: float = 0.72) -> None:
        self.sample_rate = sample_rate
        self.similarity_threshold = similarity_threshold
        self.reference_embedding: Optional[np.ndarray] = None

    def bootstrap_reference(self, audio: np.ndarray) -> None:
        emb = simple_speaker_embedding(audio, self.sample_rate)
        if np.linalg.norm(emb) > 1e-6:
            self.reference_embedding = emb

    def verify(self, audio: np.ndarray, audio_present: bool) -> SpeakerVerificationResult:
        if not audio_present:
            return SpeakerVerificationResult(None, False, self.reference_embedding is not None)

        emb = simple_speaker_embedding(audio, self.sample_rate)
        if self.reference_embedding is None:
            self.bootstrap_reference(audio)
            return SpeakerVerificationResult(None, False, False)

        sim = cosine_similarity(self.reference_embedding, emb)
        mismatch = sim < self.similarity_threshold
        return SpeakerVerificationResult(similarity=sim, is_mismatch=mismatch, has_reference=True)
