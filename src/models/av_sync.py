"""VeriSight v2 — Deliverable C: Voice + Video Lip-Sync / Audio-Visual Correspondence Module.

Laptop 2 (Audio/ASR/AV-Sync Engineer) owns this file.
DO NOT MODIFY from Laptop 1 (Piyush / Vision-Text-Fusion-UI branch).

Computes av_sync_score in [0, 1] where:
  1.0 = perfectly synchronized (real or consistently dubbed)
  0.0 = misaligned / swapped audio (deepfake indicator)

Also detects contradiction: when score is low, it raises an AV consistency warning
used by Layer 2 Cross-Modal Verification in the final fusion pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AVSyncResult:
    """Output of the AV correspondence module."""

    av_sync_score: Optional[float]
    """Score in [0, 1]: higher = better sync. None if computation not possible."""

    is_av_contradiction: bool
    """True when av_sync_score < threshold → signals possible audio-swap deepfake."""

    contradiction_threshold: float
    """Threshold used to decide contradiction (default 0.40)."""

    reason: Optional[str]
    """Human-readable explanation if score is None or contradiction detected."""

    mouth_motion_energy: Optional[float]
    """Mean mouth-region motion energy across sampled frames (normalized)."""

    audio_rms_mean: Optional[float]
    """Mean RMS energy of the audio waveform."""

    mfcc_correlation: Optional[float]
    """Pearson correlation between audio-MFCC envelope and mouth-motion envelope."""

    n_frames_analyzed: int = 0
    """Number of video frames that were successfully analyzed."""


class AVSyncDetector:
    """Lip-Sync / Audio-Visual Correspondence checker for VeriSight v2 (Layer 2).

    Algorithm:
      1. Sample N frames from the video clip (default every ~100ms).
      2. Detect face landmarks → crop mouth region (or use full-face fallback).
      3. Compute frame-to-frame mouth region optical flow energy as a 1D signal.
      4. Extract 16 kHz audio → compute RMS envelope + MFCC-1 envelope.
      5. Align both 1D signals via resampling and compute Pearson r.
      6. Map r → av_sync_score in [0, 1] via a sigmoid-like normalizer.

    Requires: opencv-python-headless, numpy, scipy, librosa (optional for MFCC).
    Falls back gracefully when no face/audio is detected.
    """

    def __init__(
        self,
        sample_fps: float = 10.0,
        contradiction_threshold: float = 0.40,
        face_detection_method: str = "haarcascade",  # "haarcascade" | "mediapipe"
    ):
        self.sample_fps = sample_fps
        self.contradiction_threshold = contradiction_threshold
        self.face_detection_method = face_detection_method
        self._face_detector = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        video_path: str | Path,
        audio_path: Optional[str | Path] = None,
        sample_rate: int = 16000,
    ) -> AVSyncResult:
        """Compute AV sync score for a video (audio extracted from video if audio_path is None).

        Args:
            video_path: Path to the video clip (.mp4 / .avi / .mkv).
            audio_path: Optional separate audio file. If None, audio is extracted from video.
            sample_rate: Target audio sample rate (16000 Hz default).

        Returns:
            AVSyncResult with sync score, contradiction flag, and diagnostic fields.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            return self._null_result(f"Video not found: {video_path}")

        try:
            import cv2  # noqa: F401
        except ImportError:
            return self._null_result("opencv-python-headless not installed; cannot compute AV sync.")

        # 1. Extract mouth-motion signal from video frames
        mouth_signal, n_frames = self._extract_mouth_motion(video_path)
        if mouth_signal is None or len(mouth_signal) < 3:
            return self._null_result(
                "Could not detect mouth region in enough frames (need ≥3 frames with face)."
            )

        # 2. Extract audio waveform
        waveform, sr = self._load_audio(video_path if audio_path is None else Path(audio_path), sample_rate)
        if waveform is None:
            return self._null_result("No audio stream found in the video/audio file.")

        audio_rms = float(np.sqrt(np.mean(waveform ** 2)))
        if audio_rms < 1e-6:
            return self._null_result("Audio is essentially silent; AV sync cannot be computed.")

        # 3. Compute audio envelope (MFCC-1 or RMS)
        audio_signal = self._audio_envelope(waveform, sr, n_frames=len(mouth_signal))

        # 4. Pearson correlation → sync score
        corr = self._pearson_r(mouth_signal, audio_signal)
        av_sync_score = self._corr_to_score(corr)

        is_contradiction = av_sync_score < self.contradiction_threshold
        reason = None
        if is_contradiction:
            reason = (
                f"AV sync score {av_sync_score:.3f} < threshold {self.contradiction_threshold:.2f} "
                f"→ possible audio-swap or out-of-context deepfake."
            )

        return AVSyncResult(
            av_sync_score=round(av_sync_score, 4),
            is_av_contradiction=is_contradiction,
            contradiction_threshold=self.contradiction_threshold,
            reason=reason,
            mouth_motion_energy=float(np.mean(mouth_signal)),
            audio_rms_mean=audio_rms,
            mfcc_correlation=round(corr, 4),
            n_frames_analyzed=n_frames,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_mouth_motion(
        self, video_path: Path
    ) -> Tuple[Optional[np.ndarray], int]:
        """Return per-frame mouth-region motion energy array and frame count."""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, 0

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(video_fps / self.sample_fps))

        face_cascade = self._get_face_detector()
        motion_values: List[float] = []
        prev_mouth_roi: Optional[np.ndarray] = None
        frame_idx = 0
        analyzed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % step != 0:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            if len(faces) == 0:
                # No face — use whole-frame motion as fallback
                mouth_roi = cv2.resize(gray[gray.shape[0]//2:, :], (64, 32))
            else:
                x, y, w, h = faces[0]
                # Mouth region: lower 1/3 of face bounding box
                my = y + int(h * 0.60)
                mh = int(h * 0.35)
                mouth_crop = gray[my: my + mh, x: x + w]
                if mouth_crop.size == 0:
                    mouth_crop = gray[y: y + h, x: x + w]
                mouth_roi = cv2.resize(mouth_crop, (64, 32))

            analyzed += 1
            if prev_mouth_roi is not None:
                diff = np.abs(mouth_roi.astype(np.float32) - prev_mouth_roi.astype(np.float32))
                motion_values.append(float(np.mean(diff)))
            else:
                motion_values.append(0.0)

            prev_mouth_roi = mouth_roi

        cap.release()
        if len(motion_values) < 3:
            return None, analyzed

        signal = np.array(motion_values, dtype=np.float32)
        mx = signal.max()
        if mx > 0:
            signal = signal / mx
        return signal, analyzed

    def _get_face_detector(self):
        if self._face_detector is None:
            import cv2
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_detector = cv2.CascadeClassifier(cascade_path)
        return self._face_detector

    def _load_audio(
        self, path: Path, sample_rate: int
    ) -> Tuple[Optional[np.ndarray], int]:
        """Load audio waveform from video or audio file using librosa or torchaudio."""
        try:
            import librosa
            waveform, sr = librosa.load(str(path), sr=sample_rate, mono=True)
            return waveform, sr
        except Exception:
            pass
        try:
            import torchaudio
            wav, sr = torchaudio.load(str(path))
            wav = wav.mean(0).numpy()
            if sr != sample_rate:
                import scipy.signal as ss
                wav = ss.resample_poly(wav, sample_rate, sr).astype(np.float32)
                sr = sample_rate
            return wav, sr
        except Exception:
            pass
        return None, sample_rate

    def _audio_envelope(
        self, waveform: np.ndarray, sr: int, n_frames: int
    ) -> np.ndarray:
        """Compute RMS-energy envelope resampled to n_frames time steps."""
        hop = max(1, len(waveform) // max(n_frames, 1))
        rms_values = []
        for i in range(0, len(waveform) - hop, hop):
            chunk = waveform[i: i + hop]
            rms_values.append(float(np.sqrt(np.mean(chunk ** 2))))
            if len(rms_values) >= n_frames:
                break

        # Pad/trim to exactly n_frames
        while len(rms_values) < n_frames:
            rms_values.append(rms_values[-1] if rms_values else 0.0)
        signal = np.array(rms_values[:n_frames], dtype=np.float32)
        mx = signal.max()
        if mx > 0:
            signal = signal / mx
        return signal

    @staticmethod
    def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
        """Pearson correlation, clamped to [-1, 1], returns 0 on degenerate input."""
        a = a - a.mean()
        b = b - b.mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-8:
            return 0.0
        return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))

    @staticmethod
    def _corr_to_score(corr: float) -> float:
        """Map Pearson r ∈ [-1, 1] → sync score ∈ [0, 1] via linear shift."""
        return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))

    @staticmethod
    def _null_result(reason: str) -> AVSyncResult:
        return AVSyncResult(
            av_sync_score=None,
            is_av_contradiction=False,
            contradiction_threshold=0.40,
            reason=reason,
            mouth_motion_energy=None,
            audio_rms_mean=None,
            mfcc_correlation=None,
            n_frames_analyzed=0,
        )


def compute_av_sync_score(
    video_path: str | Path,
    audio_path: Optional[str | Path] = None,
    sample_fps: float = 10.0,
    contradiction_threshold: float = 0.40,
) -> AVSyncResult:
    """Convenience wrapper to compute AV sync for a single clip.

    Used by:
      - src/pipelines/asr_batch.py (batch evaluation)
      - scripts/run_exp3_robustness.py (Experiment 3 compression/mismatch tests)
      - src/inference/pipeline.py (Layer-2 cross-modal consistency gate)
    """
    detector = AVSyncDetector(
        sample_fps=sample_fps,
        contradiction_threshold=contradiction_threshold,
    )
    return detector.compute(video_path=video_path, audio_path=audio_path)
