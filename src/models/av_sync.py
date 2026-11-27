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

_PURE_AUDIO_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


@dataclass
class AVSyncResult:
    """Output of the AV correspondence module."""

    av_sync_score: Optional[float]
    """Score in [0, 1]: higher = better sync. None if computation not possible."""

    is_av_contradiction: bool
    """True when av_sync_score < threshold -> signals possible audio-swap deepfake."""

    contradiction_threshold: float
    """Threshold used to decide contradiction (default 0.40)."""

    reason: Optional[str]
    """Human-readable explanation if score is None or contradiction detected."""

    mouth_motion_energy: Optional[float]
    """Mean mouth-region motion energy across sampled frames (normalized)."""

    audio_rms_mean: Optional[float]
    """Mean RMS energy of the audio waveform."""

    mfcc_correlation: Optional[float]
    """Pearson correlation between audio envelope and mouth-motion envelope."""

    n_frames_analyzed: int = 0
    """Number of video frames that were successfully analyzed."""


class AVSyncDetector:
    """Lip-Sync / Audio-Visual Correspondence checker for VeriSight v2 (Layer 2).

    Algorithm:
      1. Sample N frames from the video clip (default every ~100ms).
      2. Detect face bounding box -> crop mouth region (returns None if <3 face frames).
      3. Compute frame-to-frame mouth region motion energy as a 1D signal.
      4. Extract 16 kHz audio via soundfile / PyAV -> compute RMS + spectral envelope.
      5. Align both 1D signals to the same temporal length and compute Pearson r.
      6. Map r -> av_sync_score in [0, 1].

    Falls back gracefully returning av_sync_score=None + reason when no face or silent/missing audio.
    """

    def __init__(
        self,
        sample_fps: float = 10.0,
        contradiction_threshold: float = 0.40,
        face_detection_method: str = "haarcascade",
    ):
        self.sample_fps = sample_fps
        self.contradiction_threshold = contradiction_threshold
        self.face_detection_method = face_detection_method
        self._face_detector = None

    def compute(
        self,
        video_path: str | Path,
        audio_path: Optional[str | Path] = None,
        sample_rate: int = 16000,
    ) -> AVSyncResult:
        """Compute AV sync score for a video (audio extracted from video if audio_path is None)."""
        video_path = Path(video_path)
        if not video_path.exists():
            return self._null_result(f"Video not found: {video_path}")

        try:
            import cv2  # noqa: F401
        except ImportError:
            return self._null_result("opencv-python-headless not installed; cannot compute AV sync.")

        mouth_signal, valid_indices, total_steps, n_frames = self._extract_mouth_motion(video_path)
        if mouth_signal is None or len(mouth_signal) < 3:
            return self._null_result(
                "Could not detect mouth region in enough frames (need >=3 frames with face)."
            )

        audio_src = video_path if audio_path is None else Path(audio_path)
        if not audio_src.exists():
            return self._null_result(f"Audio source not found: {audio_src}")

        waveform, sr = self._load_audio(audio_src, sample_rate)
        if waveform is None or len(waveform) == 0:
            return self._null_result("No audio stream found in the video/audio file.")

        audio_rms = float(np.sqrt(np.mean(waveform ** 2)))
        if audio_rms < 1e-6:
            return self._null_result("Audio is essentially silent; AV sync cannot be computed.")

        # Align mouth motion steps against exact corresponding timestamps in full audio envelope
        audio_full = self._audio_envelope(waveform, sr, n_frames=max(total_steps, len(mouth_signal)))
        valid_idx = np.clip(np.asarray(valid_indices, dtype=int), 0, len(audio_full) - 1)
        audio_signal = audio_full[valid_idx]

        # Allow ±1 frame mux latency tolerance (~125ms) typical of social-media MP4 containers
        corrs = [self._pearson_r(mouth_signal, audio_signal)]
        if len(mouth_signal) >= 6:
            corrs.append(self._pearson_r(mouth_signal[1:], audio_signal[:-1]))
            corrs.append(self._pearson_r(mouth_signal[:-1], audio_signal[1:]))
        corr = max(corrs)
        av_sync_score = self._corr_to_score(corr)

        is_contradiction = av_sync_score < self.contradiction_threshold
        reason = None
        if is_contradiction:
            reason = (
                f"AV sync score {av_sync_score:.3f} < threshold {self.contradiction_threshold:.2f} "
                f"-> possible audio-swap or out-of-context deepfake."
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

    def _extract_mouth_motion(
        self, video_path: Path
    ) -> Tuple[Optional[np.ndarray], List[int], int, int]:
        """Return (mouth motion array, valid sampled step indices, total sampled steps, detected face count)."""
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, [], 0, 0

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(video_fps / max(self.sample_fps, 1.0))))
        max_sampled_frames = 150

        face_cascade = self._get_face_detector()
        motion_values: List[float] = []
        valid_indices: List[int] = []
        prev_mouth_roi: Optional[np.ndarray] = None
        smooth_box: Optional[Tuple[float, float, float, float]] = None
        missed_streak = 0
        faces_detected = 0
        frame_idx = 0
        sampled = 0

        while sampled < max_sampled_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % step != 0:
                continue
            step_pos = sampled
            sampled += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Downscale large HD frames before HaarCascade so 1080p clips run >10x faster.
            img_h, img_w = gray.shape[:2]
            if img_w > 480:
                scale = 480.0 / img_w
                gray = cv2.resize(gray, (480, max(1, int(round(img_h * scale)))))
                img_h, img_w = gray.shape[:2]

            min_face_dim = max(36, min(img_w, img_h) // 10)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face_dim, min_face_dim)
            )

            if len(faces) > 0:
                # Pick largest detected face and apply EMA smoothing to reduce Haar box jitter
                nx, ny, nw, nh = max(faces, key=lambda b: int(b[2]) * int(b[3]))
                if smooth_box is not None:
                    sx, sy, sw, sh = smooth_box
                    dist = abs(nx - sx) + abs(ny - sy)
                    if dist < 0.6 * max(sw, nw):
                        alpha = 0.35
                        smooth_box = (
                            (1 - alpha) * sx + alpha * nx,
                            (1 - alpha) * sy + alpha * ny,
                            (1 - alpha) * sw + alpha * nw,
                            (1 - alpha) * sh + alpha * nh,
                        )
                    else:
                        smooth_box = (float(nx), float(ny), float(nw), float(nh))
                        prev_mouth_roi = None
                else:
                    smooth_box = (float(nx), float(ny), float(nw), float(nh))
                missed_streak = 0
                faces_detected += 1
            elif smooth_box is not None and missed_streak < 3:
                missed_streak += 1
            else:
                prev_mouth_roi = None
                continue

            x, y, w, h = [int(round(v)) for v in smooth_box]
            my1 = max(0, min(img_h - 1, y + int(h * 0.60)))
            my2 = max(my1 + 1, min(img_h, y + int(h * 0.95)))
            mx1 = max(0, min(img_w - 1, x))
            mx2 = max(mx1 + 1, min(img_w, x + w))
            mouth_crop = gray[my1:my2, mx1:mx2]
            if mouth_crop.size == 0:
                continue
            mouth_roi = cv2.resize(mouth_crop, (64, 32))

            if prev_mouth_roi is not None:
                diff = np.abs(mouth_roi.astype(np.float32) - prev_mouth_roi.astype(np.float32))
                motion_values.append(float(np.mean(diff)))
                valid_indices.append(step_pos)
            else:
                motion_values.append(0.0)
                valid_indices.append(step_pos)
            prev_mouth_roi = mouth_roi

        cap.release()
        if faces_detected < 3 or len(motion_values) < 3:
            if sampled >= 3:
                mtcnn_res = self._extract_mouth_motion_mtcnn(video_path)
                if mtcnn_res[0] is not None:
                    return mtcnn_res
            return None, [], sampled, faces_detected

        signal = np.array(motion_values, dtype=np.float32)
        mx = float(signal.max())
        if mx > 0:
            signal = signal / mx
        return signal, valid_indices, sampled, faces_detected

    def _extract_mouth_motion_mtcnn(
        self, video_path: Path
    ) -> Tuple[Optional[np.ndarray], List[int], int, int]:
        """Keyframe MTCNN fallback when Haar fails on glasses, warm light, or UI forehead overlays."""
        import cv2
        from src.inference.pretrained_models import FaceCropHelper

        cropper = FaceCropHelper(target_size=224, margin_ratio=0.10)
        if cropper._get_mtcnn() is None:
            return None, [], 0, 0

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, [], 0, 0

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(video_fps / max(self.sample_fps, 1.0))))
        max_sampled_frames = 150

        sampled_grays: List[np.ndarray] = []
        sampled_rgbs: List[np.ndarray] = []
        frame_idx = 0
        while len(sampled_grays) < max_sampled_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % step != 0:
                continue
            fh, fw = frame.shape[:2]
            if fw > 480:
                scale = 480.0 / fw
                frame = cv2.resize(frame, (480, max(1, int(round(fh * scale)))))
            sampled_rgbs.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            sampled_grays.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        cap.release()

        sampled = len(sampled_grays)
        if sampled < 3:
            return None, [], sampled, 0

        key_count = min(14, sampled)
        key_indices = sorted(set(np.linspace(0, sampled - 1, key_count, dtype=int).tolist()))
        detected_boxes: dict[int, Tuple[float, float, float, float]] = {}
        for ki in key_indices:
            bbox = cropper.detect_face_bbox(sampled_rgbs[ki])
            if bbox is not None:
                detected_boxes[ki] = tuple(float(v) for v in bbox)

        if len(detected_boxes) < 3:
            return None, [], sampled, len(detected_boxes)

        known_k = sorted(detected_boxes.keys())
        first_k, last_k = known_k[0], known_k[-1]
        img_h, img_w = sampled_grays[0].shape[:2]
        motion_values: List[float] = []
        valid_indices: List[int] = []
        prev_mouth_roi: Optional[np.ndarray] = None

        for s_idx in range(first_k, last_k + 1):
            if s_idx in detected_boxes:
                bx, by, bw, bh = detected_boxes[s_idx]
            else:
                right_pos = int(np.searchsorted(known_k, s_idx))
                k0 = known_k[right_pos - 1]
                k1 = known_k[right_pos]
                t = (s_idx - k0) / max(k1 - k0, 1)
                b0, b1 = detected_boxes[k0], detected_boxes[k1]
                bx = (1 - t) * b0[0] + t * b1[0]
                by = (1 - t) * b0[1] + t * b1[1]
                bw = (1 - t) * b0[2] + t * b1[2]
                bh = (1 - t) * b0[3] + t * b1[3]

            x, y, w, h = int(round(bx)), int(round(by)), int(round(bw)), int(round(bh))
            my1 = max(0, min(img_h - 1, y + int(h * 0.60)))
            my2 = max(my1 + 1, min(img_h, y + int(h * 0.95)))
            mx1 = max(0, min(img_w - 1, x))
            mx2 = max(mx1 + 1, min(img_w, x + w))
            mouth_crop = sampled_grays[s_idx][my1:my2, mx1:mx2]
            if mouth_crop.size == 0:
                continue
            mouth_roi = cv2.resize(mouth_crop, (64, 32))
            if prev_mouth_roi is not None:
                diff = np.abs(mouth_roi.astype(np.float32) - prev_mouth_roi.astype(np.float32))
                motion_values.append(float(np.mean(diff)))
                valid_indices.append(s_idx)
            else:
                motion_values.append(0.0)
                valid_indices.append(s_idx)
            prev_mouth_roi = mouth_roi

        if len(motion_values) < 3:
            return None, [], sampled, len(detected_boxes)

        signal = np.array(motion_values, dtype=np.float32)
        mx = float(signal.max())
        if mx > 0:
            signal = signal / mx
        return signal, valid_indices, sampled, len(detected_boxes)

    def _get_face_detector(self):
        if self._face_detector is None:
            import cv2
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_detector = cv2.CascadeClassifier(cascade_path)
        return self._face_detector

    def _load_audio(
        self, path: Path, sample_rate: int
    ) -> Tuple[Optional[np.ndarray], int]:
        """Load 16 kHz mono float32 waveform without hanging on missing external ffmpeg.exe."""
        ext = path.suffix.lower()

        if ext in _PURE_AUDIO_EXTS:
            try:
                import soundfile as sf
                wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                if sr != sample_rate:
                    import scipy.signal as ss
                    wav = ss.resample_poly(wav, sample_rate, sr).astype(np.float32)
                return wav.astype(np.float32, copy=False), sample_rate
            except Exception:
                pass

        # PyAV bundles FFmpeg C libraries directly and decodes video/audio containers fast.
        try:
            import av
            with av.open(str(path)) as container:
                if not container.streams.audio:
                    return None, sample_rate
                resampler = av.audio.resampler.AudioResampler(
                    format="flt", layout="mono", rate=sample_rate
                )
                chunks: List[np.ndarray] = []
                for frame in container.decode(audio=0):
                    for rframe in resampler.resample(frame):
                        chunks.append(rframe.to_ndarray().reshape(-1))
                for rframe in resampler.resample(None):
                    chunks.append(rframe.to_ndarray().reshape(-1))
            if chunks:
                return np.concatenate(chunks).astype(np.float32, copy=False), sample_rate
        except Exception as exc:
            logger.debug("PyAV audio load failed for %s: %s", path, exc)

        try:
            import torchaudio
            wav_t, sr = torchaudio.load(str(path))
            wav = wav_t.mean(0).numpy().astype(np.float32)
            if sr != sample_rate:
                import scipy.signal as ss
                wav = ss.resample_poly(wav, sample_rate, sr).astype(np.float32)
            return wav, sample_rate
        except Exception as exc:
            logger.debug("torchaudio load failed for %s: %s", path, exc)

        return None, sample_rate

    def _audio_envelope(
        self, waveform: np.ndarray, sr: int, n_frames: int
    ) -> np.ndarray:
        """Compute blended RMS + MFCC-1 energy envelope resampled to n_frames steps."""
        hop = max(1, len(waveform) // max(n_frames, 1))
        rms_values: List[float] = []
        for i in range(0, len(waveform) - hop + 1, hop):
            chunk = waveform[i: i + hop]
            rms_values.append(float(np.sqrt(np.mean(chunk ** 2))))
            if len(rms_values) >= n_frames:
                break

        while len(rms_values) < n_frames:
            rms_values.append(rms_values[-1] if rms_values else 0.0)
        rms_sig = np.array(rms_values[:n_frames], dtype=np.float32)
        if rms_sig.max() > 0:
            rms_sig = rms_sig / rms_sig.max()

        # Blend with vocal-band (300-3400 Hz) log-energy proxy via FFT when sample length permits.
        try:
            band_values: List[float] = []
            freqs = None
            band_mask = None
            for i in range(0, len(waveform) - hop + 1, hop):
                chunk = waveform[i: i + hop]
                if freqs is None or len(chunk) != prev_len:
                    prev_len = len(chunk)
                    freqs = np.fft.rfftfreq(prev_len, d=1.0 / sr)
                    band_mask = (freqs >= 300.0) & (freqs <= 3400.0)
                spec = np.abs(np.fft.rfft(chunk))
                band_e = float(np.sqrt(np.mean(spec[band_mask] ** 2))) if np.any(band_mask) else 0.0
                band_values.append(band_e)
                if len(band_values) >= n_frames:
                    break
            while len(band_values) < n_frames:
                band_values.append(band_values[-1] if band_values else 0.0)
            band_sig = np.array(band_values[:n_frames], dtype=np.float32)
            if band_sig.max() > 0:
                band_sig = band_sig / band_sig.max()
            return 0.5 * rms_sig + 0.5 * band_sig
        except Exception:
            return rms_sig

    @staticmethod
    def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
        """Pearson correlation, clamped to [-1, 1], returns 0 on degenerate input."""
        a = a - a.mean()
        b = b - b.mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-8:
            return 0.0
        return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))

    @staticmethod
    def _corr_to_score(corr: float) -> float:
        """Map Pearson r in [-1, 1] -> sync score in [0, 1] via linear shift."""
        return float(np.clip((corr + 1.0) / 2.0, 0.0, 1.0))

    def _null_result(self, reason: str) -> AVSyncResult:
        return AVSyncResult(
            av_sync_score=None,
            is_av_contradiction=False,
            contradiction_threshold=self.contradiction_threshold,
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
