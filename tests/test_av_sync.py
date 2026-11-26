"""VeriSight v2 — Unit Tests for AV Sync Module (Laptop 2).

Run with: pytest tests/test_av_sync.py -v
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest


class TestAVSyncDetectorInternals:
    """Unit tests for static helper methods."""

    def test_pearson_r_identical(self):
        from src.models.av_sync import AVSyncDetector
        a = np.array([0.1, 0.5, 0.3, 0.9, 0.2], dtype=np.float32)
        r = AVSyncDetector._pearson_r(a, a.copy())
        assert abs(r - 1.0) < 1e-5

    def test_pearson_r_opposite(self):
        from src.models.av_sync import AVSyncDetector
        a = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        b = np.array([0.9, 0.5, 0.1], dtype=np.float32)
        r = AVSyncDetector._pearson_r(a, b)
        assert r < -0.9

    def test_pearson_r_constant_array(self):
        from src.models.av_sync import AVSyncDetector
        a = np.ones(10, dtype=np.float32)
        b = np.array([0.1, 0.9, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 1.0], dtype=np.float32)
        r = AVSyncDetector._pearson_r(a, b)
        assert r == 0.0

    def test_corr_to_score_range(self):
        from src.models.av_sync import AVSyncDetector
        assert AVSyncDetector._corr_to_score(1.0) == 1.0
        assert AVSyncDetector._corr_to_score(-1.0) == 0.0
        assert abs(AVSyncDetector._corr_to_score(0.0) - 0.5) < 1e-5
        assert AVSyncDetector._corr_to_score(2.0) == 1.0
        assert AVSyncDetector._corr_to_score(-2.0) == 0.0

    def test_null_result_structure(self):
        from src.models.av_sync import AVSyncDetector, AVSyncResult
        detector = AVSyncDetector()
        result = detector._null_result("test reason")
        assert isinstance(result, AVSyncResult)
        assert result.av_sync_score is None
        assert result.is_av_contradiction is False
        assert result.reason == "test reason"
        assert result.n_frames_analyzed == 0


class TestAVSyncResult:
    """Tests for the AVSyncResult dataclass."""

    def test_dataclass_fields(self):
        from src.models.av_sync import AVSyncResult
        r = AVSyncResult(
            av_sync_score=0.75,
            is_av_contradiction=False,
            contradiction_threshold=0.40,
            reason=None,
            mouth_motion_energy=0.3,
            audio_rms_mean=0.1,
            mfcc_correlation=0.6,
            n_frames_analyzed=15,
        )
        assert r.av_sync_score == 0.75
        assert not r.is_av_contradiction
        assert r.n_frames_analyzed == 15

    def test_contradiction_flag(self):
        from src.models.av_sync import AVSyncResult
        r = AVSyncResult(
            av_sync_score=0.25,
            is_av_contradiction=True,
            contradiction_threshold=0.40,
            reason="Low AV sync",
            mouth_motion_energy=0.1,
            audio_rms_mean=0.05,
            mfcc_correlation=-0.3,
            n_frames_analyzed=10,
        )
        assert r.is_av_contradiction is True
        assert r.av_sync_score < r.contradiction_threshold


class TestComputeAVSyncScore:
    """Integration tests using demo_samples."""

    def test_missing_video_returns_null(self):
        from src.models.av_sync import compute_av_sync_score
        result = compute_av_sync_score("nonexistent_file_xyz.mp4")
        assert result.av_sync_score is None
        assert result.reason is not None

    @pytest.mark.skipif(
        not Path("demo_samples/test_pattern_with_audio.mp4").exists(),
        reason="Demo video not available",
    )
    def test_no_face_video_returns_null_with_reason(self):
        from src.models.av_sync import compute_av_sync_score
        result = compute_av_sync_score("demo_samples/test_pattern_with_audio.mp4")
        assert result.av_sync_score is None
        assert result.reason is not None

    @pytest.mark.skipif(
        not Path("demo_samples/test_video_realface_with_audio.mp4").exists(),
        reason="Realface demo video not available",
    )
    def test_realface_video_returns_score(self):
        from src.models.av_sync import compute_av_sync_score
        result = compute_av_sync_score("demo_samples/test_video_realface_with_audio.mp4")
        assert result.av_sync_score is None or (0.0 <= result.av_sync_score <= 1.0)

    @pytest.mark.skipif(
        not Path("demo_samples/test_video_faceswap_with_audio.mp4").exists(),
        reason="Face-swap demo video not available",
    )
    def test_faceswap_video(self):
        from src.models.av_sync import compute_av_sync_score
        result = compute_av_sync_score("demo_samples/test_video_faceswap_with_audio.mp4")
        assert result.av_sync_score is None or (0.0 <= result.av_sync_score <= 1.0)


class TestAudioTextConsistency:
    """Tests for the audio-text consistency helper in asr_batch.py."""

    def test_identical_text_high_score(self):
        from src.pipelines.asr_batch import compute_audio_text_consistency
        score = compute_audio_text_consistency(
            "The prime minister addressed the nation today",
            "The prime minister addressed the nation today",
            use_embeddings=False,
        )
        assert score >= 0.8

    def test_empty_inputs_neutral(self):
        from src.pipelines.asr_batch import compute_audio_text_consistency
        score = compute_audio_text_consistency("", "", use_embeddings=False)
        assert 0.0 <= score <= 1.0

    def test_unrelated_texts_low_score(self):
        from src.pipelines.asr_batch import compute_audio_text_consistency
        score = compute_audio_text_consistency("cat dog fish", "quantum physics relativity", use_embeddings=False)
        assert score == 0.0


class TestASRBatchModule:
    """Tests for asr_batch helper functions."""

    def test_import_success(self):
        from src.pipelines import asr_batch  # noqa: F401

    def test_all_exts_defined(self):
        from src.pipelines.asr_batch import ALL_EXTS, AUDIO_EXTS, VIDEO_EXTS
        assert ".wav" in AUDIO_EXTS
        assert ".mp4" in VIDEO_EXTS
        assert ".flac" in AUDIO_EXTS
        assert len(ALL_EXTS) == len(AUDIO_EXTS) + len(VIDEO_EXTS)
