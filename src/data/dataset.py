"""PyTorch Dataset implementations for video-only, audio-only, and multimodal deepfake detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.audio_preprocessing import extract_audio, load_waveform
from src.data.video_preprocessing import detect_and_crop_faces, extract_frames

logger = logging.getLogger(__name__)

LABEL_TO_INT = {"REAL": 0.0, "FAKE": 1.0}


class VideoDataset(Dataset):
    """Dataset for video-only deepfake detection."""

    def __init__(
        self,
        manifest_or_csv: pd.DataFrame | str | Path,
        max_frames: int = 8,
        face_detector: Optional[Any] = None,
        image_size: int = 224,
    ):
        if isinstance(manifest_or_csv, (str, Path)):
            self.df = pd.read_csv(manifest_or_csv)
        else:
            self.df = manifest_or_csv.copy()

        self.max_frames = max_frames
        self.face_detector = face_detector
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        video_path = row["video_path"]
        label_str = row.get("video_label", "REAL")
        label = torch.tensor(LABEL_TO_INT.get(label_str, 0.0), dtype=torch.float32)

        frames = extract_frames(video_path, max_frames=self.max_frames)
        face_tensors, _ = detect_and_crop_faces(
            frames, detector=self.face_detector, image_size=self.image_size
        )

        if not face_tensors:
            # Fallback: dummy zero tensor if no face found in sample
            faces_stack = torch.zeros((self.max_frames, 3, self.image_size, self.image_size), dtype=torch.float32)
        else:
            if len(face_tensors) < self.max_frames:
                pad_count = self.max_frames - len(face_tensors)
                face_tensors = face_tensors + [face_tensors[-1]] * pad_count
            elif len(face_tensors) > self.max_frames:
                face_tensors = face_tensors[: self.max_frames]
            faces_stack = torch.stack(face_tensors, dim=0)

        return faces_stack, label


class AudioDataset(Dataset):
    """Dataset for audio-only deepfake detection."""

    def __init__(
        self,
        manifest_or_csv: pd.DataFrame | str | Path,
        sample_rate: int = 16000,
        max_duration_seconds: float = 4.0,
    ):
        if isinstance(manifest_or_csv, (str, Path)):
            self.df = pd.read_csv(manifest_or_csv)
        else:
            self.df = manifest_or_csv.copy()

        self.sample_rate = sample_rate
        self.max_duration_seconds = max_duration_seconds
        self.target_length = int(sample_rate * max_duration_seconds)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        path_str = row["video_path"]
        label_str = row.get("audio_label", "REAL")
        label = torch.tensor(LABEL_TO_INT.get(label_str, 0.0), dtype=torch.float32)

        p = Path(path_str)
        # If it's a WAV or audio file directly
        if p.suffix.lower() == ".wav":
            wav_path = p
        else:
            try:
                wav_path = extract_audio(p, sample_rate=self.sample_rate)
            except Exception as e:
                logger.debug(f"Audio extraction failed for {p}: {e}")
                wav_path = None

        if wav_path and wav_path.exists():
            waveform = load_waveform(
                wav_path,
                target_sample_rate=self.sample_rate,
                max_duration_seconds=self.max_duration_seconds,
            )
        else:
            waveform = None

        if waveform is None:
            waveform = torch.zeros(self.target_length, dtype=torch.float32)

        return waveform, label


class MultimodalDataset(Dataset):
    """Dataset yielding synchronized video face crops and audio waveforms with modality labels."""

    def __init__(
        self,
        manifest_or_csv: pd.DataFrame | str | Path,
        max_frames: int = 8,
        face_detector: Optional[Any] = None,
        image_size: int = 224,
        sample_rate: int = 16000,
        max_duration_seconds: float = 4.0,
    ):
        if isinstance(manifest_or_csv, (str, Path)):
            self.df = pd.read_csv(manifest_or_csv)
        else:
            self.df = manifest_or_csv.copy()

        self.video_ds = VideoDataset(
            self.df, max_frames=max_frames, face_detector=face_detector, image_size=image_size
        )
        self.audio_ds = AudioDataset(
            self.df, sample_rate=sample_rate, max_duration_seconds=max_duration_seconds
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        faces_stack, v_label = self.video_ds[idx]
        waveform, a_label = self.audio_ds[idx]

        overall_label_str = row.get("overall_label", "REAL")
        overall_label = torch.tensor(LABEL_TO_INT.get(overall_label_str, 0.0), dtype=torch.float32)

        return {
            "faces": faces_stack,
            "waveform": waveform,
            "overall_label": overall_label,
            "video_label": v_label,
            "audio_label": a_label,
            "identity_id": row.get("identity_id", "unknown"),
            "category": row.get("category", "unknown"),
        }
