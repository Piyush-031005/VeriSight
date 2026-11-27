"""VeriSight v2 Audio Branch: Frozen wav2vec2 XLS-R + Attention Pooling + Temporal Saliency."""

from __future__ import annotations

import logging
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AudioV2Detector(nn.Module):
    """Multilingual acoustic deepfake detector with temporal segment saliency.

    Uses frozen wav2vec2-xls-r-300m (or wav2vec2-base / mock) + attentive pooling
    so it outputs both a 256-d embedding and frame-level attention weights showing
    which seconds of audio triggered a synthetic speech flag.
    """

    def __init__(
        self,
        backbone_name: str = "facebook/wav2vec2-xls-r-300m",
        freeze_backbone: bool = True,
        embedding_dim: int = 256,
        dropout: float = 0.3,
        use_mock_backbone: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.use_mock_backbone = use_mock_backbone

        if use_mock_backbone or backbone_name == "mock":
            self.backbone = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=160, stride=80),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=10, stride=5),
                nn.ReLU(),
            )
            hidden_dim = 64
        else:
            try:
                from transformers import Wav2Vec2Model

                try:
                    self.backbone = Wav2Vec2Model.from_pretrained(backbone_name, local_files_only=True)
                except Exception:
                    self.backbone = Wav2Vec2Model.from_pretrained(backbone_name)
                hidden_dim = self.backbone.config.hidden_size
                if freeze_backbone:
                    for p in self.backbone.parameters():
                        p.requires_grad = False
            except Exception as exc:
                logger.warning(f"Could not load '{backbone_name}' ({exc}); using mock backbone.")
                self.backbone = nn.Sequential(
                    nn.Conv1d(1, 32, kernel_size=160, stride=80),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=10, stride=5),
                    nn.ReLU(),
                )
                hidden_dim = 64
                self.use_mock_backbone = True

        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )

    def forward(
        self, waveform: torch.Tensor, return_saliency: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass on 16 kHz waveform (B, N_samples).

        Returns:
            logits: (B,)
            embedding: (B, embedding_dim) L2-normalized
            saliency (optional): (B, T_frames) normalized attention weights over time
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if self.use_mock_backbone or self.backbone_name == "mock":
            seq = self.backbone(waveform.unsqueeze(1)).transpose(1, 2)  # (B, T, 64)
        else:
            seq = self.backbone(waveform).last_hidden_state  # (B, T, H)

        attn_scores = self.temporal_attn(seq).squeeze(-1)  # (B, T)
        attn_weights = F.softmax(attn_scores, dim=-1)      # (B, T)
        pooled = torch.sum(seq * attn_weights.unsqueeze(-1), dim=1)  # (B, H)

        emb = F.normalize(self.adapter(pooled), p=2, dim=-1)
        logits = self.classifier(emb).squeeze(-1)

        if return_saliency:
            return logits, emb, attn_weights
        return logits, emb
