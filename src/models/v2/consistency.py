"""VeriSight v2 Layer-2 Cross-Modal Consistency Verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn.functional as F


@dataclass
class ConsistencyScores:
    """Container for Layer-2 cross-modal consistency signals in [0, 1]."""

    visual_text: float   # Image/Frame <-> Caption semantic alignment
    audio_text: float    # Whisper ASR transcript <-> Caption semantic alignment
    av_sync: float       # Voice <-> Face/Lip correspondence
    is_contradiction: bool
    reason: Optional[str] = None


def compute_cosine_similarity(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    normalize_to_unit: bool = True,
) -> torch.Tensor:
    """Compute cosine similarity between two embedding batches (B, D).

    If normalize_to_unit is True, maps [-1, 1] -> [0, 1] via (cos + 1) / 2.
    """
    a = F.normalize(emb_a, p=2, dim=-1)
    b = F.normalize(emb_b, p=2, dim=-1)
    cos = torch.sum(a * b, dim=-1)
    if normalize_to_unit:
        return ((cos + 1.0) * 0.5).clamp(0.0, 1.0)
    return cos


def compute_consistency_vector(
    visual_emb: Optional[torch.Tensor],
    audio_emb: Optional[torch.Tensor],
    text_emb: Optional[torch.Tensor],
    av_sync_scores: Optional[torch.Tensor] = None,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build (B, 3) cross-modal consistency vector [c_vt, c_at, c_av].

    Defaults missing modality pairs to neutral 0.50 so consistency-gated fusion
    never breaks when one modality is absent.
    """
    if device is None:
        for t in (visual_emb, audio_emb, text_emb, av_sync_scores):
            if t is not None:
                device = t.device
                batch_size = t.shape[0]
                break
        if device is None:
            device = torch.device("cpu")

    neutral = torch.full((batch_size,), 0.5, dtype=torch.float32, device=device)

    if visual_emb is not None and text_emb is not None:
        c_vt = compute_cosine_similarity(visual_emb, text_emb)
    else:
        c_vt = neutral.clone()

    if audio_emb is not None and text_emb is not None:
        c_at = compute_cosine_similarity(audio_emb, text_emb)
    else:
        c_at = neutral.clone()

    if av_sync_scores is not None:
        c_av = av_sync_scores.view(-1).clamp(0.0, 1.0)
    elif visual_emb is not None and audio_emb is not None:
        c_av = compute_cosine_similarity(visual_emb, audio_emb)
    else:
        c_av = neutral.clone()

    return torch.stack([c_vt, c_at, c_av], dim=-1)
