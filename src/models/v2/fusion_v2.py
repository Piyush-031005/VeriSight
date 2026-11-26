"""VeriSight v2 Fusion Architectures (Exp 2): Concat MLP, Cross-Attention, and Consistency-Gated Fusion."""

from __future__ import annotations

from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusionMLP(nn.Module):
    """Approach A (Baseline): Direct concatenation of modality embeddings + consistency scores."""

    def __init__(self, emb_dim: int = 256, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        in_dim = emb_dim * 3 + 3 + 3  # 3 embeddings + 3 consistency scores + 3 availability masks
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        v_emb: torch.Tensor,
        a_emb: torch.Tensor,
        t_emb: torch.Tensor,
        consistency: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([v_emb, a_emb, t_emb, consistency, modality_mask], dim=-1)
        logits = self.net(x).squeeze(-1)
        weights = modality_mask / (modality_mask.sum(dim=-1, keepdim=True) + 1e-8)
        return logits, weights


class CrossAttentionFusion(nn.Module):
    """Approach B: Multi-head self/cross-attention across [Visual, Audio, Text] modality tokens."""

    def __init__(self, emb_dim: int = 256, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim + 3, emb_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, 1),
        )

    def forward(
        self,
        v_emb: torch.Tensor,
        a_emb: torch.Tensor,
        t_emb: torch.Tensor,
        consistency: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.stack([v_emb, a_emb, t_emb], dim=1)  # (B, 3, D)
        key_padding_mask = modality_mask == 0               # True = ignored
        attn_out, attn_weights = self.attn(tokens, tokens, tokens, key_padding_mask=key_padding_mask)
        tokens = self.norm(tokens + attn_out)
        # Masked mean pooling
        mask_f = modality_mask.unsqueeze(-1)
        pooled = (tokens * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + 1e-8)
        logits = self.head(torch.cat([pooled, consistency], dim=-1)).squeeze(-1)
        mod_weights = attn_weights.mean(dim=1)
        return logits, mod_weights


class ConsistencyGatedFusion(nn.Module):
    """Approach C (Proposed Novelty): Consistency-Gated Multimodal Fusion with Modality-Dropout.

    Uses Layer-2 cross-modal consistency scores [c_vt, c_at, c_av] plus modality availability
    to dynamically gate and reweight Visual, Audio, and Text representations.
    """

    def __init__(
        self,
        emb_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        modality_dropout_prob: float = 0.2,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.modality_dropout_prob = modality_dropout_prob

        # Gate network takes [v_emb, a_emb, t_emb, consistency (3), mask (3)] -> 3 modality gates
        self.gate_net = nn.Sequential(
            nn.Linear(emb_dim * 3 + 6, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 3),
        )

        # Consistency projection adds explicit contradiction signal into final classifier
        self.consistency_proj = nn.Sequential(
            nn.Linear(3, 32),
            nn.GELU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(emb_dim + 32, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def apply_modality_dropout(
        self,
        v_emb: torch.Tensor,
        a_emb: torch.Tensor,
        t_emb: torch.Tensor,
        modality_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly drop individual modalities during training while keeping >=1 active."""
        if not self.training or self.modality_dropout_prob <= 0.0:
            return v_emb, a_emb, t_emb, modality_mask

        b = modality_mask.shape[0]
        keep = (torch.rand((b, 3), device=modality_mask.device) > self.modality_dropout_prob).float()
        new_mask = modality_mask * keep
        # Guarantee at least one modality remains active per row
        empty_rows = new_mask.sum(dim=1) == 0
        if empty_rows.any():
            new_mask[empty_rows] = modality_mask[empty_rows]

        v_out = v_emb * new_mask[:, 0:1]
        a_out = a_emb * new_mask[:, 1:2]
        t_out = t_emb * new_mask[:, 2:3]
        return v_out, a_out, t_out, new_mask

    def forward(
        self,
        v_emb: torch.Tensor,
        a_emb: torch.Tensor,
        t_emb: torch.Tensor,
        consistency: torch.Tensor,
        modality_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if modality_mask is None:
            modality_mask = torch.ones((v_emb.shape[0], 3), dtype=torch.float32, device=v_emb.device)

        v_emb, a_emb, t_emb, modality_mask = self.apply_modality_dropout(
            v_emb, a_emb, t_emb, modality_mask
        )

        gate_in = torch.cat([v_emb, a_emb, t_emb, consistency, modality_mask], dim=-1)
        gate_logits = self.gate_net(gate_in)
        gate_logits = gate_logits.masked_fill(modality_mask == 0, -1e4)
        gates = F.softmax(gate_logits, dim=-1)  # (B, 3)

        fused_emb = (
            gates[:, 0:1] * v_emb
            + gates[:, 1:2] * a_emb
            + gates[:, 2:3] * t_emb
        )
        c_feat = self.consistency_proj(consistency)
        logits = self.classifier(torch.cat([fused_emb, c_feat], dim=-1)).squeeze(-1)
        return logits, gates


def build_fusion_model(
    strategy: Literal["concat", "cross_attention", "consistency_gated"] = "consistency_gated",
    emb_dim: int = 256,
    hidden_dim: int = 256,
    num_heads: int = 4,
    dropout: float = 0.3,
    modality_dropout_prob: float = 0.2,
) -> nn.Module:
    """Factory returning one of the 3 Experiment-2 fusion architectures."""
    if strategy == "concat":
        return ConcatFusionMLP(emb_dim=emb_dim, hidden_dim=hidden_dim, dropout=dropout)
    if strategy == "cross_attention":
        return CrossAttentionFusion(emb_dim=emb_dim, num_heads=num_heads, dropout=dropout)
    if strategy == "consistency_gated":
        return ConsistencyGatedFusion(
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            modality_dropout_prob=modality_dropout_prob,
        )
    raise ValueError(f"Unknown fusion strategy: {strategy}")
