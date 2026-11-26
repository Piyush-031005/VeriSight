"""VeriSight v2 Multilingual Text Branch: Frozen XLM-RoBERTa / MuRIL + Token Attribution."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class TextV2Detector(nn.Module):
    """Multilingual (English + Hindi + Hinglish) text & caption verification encoder.

    Uses frozen XLM-RoBERTa (or MuRIL / mock embedding encoder for unit tests)
    with a trainable 256-d adapter and token-level attention/gradient attribution.
    """

    def __init__(
        self,
        backbone_name: str = "xlm-roberta-base",
        freeze_backbone: bool = True,
        embedding_dim: int = 256,
        dropout: float = 0.3,
        use_mock_backbone: bool = False,
        mock_vocab_size: int = 1000,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.use_mock_backbone = use_mock_backbone

        if use_mock_backbone or backbone_name == "mock":
            self.mock_embed = nn.Embedding(mock_vocab_size, 64)
            hidden_dim = 64
        else:
            try:
                from transformers import AutoModel

                try:
                    self.backbone = AutoModel.from_pretrained(backbone_name, local_files_only=True)
                except Exception:
                    self.backbone = AutoModel.from_pretrained(backbone_name)
                hidden_dim = self.backbone.config.hidden_size
                if freeze_backbone:
                    for p in self.backbone.parameters():
                        p.requires_grad = False
            except Exception as exc:
                logger.warning(f"Could not load '{backbone_name}' ({exc}); using mock backbone.")
                self.mock_embed = nn.Embedding(mock_vocab_size, 64)
                hidden_dim = 64
                self.use_mock_backbone = True

        self.token_attn = nn.Sequential(
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
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_token_weights: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass over tokenized input_ids (B, L).

        Returns:
            logits: (B,)
            embedding: (B, embedding_dim) L2-normalized
            token_weights (optional): (B, L) normalized attribution weights
        """
        if attention_mask is None:
            attention_mask = (input_ids != 0).long()

        if self.use_mock_backbone or self.backbone_name == "mock":
            seq = self.mock_embed(input_ids.clamp(min=0, max=self.mock_embed.num_embeddings - 1))
        else:
            seq = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        raw_scores = self.token_attn(seq).squeeze(-1)
        raw_scores = raw_scores.masked_fill(attention_mask == 0, -1e4)
        token_weights = F.softmax(raw_scores, dim=-1)
        pooled = torch.sum(seq * token_weights.unsqueeze(-1), dim=1)

        emb = F.normalize(self.adapter(pooled), p=2, dim=-1)
        logits = self.classifier(emb).squeeze(-1)

        if return_token_weights:
            return logits, emb, token_weights
        return logits, emb

    @torch.no_grad()
    def explain_tokens(
        self, tokens: List[str], token_weights: torch.Tensor, top_k: int = 8
    ) -> List[Dict[str, float | str]]:
        """Extract top contributing tokens from normalized attention weights."""
        weights = token_weights.detach().cpu().tolist()
        pairs = [
            {"token": tok, "weight": round(float(w), 4)}
            for tok, w in zip(tokens, weights)
            if tok not in {"<s>", "</s>", "<pad>", "[CLS]", "[SEP]", "[PAD]"}
        ]
        pairs.sort(key=lambda item: item["weight"], reverse=True)
        return pairs[:top_k]
