"""VeriSight v2 Visual Branch: Frozen CLIP + Spatial Artifact Backbone + Grad-CAM."""

from __future__ import annotations

import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class VisualV2Detector(nn.Module):
    """Dual-signal visual deepfake & semantic encoder with Grad-CAM support.

    Combines:
      1. Spatial artifact features (EfficientNet-B0 or lightweight mock CNN)
      2. Semantic visual features (frozen CLIP ViT-B/32 or mock projection)
    Outputs:
      logits: (B,) manipulation logit
      embedding: (B, embedding_dim) L2-normalized visual representation
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        dropout: float = 0.3,
        freeze_backbones: bool = True,
        use_clip: bool = True,
        clip_name: str = "openai/clip-vit-base-patch32",
        artifact_backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        use_mock_backbone: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_mock_backbone = use_mock_backbone
        self.use_clip = use_clip and not use_mock_backbone

        if use_mock_backbone:
            self.artifact_conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
            artifact_dim = 64
            clip_dim = 0
        else:
            import timm

            base_net = timm.create_model(artifact_backbone, pretrained=pretrained, num_classes=0)
            self.artifact_net = base_net
            artifact_dim = getattr(base_net, "num_features", 1280)
            if freeze_backbones:
                for p in self.artifact_net.parameters():
                    p.requires_grad = False

            clip_dim = 0
            if self.use_clip:
                try:
                    from transformers import CLIPVisionModel

                    self.clip_vision = CLIPVisionModel.from_pretrained(clip_name)
                    clip_dim = self.clip_vision.config.hidden_size
                    if freeze_backbones:
                        for p in self.clip_vision.parameters():
                            p.requires_grad = False
                except Exception as exc:
                    logger.warning(f"CLIP vision load failed ({exc}); using artifact backbone only.")
                    self.use_clip = False

        total_in = artifact_dim + clip_dim
        self.adapter = nn.Sequential(
            nn.Linear(total_in, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 1),
        )
        # Optional CLIP-aligned projection for Image-Text consistency
        self.clip_proj = nn.Linear(embedding_dim, 512) if clip_dim == 0 else None

    def _encode_frames(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode 4D image batch (N, 3, H, W) -> (features, spatial_feature_map)."""
        if self.use_mock_backbone:
            fmap = self.artifact_conv(x)
            pooled = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
            return pooled, fmap

        # Extract spatial feature map from timm if available for Grad-CAM
        if hasattr(self.artifact_net, "forward_features"):
            fmap = self.artifact_net.forward_features(x)
            art_feat = self.artifact_net.forward_head(fmap, pre_logits=True) if hasattr(self.artifact_net, "forward_head") else F.adaptive_avg_pool2d(fmap, 1).flatten(1)
        else:
            fmap = None
            art_feat = self.artifact_net(x)

        if self.use_clip:
            clip_out = self.clip_vision(pixel_values=x).pooler_output
            feats = torch.cat([art_feat, clip_out], dim=-1)
        else:
            feats = art_feat
        return feats, fmap

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass supporting 4D (B, 3, H, W) or 5D (B, T, 3, H, W) video tensors."""
        if x.dim() == 5:
            b, t, c, h, w = x.shape
            feats, _ = self._encode_frames(x.view(b * t, c, h, w))
            emb_flat = self.adapter(feats)
            logits_flat = self.classifier(emb_flat).squeeze(-1)
            emb = F.normalize(emb_flat.view(b, t, self.embedding_dim).mean(dim=1), p=2, dim=-1)
            logits = logits_flat.view(b, t).mean(dim=1)
            return logits, emb
        if x.dim() == 4:
            feats, _ = self._encode_frames(x)
            emb = F.normalize(self.adapter(feats), p=2, dim=-1)
            logits = self.classifier(emb).squeeze(-1)
            return logits, emb
        raise ValueError(f"Expected 4D or 5D input tensor, got {x.shape}")

    def compute_gradcam(self, x: torch.Tensor) -> torch.Tensor:
        """Compute normalized Grad-CAM spatial saliency heatmap (B, H, W) for 4D images."""
        if x.dim() == 5:
            x = x[:, 0]
        x_in = x.detach().requires_grad_(True)
        with torch.enable_grad():
            feats, fmap = self._encode_frames(x_in)
            if fmap is None:
                return torch.zeros((x.shape[0], x.shape[2], x.shape[3]), device=x.device)
            fmap.retain_grad()
            emb = self.adapter(feats)
            logit = self.classifier(emb).squeeze(-1).sum()
            self.zero_grad(set_to_none=True)
            logit.backward()

            grads = fmap.grad
            if grads is None:
                return torch.zeros((x.shape[0], x.shape[2], x.shape[3]), device=x.device)
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * fmap).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
            cam = cam.squeeze(1)
            flat_min = cam.view(cam.shape[0], -1).min(dim=1)[0].view(-1, 1, 1)
            flat_max = cam.view(cam.shape[0], -1).max(dim=1)[0].view(-1, 1, 1)
            return ((cam - flat_min) / (flat_max - flat_min + 1e-8)).detach()
