"""VeriSight v2 shared multimodal modules."""

from src.models.v2.visual_model import VisualV2Detector
from src.models.v2.audio_model import AudioV2Detector
from src.models.v2.text_model import TextV2Detector
from src.models.v2.consistency import ConsistencyScores, compute_cosine_similarity, compute_consistency_vector
from src.models.v2.fusion_v2 import (
    ConcatFusionMLP,
    CrossAttentionFusion,
    ConsistencyGatedFusion,
    build_fusion_model,
)
from src.models.v2.uncertainty import (
    TemperatureScaler,
    mc_dropout_predict,
    compute_ece,
    abstain_verdict,
)

__all__ = [
    "VisualV2Detector",
    "AudioV2Detector",
    "TextV2Detector",
    "ConsistencyScores",
    "compute_cosine_similarity",
    "compute_consistency_vector",
    "ConcatFusionMLP",
    "CrossAttentionFusion",
    "ConsistencyGatedFusion",
    "build_fusion_model",
    "TemperatureScaler",
    "mc_dropout_predict",
    "compute_ece",
    "abstain_verdict",
]
