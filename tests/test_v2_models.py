"""Unit tests for VeriSight v2 shared models, consistency, fusion, and calibration."""

import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.v2 import (
    VisualV2Detector,
    AudioV2Detector,
    TextV2Detector,
    compute_consistency_vector,
    build_fusion_model,
    TemperatureScaler,
    mc_dropout_predict,
    compute_ece,
    abstain_verdict,
)


def test_visual_v2_forward_and_gradcam():
    model = VisualV2Detector(embedding_dim=256, use_mock_backbone=True)
    x_video = torch.randn(2, 4, 3, 64, 64)
    logits, emb = model(x_video)
    assert logits.shape == (2,)
    assert emb.shape == (2, 256)

    cam = model.compute_gradcam(x_video[:, 0])
    assert cam.shape == (2, 64, 64)
    assert float(cam.min()) >= 0.0 and float(cam.max()) <= 1.0001


def test_audio_v2_forward_and_saliency():
    model = AudioV2Detector(embedding_dim=256, use_mock_backbone=True)
    wav = torch.randn(2, 16000)
    logits, emb, sal = model(wav, return_saliency=True)
    assert logits.shape == (2,)
    assert emb.shape == (2, 256)
    assert sal.shape[0] == 2
    assert torch.allclose(sal.sum(dim=-1), torch.ones(2), atol=1e-4)


def test_text_v2_forward_and_tokens():
    model = TextV2Detector(embedding_dim=256, use_mock_backbone=True)
    ids = torch.randint(1, 500, (2, 12))
    mask = torch.ones_like(ids)
    logits, emb, weights = model(ids, mask, return_token_weights=True)
    assert logits.shape == (2,)
    assert emb.shape == (2, 256)
    assert weights.shape == (2, 12)
    top = model.explain_tokens(["the", "video", "is", "fake"], weights[0][:4], top_k=2)
    assert len(top) == 2


def test_consistency_and_all_three_fusion_strategies():
    v = torch.randn(4, 256)
    a = torch.randn(4, 256)
    t = torch.randn(4, 256)
    av_sync = torch.tensor([0.9, 0.2, 0.8, 0.1])
    c = compute_consistency_vector(v, a, t, av_sync_scores=av_sync)
    assert c.shape == (4, 3)
    assert float(c.min()) >= 0.0 and float(c.max()) <= 1.0

    mask = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=torch.float32)
    for strategy in ("concat", "cross_attention", "consistency_gated"):
        fusion = build_fusion_model(strategy=strategy, emb_dim=256)
        fusion.train()
        logits, gates = fusion(v, a, t, c, mask)
        assert logits.shape == (4,)
        assert gates.shape == (4, 3)


def test_uncertainty_calibration_and_abstention():
    scaler = TemperatureScaler(init_temp=1.0)
    logits = torch.tensor([2.5, -2.1, 0.2, -0.1, 3.0, -3.2])
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    temp = scaler.fit(logits, labels, max_iter=10)
    assert temp > 0.0

    fusion = build_fusion_model("consistency_gated", emb_dim=256)
    v = torch.randn(3, 256)
    a = torch.randn(3, 256)
    t = torch.randn(3, 256)
    c = compute_consistency_vector(v, a, t)
    m = torch.ones(3, 3)
    mc = mc_dropout_predict(lambda: fusion(v, a, t, c, m)[0], fusion, n_samples=5, temperature=temp)
    assert mc["mean_prob"].shape == (3,)
    assert mc["entropy"].shape == (3,)

    ece = compute_ece(np.array([0.9, 0.1, 0.8]), np.array([1, 0, 1]))
    assert 0.0 <= ece <= 1.0

    verdict_u, _ = abstain_verdict(0.51, entropy=0.95, entropy_threshold=0.45)
    verdict_f, _ = abstain_verdict(0.92, entropy=0.20, entropy_threshold=0.45)
    assert verdict_u == "UNCERTAIN"
    assert verdict_f == "FAKE"
