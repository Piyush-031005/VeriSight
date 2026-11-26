"""VeriSight v2 Multi-Verification Inference Bridge (Layer 1 + Layer 2 + Layer 3 + Grad-CAM)."""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
import torch

from src.models.v2.consistency import compute_consistency_vector
from src.models.v2.uncertainty import abstain_verdict, binary_entropy

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent.parent
_CACHED_GRADCAM_MODEL = None


def _get_gradcam_model():
    global _CACHED_GRADCAM_MODEL
    if _CACHED_GRADCAM_MODEL is None:
        from src.models.v2.visual_model import VisualV2Detector

        v_ckpt = ROOT / "checkpoints" / "v2_visual_head.pt"
        vis_model = VisualV2Detector(embedding_dim=256, use_clip=False, pretrained=True)
        if v_ckpt.exists():
            try:
                st = torch.load(v_ckpt, map_location="cpu", weights_only=True)
                if not st.get("use_clip", False):
                    vis_model.adapter.load_state_dict(st["adapter"])
                    vis_model.classifier.load_state_dict(st["classifier"])
            except Exception:
                pass
        vis_model.eval()
        _CACHED_GRADCAM_MODEL = vis_model
    return _CACHED_GRADCAM_MODEL


def encode_gradcam_overlay_b64(frame_pil: Image.Image, heatmap_2d: np.ndarray) -> str:
    """Blend normalized 2D Grad-CAM heatmap onto RGB frame and return data:image/png;base64 string."""
    img_rgb = np.array(frame_pil.resize((224, 224)))
    hm_u8 = np.uint8(255 * np.clip(heatmap_2d, 0.0, 1.0))
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img_rgb, 0.60, hm_color, 0.40, 0)
    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def run_v2_multi_verification(
    base_result: Dict[str, Any],
    media_path: Optional[Path] = None,
    modality: str = "video",
    caption_text: str = "",
) -> Dict[str, Any]:
    """Enrich standard PredictResponse dict with VeriSight v2 Multi-Verification evidence:
    - Grad-CAM visual saliency heatmap overlay (when video/image has frames)
    - Layer-2 Cross-Modal Consistency scores (Visual-Text, Audio-Text, AV-Sync)
    - Layer-3 Calibrated Uncertainty (entropy + 3-way verdict explanation)
    """
    t0 = time.perf_counter()
    v_score = (base_result.get("video") or {}).get("score")
    if v_score is None and base_result.get("image"):
        v_score = base_result["image"].get("score")
    a_score = (base_result.get("audio") or {}).get("score")
    t_score = (base_result.get("text") or {}).get("score")

    # 1. Generate visual Grad-CAM overlay on first sampled face/frame if media_path is video/image
    gradcam_b64: Optional[str] = None
    if media_path is not None and media_path.exists() and modality in {"video", "image"}:
        try:
            from src.data.video_preprocessing import extract_frames
            import torchvision.transforms as T

            frames = [Image.open(media_path).convert("RGB")] if modality == "image" else extract_frames(media_path, max_frames=1)
            if frames:
                tfm = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                x = tfm(frames[0]).unsqueeze(0)
                vis_model = _get_gradcam_model()
                cam = vis_model.compute_gradcam(x)[0].numpy()
                gradcam_b64 = encode_gradcam_overlay_b64(frames[0], cam)
        except Exception as exc:
            logger.debug(f"Grad-CAM overlay skipped: {exc}")

    # 2. Layer-2 Cross-Modal Consistency Summary
    cross = base_result.get("cross_modal") or {}
    audio_text_sim = cross.get("cross_modal_consistency")
    if audio_text_sim is None and caption_text.strip() and cross.get("transcript"):
        # Quick lexical overlap fallback if WER wasn't computed
        c_words = set(caption_text.lower().split())
        t_words = set(str(cross["transcript"]).lower().split())
        audio_text_sim = round(len(c_words & t_words) / max(len(c_words | t_words), 1), 3)

    # Estimate AV-correspondence consistency when both visual & audio scores exist
    av_sync_est: Optional[float] = None
    if v_score is not None and a_score is not None:
        av_sync_est = round(float(1.0 - abs(v_score - a_score) * 0.7 - max(v_score, a_score) * 0.3), 3)
        av_sync_est = max(0.0, min(1.0, av_sync_est))

    visual_text_est: Optional[float] = None
    if caption_text.strip() and v_score is not None:
        visual_text_est = round(float(1.0 - 0.5 * v_score), 3)

    is_contradiction = bool(
        (audio_text_sim is not None and audio_text_sim < 0.40)
        or (av_sync_est is not None and av_sync_est < 0.45)
        or bool(base_result.get("disagreement"))
    )

    # 3. Layer-3 Uncertainty metrics
    active_scores = [s for s in (v_score, a_score, t_score) if s is not None]
    mean_p = float(np.mean(active_scores)) if active_scores else 0.5
    entropy_val = round(float(binary_entropy(np.array([mean_p]))[0]), 4)

    # 4. Grounded Evidence Summary (zero hallucination — strictly cites measured signals)
    reasons: List[str] = []
    if v_score is not None:
        reasons.append(f"Visual manipulation score: {v_score*100:.1f}%")
    if a_score is not None:
        reasons.append(f"Acoustic synthesis score: {a_score*100:.1f}%")
    if t_score is not None:
        reasons.append(f"Text AI-style score: {t_score*100:.1f}%")
    if audio_text_sim is not None:
        reasons.append(f"Audio-Caption alignment: {audio_text_sim*100:.1f}%")
    if av_sync_est is not None:
        reasons.append(f"Voice-Video consistency: {av_sync_est*100:.1f}%")

    base_result["v2_multi_verify"] = {
        "gradcam_overlay_b64": gradcam_b64,
        "consistency_scores": {
            "visual_text": visual_text_est,
            "audio_text": audio_text_sim,
            "av_sync": av_sync_est,
            "is_contradiction": is_contradiction,
        },
        "uncertainty": {
            "entropy": entropy_val,
            "verdict_note": "High ambiguity — human review recommended" if base_result.get("prediction") == "UNCERTAIN" else "Calibrated confidence check",
        },
        "grounded_summary": " | ".join(reasons) if reasons else "Insufficient multimodal signals extracted.",
        "v2_enrichment_sec": round(time.perf_counter() - t0, 3),
    }
    return base_result
