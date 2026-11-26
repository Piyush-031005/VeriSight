"""CPU visual detectors for face manipulation and whole-image synthesis."""
from __future__ import annotations

import logging
import json
import math
from .runtime import positive_index
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Face Detector and Margin Cropper
# ---------------------------------------------------------------------------

class FaceCropHelper:
    """Detects frontal human faces and extracts crops with a specified margin."""

    def __init__(self, target_size: int = 224, margin_ratio: float = 0.18):
        self.target_size = target_size
        self.margin_ratio = margin_ratio
        self._cascade = None
        self._mtcnn = None

    def _get_cascade(self):
        if self._cascade is None:
            if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
                return None
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(cascade_path)
        return self._cascade

    def _get_mtcnn(self):
        if self._mtcnn is None:
            try:
                from facenet_pytorch import MTCNN
                self._mtcnn = MTCNN(
                    image_size=self.target_size,
                    margin=0,
                    keep_all=True,
                    select_largest=True,
                    device="cpu",
                    post_process=False,
                )
            except Exception as e:
                logger.warning("MTCNN not initialized: %s; using Haar cascade.", e)
                self._mtcnn = False
        return self._mtcnn if self._mtcnn is not False else None

    def detect_face_bbox(self, image_rgb: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Find largest frontal human face bounding box (x, y, w, h)."""
        h, w = image_rgb.shape[:2]

        # 1. Try MTCNN first if available
        mtcnn = self._get_mtcnn()
        if mtcnn is not None:
            try:
                pil_img = Image.fromarray(image_rgb)
                boxes, _ = mtcnn.detect(pil_img)
                if boxes is not None and len(boxes) > 0:
                    # Pick largest bounding box by area
                    best_box = None
                    max_area = 0
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        area = max(0, x2 - x1) * max(0, y2 - y1)
                        if area > max_area and area >= 1200:  # At least ~35x35
                            max_area = area
                            best_box = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    if best_box is not None:
                        return best_box
            except Exception as e:
                logger.debug("MTCNN detection exception: %s", e)

        # 2. Fallback to OpenCV Haar Cascade
        cascade = self._get_cascade()
        if cascade is not None:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(36, 36),
            )
            if len(faces) > 0:
                # Select largest face by area
                best_face = max(faces, key=lambda f: f[2] * f[3])
                return int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])

        return None

    def crop_and_format_face(
        self,
        image_rgb: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Crop face with margin_ratio (18%), resize to 224x224 RGB, float32 in [0, 255]."""
        x, y, bw, bh = bbox
        ih, iw = image_rgb.shape[:2]

        # 18% margin on all 4 borders
        dx = int(round(bw * self.margin_ratio))
        dy = int(round(bh * self.margin_ratio))

        x1 = max(0, x - dx)
        y1 = max(0, y - dy)
        x2 = min(iw, x + bw + dx)
        y2 = min(ih, y + bh + dy)

        crop = image_rgb[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            crop = cv2.resize(image_rgb, (self.target_size, self.target_size))
        else:
            crop = cv2.resize(crop, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

        # float32 in [0.0, 255.0] (unnormalized RGB layout NHWC)
        return crop.astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Face-Swap Detector: Samson5827/deepfakeguard-v2 (ONNX export on CPU)
# ---------------------------------------------------------------------------

class FaceSwapDetector:
    """Face-swap manipulation detector using Samson5827/deepfakeguard-v2 ONNX on CPU.

    Model Card Specification:
    - 18% margin on cropped face
    - Resize to 224x224 RGB
    - float32 pixels in range [0, 255]
    - ONNX input layout: (B, 224, 224, 3) (NHWC)
    - Output: scalar REAL score
    - Face-swap score = 1.0 - real_output
    - If no detectable face: returns score = None, reason = "No usable face"
    """

    def __init__(self):
        self.session = None
        self.model_loaded = False
        self.error: Optional[str] = None
        self.model_name = "Samson5827/deepfakeguard-v2"
        self.cropper = FaceCropHelper(target_size=224, margin_ratio=0.18)

    def load(self) -> bool:
        if self.model_loaded and self.session is not None:
            return True
        try:
            from huggingface_hub import hf_hub_download
            import onnxruntime as ort

            labels = json.loads(Path(hf_hub_download(self.model_name, "labels_v2.json")).read_text())
            self.fake_index = positive_index(labels["index_to_label"], {"fake"}, {"real"})
            model_path = hf_hub_download(
                repo_id=self.model_name,
                filename="deepfakeguard_v2.onnx",
            )
            # Create ONNX session on CPU with low thread count to bound CPU usage
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                model_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.model_loaded = True
            logger.info("Successfully loaded ONNX checkpoint: %s", self.model_name)
            return True
        except Exception as e:
            self.error = str(e)
            logger.error("Failed to load %s: %s", self.model_name, e)
            return False

    def predict_crop(self, face_crop_nhwc: np.ndarray) -> float:
        """Infers scalar fake probability from a single (1, 224, 224, 3) float32 [0..255] crop."""
        if not self.model_loaded and not self.load():
            raise RuntimeError(f"Model unavailable: {self.error}")

        input_name = self.session.get_inputs()[0].name
        raw_out = self.session.run(None, {input_name: face_crop_nhwc})
        real_score = float(raw_out[0][0, 0])
        if not math.isfinite(real_score) or not 0 <= real_score <= 1:
            raise ValueError("Invalid face detector sigmoid output")
        fake_score = real_score if self.fake_index == 1 else 1 - real_score
        return fake_score

    def predict_image(
        self,
        image_rgb: np.ndarray,
        provided_crop: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Detect face in photographic image, crop with 18% margin, and predict fake score."""
        if not self.model_loaded and not self.load():
            return {
                "score": None,
                "has_face": False,
                "reason": f"Model unavailable: {self.error}",
                "model": self.model_name,
            }

        # 1. Obtain face crop
        if provided_crop is not None:
            crop = cv2.resize(provided_crop, (224, 224)).astype(np.float32)
            has_face = True
        else:
            bbox = self.cropper.detect_face_bbox(image_rgb)
            if bbox is None:
                return {
                    "score": None,
                    "has_face": False,
                    "reason": "No usable face",
                    "model": self.model_name,
                }
            crop = self.cropper.crop_and_format_face(image_rgb, bbox)
            has_face = True

        # 2. Run ONNX inference
        face_input = np.expand_dims(crop, axis=0)  # (1, 224, 224, 3)
        fake_score = self.predict_crop(face_input)

        return {
            "score": round(fake_score, 4),
            "has_face": has_face,
            "reason": None,
            "model": self.model_name,
        }

    def predict_video_frames(
        self,
        frames_rgb: List[np.ndarray],
    ) -> Dict[str, Any]:
        """Process sampled video frames with deepfakeguard-v2 face-swap detection.

        Returns median frame score across usable faces and number of frames with usable faces.
        """
        if not self.model_loaded and not self.load():
            return {
                "score": None,
                "frames_analyzed": len(frames_rgb),
                "faces_detected": 0,
                "has_face": False,
                "reason": f"Model unavailable: {self.error}",
                "model": self.model_name,
                "frame_scores": [],
            }

        frame_scores: List[float] = []
        inference_errors = 0
        for frame in frames_rgb:
            bbox = self.cropper.detect_face_bbox(frame)
            if bbox is not None:
                crop = self.cropper.crop_and_format_face(frame, bbox)
                face_input = np.expand_dims(crop, axis=0)
                try:
                    f_score = self.predict_crop(face_input)
                    frame_scores.append(f_score)
                except Exception as e:
                    inference_errors += 1
                    logger.warning("Frame face scoring error: %s", e)

        faces_count = len(frame_scores)
        if faces_count == 0:
            return {
                "score": None,
                "frames_analyzed": len(frames_rgb),
                "faces_detected": 0,
                "has_face": False,
                "reason": "Face inference failed on detected crops" if inference_errors else "No usable face",
                "model": self.model_name,
                "frame_scores": [],
            }

        median_score = float(np.median(frame_scores))
        return {
            "score": round(median_score, 4),
            "frames_analyzed": len(frames_rgb),
            "faces_detected": faces_count,
            "has_face": True,
            "reason": None,
            "model": self.model_name,
            "frame_scores": [round(s, 4) for s in frame_scores],
        }


# ---------------------------------------------------------------------------
# 3. General AI-Image Detector: Organika/sdxl-detector
# ---------------------------------------------------------------------------

class SDXLDetector:
    """General AI-image detector for diffusion/synthetic imagery using Organika/sdxl-detector."""

    def __init__(self):
        self.processor = None
        self.model = None
        self.model_loaded = False
        self.error: Optional[str] = None
        self.model_name = "Organika/sdxl-detector"

    def load(self) -> bool:
        if self.model_loaded and self.model is not None:
            return True
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model, loading = AutoModelForImageClassification.from_pretrained(
                self.model_name, use_safetensors=True, output_loading_info=True)
            if loading["missing_keys"] or loading["mismatched_keys"]:
                raise ValueError("Image checkpoint did not supply all model weights")
            self.model.eval()
            self.model_loaded = True
            logger.info("Successfully loaded %s", self.model_name)
            return True
        except Exception as e:
            self.error = str(e)
            logger.error("Failed to load %s: %s", self.model_name, e)
            return False

    def predict_image(self, pil_image: Image.Image) -> Dict[str, Any]:
        """Detect synthetic / AI image generation patterns using Swin Transformer backbone."""
        if not self.model_loaded and not self.load():
            return {
                "score": None,
                "available": False,
                "reason": f"Model unavailable: {self.error}",
                "model": self.model_name,
            }

        try:
            inputs = self.processor(images=pil_image.convert("RGB"), return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)
                # id2label: {0: 'artificial', 1: 'human'}
                index = positive_index(self.model.config.id2label, {"artificial"}, {"human"})
                artificial_score = float(probs[0, index].item())
                if not math.isfinite(artificial_score):
                    raise ValueError("Nonfinite image model output")

            return {
                "score": round(artificial_score, 4),
                "available": True,
                "reason": None,
                "model": self.model_name,
            }
        except Exception as e:
            logger.exception("SDXLDetector inference failed: %s", e)
            return {
                "score": None,
                "available": False,
                "reason": f"Inference failure: {str(e)}",
                "model": self.model_name,
            }
