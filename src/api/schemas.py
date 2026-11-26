"""Pydantic schemas for the FastAPI deepfake detection inference API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModelReadiness(BaseModel):
    loaded: bool = Field(..., description="Whether model object is loaded in memory")
    checkpoint_loaded: bool = Field(..., description="Whether trained weights checkpoint was loaded")
    checkpoint_path: Optional[str] = Field(None, description="Path to loaded checkpoint file")
    is_mock: bool = Field(False, description="Whether model is using a mock backbone or mock weights")
    model_name: str = Field(..., description="Architecture name")
    error: Optional[str] = Field(None, description="Error message if checkpoint loading failed")


class SentenceEvidence(BaseModel):
    start: int
    end: int
    text: str
    score: float


class ScoreSpan(SentenceEvidence):
    new_tokens: int


class VideoEvidence(BaseModel):
    band: Optional[tuple[float, float]] = None
    score: Optional[float] = Field(None, description="Video fake probability in [0, 1]")
    frames_analyzed: int = Field(0, description="Total video frames sampled")
    faces_detected: int = Field(0, description="Number of frames with detected face crops")
    available: bool = Field(True, description="Whether video modality was successfully processed")
    model: str = Field("Samson5827/deepfakeguard-v2", description="Video backbone architecture")
    checkpoint_loaded: bool = Field(True, description="Whether trained weights checkpoint was loaded")
    is_mock: bool = Field(False, description="Whether video model uses mock or untrained weights")
    reason: Optional[str] = Field(None, description="Diagnostic reason when score is omitted")


class AudioEvidence(BaseModel):
    band: Optional[tuple[float, float]] = None
    usable_seconds: float = 0
    windows: int = 0
    score: Optional[float] = Field(None, description="Audio fake probability in [0, 1]")
    duration_seconds: float = Field(0.0, description="Extracted audio duration in seconds")
    available: bool = Field(True, description="Whether audio modality was successfully processed")
    model: str = Field(..., description="Configured acoustic checkpoint")
    checkpoint_loaded: bool = Field(True, description="Whether trained weights checkpoint was loaded")
    is_mock: bool = Field(False, description="Whether audio model uses mock backbone or untrained weights")
    reason: Optional[str] = Field(None, description="Diagnostic reason when score is omitted")


class FusionEvidence(BaseModel):
    ranking_score: Optional[float] = None
    band: Optional[tuple[float, float]] = None
    score: Optional[float] = Field(None, description="Uncalibrated decision index in [0, 1]")
    method: str = Field("separate_modality_assessment", description="Fusion methodology")
    threshold_used: Optional[float] = Field(None, description="Calibrated abstention threshold tau")
    threshold_name: Optional[str] = Field(None, description="Name of threshold applied")
    checkpoint_loaded: bool = Field(False, description="Whether trained weights checkpoint was loaded")
    is_calibrated: bool = Field(False, description="Whether thresholds are calibrated on validation data")


class ImageEvidence(BaseModel):
    score: Optional[float] = Field(None, description="Image manipulation probability in [0, 1]")
    faces_detected: int = Field(0, description="Detected faces in the image")
    available: bool = Field(True, description="Whether image analysis succeeded")
    model: str = Field("Samson5827/deepfakeguard-v2", description="Image feature extractor model")
    resolution: Optional[str] = Field(None, description="Image dimensions WxH")
    reason: Optional[str] = Field(None, description="Diagnostic reason when score is omitted")
    check_type: Optional[str] = Field("face_swap", description="Evaluation check type: face_swap or general_ai")
    sdxl_score: Optional[float] = Field(None, description="Organika/sdxl-detector synthetic score if requested")


class TextEvidence(BaseModel):
    band: Optional[tuple[float, float]] = None
    sentences: List[SentenceEvidence] = Field(default_factory=list)
    spans: List[ScoreSpan] = Field(default_factory=list)
    score: Optional[float] = Field(None, description="Synthetic text probability in [0, 1]")
    word_count: int = Field(0, description="Word count of analyzed text")
    burstiness_score: Optional[float] = Field(None, description="Sentence length variance metric")
    perplexity_score: Optional[float] = Field(None, description="Perplexity / entropy metric")
    lexical_diversity: Optional[float] = Field(None, description="Type-token ratio")
    available: bool = Field(True, description="Whether text analysis succeeded")
    model: str = Field("fakespot-ai/roberta-base-ai-text-detection-v1", description="Text analysis model")
    label: Optional[str] = Field("Experimental AI-writing style estimate", description="Descriptive result label")
    reason: Optional[str] = Field(None, description="Diagnostic reason when score is omitted")


class CrossModalTextEvidence(BaseModel):
    transcript: Optional[str] = None
    language: Optional[str] = None
    wer: Optional[float] = None
    cer: Optional[float] = None
    mismatch: Optional[float] = None
    similarity_method: Optional[str] = None
    cross_modal_consistency: Optional[float] = Field(None, description="Consistency score between visual/audio and text")
    extracted_text_detected: bool = Field(False, description="Whether on-screen text was identified")
    ocr_word_count: int = Field(0, description="Number of detected on-screen words")
    transcript_status: Optional[str] = Field("Transcript comparison unavailable", description="Status of transcript comparison")
    reason: Optional[str] = Field(None, description="Diagnostic explanation")


class TimingEvidence(BaseModel):
    frame_sampling_sec: Optional[float] = None
    face_detection_sec: Optional[float] = None
    audio_extraction_sec: Optional[float] = None
    video_inference_sec: Optional[float] = None
    audio_inference_sec: Optional[float] = None
    fusion_inference_sec: Optional[float] = None
    total_pipeline_sec: float


class PredictResponse(BaseModel):
    prediction: str = Field(..., description="Verdict: REAL, FAKE, or UNCERTAIN")
    confidence: Optional[float] = Field(None, description="Top-class confidence score in [0.5, 1.0]")
    prediction_path: str = Field(..., description="Inference path taken: fusion, video_only, audio_only, image_only, text_only, video_text_fusion, or no_modalities")
    is_mock: bool = Field(False, description="Whether inference results are from mock or untrained models")
    models_ready: bool = Field(True, description="Whether all models have trained checkpoints loaded")
    video: VideoEvidence
    audio: AudioEvidence
    fusion: FusionEvidence
    timing: TimingEvidence
    warnings: List[str] = Field(default_factory=list, description="Diagnostic warnings or issues encountered")
    image: Optional[ImageEvidence] = None
    text: Optional[TextEvidence] = None
    cross_modal: Optional[CrossModalTextEvidence] = None
    disagreement: Optional[str] = Field(None, description="Cross-modal disagreement explanation")
    v2_multi_verify: Optional[Dict[str, Any]] = Field(None, description="VeriSight v2 Multi-Verification evidence (Grad-CAM, cross-modal consistency, uncertainty)")


class TextPredictRequest(BaseModel):
    text: str = Field(..., description="Text content to inspect for synthetic / AI generation")


class HealthResponse(BaseModel):
    status: str = Field("ok", description="Service health status")
    version: str = Field("0.1.0", description="API version")
    device: str = Field("cpu", description="Execution hardware device")
    ffmpeg_available: bool = Field(True, description="Whether ffmpeg CLI binary is on PATH")
    models_loaded: bool = Field(False, description="Whether neural network models are in memory")
    models_ready: bool = Field(False, description="Whether trained checkpoints are loaded for all models")
    thresholds_calibrated: bool = Field(False, description="Whether abstention thresholds were calibrated")
    video_model: Optional[ModelReadiness] = None
    audio_model: Optional[ModelReadiness] = None
    fusion_model: Optional[ModelReadiness] = None
