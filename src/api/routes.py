"""FastAPI endpoint route definitions."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Literal
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from src.api.schemas import HealthResponse, PredictResponse, TextPredictRequest
from src.data.audio_preprocessing import is_ffmpeg_available
from src.inference.v2_bridge import run_v2_multi_verification

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".webm", ".mkv"}
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".aac", ".flac", ".m4a", ".ogg"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB max for local upload
MAX_TEXT_LENGTH_CHARS = 50_000  # 50,000 characters limit for text payload


async def reference_caption(text, subtitles):
    if subtitles is not None:
        if text and text.strip():
            raise HTTPException(400, "Supply one reference caption")
        raw = await subtitles.read(4 * MAX_TEXT_LENGTH_CHARS + 1)
        if len(raw) > 4 * MAX_TEXT_LENGTH_CHARS:
            raise HTTPException(413, "Subtitle file exceeds size limit")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise HTTPException(400, "Subtitles must use UTF-8") from exc
    if text and len(text) > MAX_TEXT_LENGTH_CHARS:
        raise HTTPException(400, "Caption exceeds maximum allowed length")
    return text


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Health status and dependency diagnostic check."""
    pipeline = getattr(request.app.state, "pipeline", None)
    config = getattr(request.app.state, "config", None)

    readiness = pipeline.get_readiness() if (pipeline and hasattr(pipeline, "get_readiness")) else None

    return HealthResponse(
        status="ok",
        version="0.1.0",
        device=config.device if config else "cpu",
        ffmpeg_available=is_ffmpeg_available(),
        models_loaded=bool(readiness and readiness["models_loaded"]),
        models_ready=readiness["models_ready"] if readiness else False,
        thresholds_calibrated=readiness["thresholds_calibrated"] if readiness else False,
        video_model=readiness["video_model"] if readiness else None,
        audio_model=readiness["audio_model"] if readiness else None,
        fusion_model=readiness["fusion_model"] if readiness else None,
    )


@router.post("/predict", response_model=PredictResponse)
async def predict_video(
    request: Request,
    file: UploadFile = File(...),
    text: Optional[str] = Form(None),
    subtitles: Optional[UploadFile] = File(None),
    strict: bool = False,
    strategy: Optional[Literal["disagreement_max","confidence_weighted_fusion"]] = None,
):
    """Analyze an uploaded video file (with optional on-screen text) and produce multimodal prediction and evidence."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Inference pipeline is not initialized.")

    # Validate file extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{suffix}'. Allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}",
        )

    # Save to temp file securely
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_path = Path(temp_file.name)

    try:
        size = 0
        while content := await file.read(1024 * 1024):  # 1MB chunks
            size += len(content)
            if size > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Uploaded file exceeds {MAX_FILE_SIZE_BYTES // (1024*1024)} MB limit.",
                )
            temp_file.write(content)
        temp_file.close()

        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded video file is empty (0 bytes).")

        # Run inference: if text is provided, run cross-modal video-text inference
        text = await reference_caption(text, subtitles)
        if text and text.strip():
            result = pipeline.predict_video_text(temp_path, text_content=text, strategy=strategy)
        else:
            result = pipeline.predict(temp_path, strategy=strategy)

        if strict and not result["models_ready"]:
            raise HTTPException(503, "Required video or audio evidence unavailable")
        result = run_v2_multi_verification(result, media_path=temp_path, modality="video", caption_text=text or "")
        return PredictResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Inference failed with unexpected error.")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        temp_file.close()
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@router.post("/predict/audio", response_model=PredictResponse)
async def predict_audio(
    request: Request,
    file: UploadFile = File(...),
    text: Optional[str] = Form(None),
    subtitles: Optional[UploadFile] = File(None),
    strategy: Optional[Literal["disagreement_max","confidence_weighted_fusion"]] = None,
):
    """Analyze a standalone audio file for acoustic deepfake signals."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Inference pipeline is not initialized.")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{suffix}'. Allowed: {sorted(ALLOWED_AUDIO_EXTENSIONS)}",
        )

    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_path = Path(temp_file.name)

    try:
        size = 0
        while content := await file.read(1024 * 1024):
            size += len(content)
            if size > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Uploaded file exceeds {MAX_FILE_SIZE_BYTES // (1024*1024)} MB limit.",
                )
            temp_file.write(content)
        temp_file.close()

        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty (0 bytes).")

        text = await reference_caption(text, subtitles)
        result = pipeline.predict_audio(temp_path, text, strategy)
        result = run_v2_multi_verification(result, media_path=temp_path, modality="audio", caption_text=text or "")
        return PredictResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Audio inference failed.")
        raise HTTPException(status_code=500, detail=f"Audio inference error: {str(e)}")
    finally:
        temp_file.close()
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@router.post("/predict/image", response_model=PredictResponse)
async def predict_image(
    request: Request,
    file: UploadFile = File(...),
    check_type: Optional[str] = Form("face_swap"),
):
    """Analyze a photographic image for face manipulation or generative synthesis."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Inference pipeline is not initialized.")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{suffix}'. Allowed: {sorted(ALLOWED_IMAGE_EXTENSIONS)}",
        )

    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_path = Path(temp_file.name)

    try:
        size = 0
        while content := await file.read(1024 * 1024):
            size += len(content)
            if size > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Uploaded file exceeds {MAX_FILE_SIZE_BYTES // (1024*1024)} MB limit.",
                )
            temp_file.write(content)
        temp_file.close()

        if size == 0:
            raise HTTPException(status_code=400, detail="Uploaded image file is empty (0 bytes).")

        if check_type not in {None, "face_swap", "sdxl", "general_ai"}:
            raise HTTPException(400, "Unknown image check")
        result = pipeline.predict_image(temp_path, check_type=check_type or "face_swap")
        result = run_v2_multi_verification(result, media_path=temp_path, modality="image")
        return PredictResponse(**result)
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning("Unprocessable image uploaded: %s", e)
        if "megapixels" in str(e):
            raise HTTPException(413, str(e)) from e
        raise HTTPException(
            status_code=400,
            detail="The uploaded file could not be decoded as a valid image. It may be corrupt or an unrecognized format.",
        )
    except Exception as e:
        logger.exception("Image inference failed.")
        raise HTTPException(status_code=500, detail=f"Image inference error: {str(e)}")
    finally:
        temp_file.close()
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@router.post("/predict/text", response_model=PredictResponse)
async def predict_text(
    request: Request,
):
    """Analyze text for synthetic generation, burstiness variance, and perplexity patterns."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Inference pipeline is not initialized.")

    content = None
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        try:
            data = await request.json()
            content = data.get("text")
        except Exception:
            pass
    elif "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        try:
            form = await request.form()
            content = form.get("text")
        except Exception:
            pass

    if not content:
        body_bytes = await request.body()
        if body_bytes:
            try:
                import json
                data = json.loads(body_bytes)
                content = data.get("text")
            except Exception:
                content = body_bytes.decode("utf-8", errors="ignore")

    if not content or not str(content).strip():
        raise HTTPException(status_code=400, detail="Text content must be provided.")

    text_str = str(content)
    if len(text_str) > MAX_TEXT_LENGTH_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Text exceeds maximum allowed length of {MAX_TEXT_LENGTH_CHARS:,} characters.",
        )

    try:
        result = pipeline.predict_text(text_str)
        result = run_v2_multi_verification(result, modality="text", caption_text=text_str)
        return PredictResponse(**result)
    except Exception as e:
        logger.exception("Text inference failed.")
        raise HTTPException(status_code=500, detail=f"Text inference error: {str(e)}")


@router.post("/predict/video-text", response_model=PredictResponse)
async def predict_video_text(
    request: Request,
    file: UploadFile = File(...),
    text: Optional[str] = Form(None),
    subtitles: Optional[UploadFile] = File(None),
    strategy: Optional[Literal["disagreement_max","confidence_weighted_fusion"]] = None,
):
    """Analyze a video file containing on-screen text, burnt-in captions, or subtitles."""
    # Delegate to predict_video which natively accepts video + optional subtitles/text
    return await predict_video(request=request, file=file, text=text, subtitles=subtitles, strict=False, strategy=strategy)
