"""Pretrained-only inference shared by FastAPI, the benchmark and the live checks."""
import time
import subprocess
import wave
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.config import load_config
from .decision import Signal, classify, fuse
from .language import TEXT_MODELS, analyze_text, compare_caption, caption_reference
from .runtime import ModelSlot
from .speech import AUDIO_MODELS, NoAudioStreamError, acoustic, decode, transcribe

FACE = "Samson5827/deepfakeguard-v2"
IMAGE = "Organika/sdxl-detector"


def empty(name, reason):
    return dict(score=None, available=False, reason=reason, model=name, checkpoint_loaded=False)


def response(path, audio_model):
    return dict(prediction="UNCERTAIN", confidence=None, prediction_path=path, is_mock=False,
                models_ready=False, video=empty(FACE, "No video evidence requested"),
                audio=empty(audio_model, "No audio evidence requested"), warnings=[],
                fusion=dict(score=None, method=path, checkpoint_loaded=False, is_calibrated=False))


def direct(result, branch, band):
    score = result[branch]["score"]
    verdict = classify(score, band) if band[0] != band[1] else (
        "UNCERTAIN" if score is None else "FAKE" if score >= band[1] else "REAL")
    result[branch]["band"] = band
    if verdict == "UNCERTAIN" and score is not None:
        result[branch]["reason"] = "Score lies inside the uncalibrated uncertainty band."
    result.update(prediction=verdict, confidence=None if verdict == "UNCERTAIN" else
                  (score if verdict == "FAKE" else 1-score), models_ready=score is not None)


def finish(result, start):
    result["timing"] = {"total_pipeline_sec": time.perf_counter()-start}
    result["warnings"] = [e["reason"] for k,e in result.items()
                          if k in {"text","image","audio","video"} and e and e.get("reason")
                          and not e["reason"].endswith("evidence requested")]
    return result


def load_visual(name):
    from .pretrained_models import FaceSwapDetector, SDXLDetector
    detector = FaceSwapDetector() if name == FACE else SDXLDetector()
    if not detector.load():
        raise ValueError(detector.error)
    return detector


class InferencePipeline:
    def __init__(self, config=None):
        import torch
        torch.set_num_threads(min(torch.get_num_threads(), 2))
        self.config = config or load_config()
        self.settings = self.config.inference
        if self.settings.text_model not in TEXT_MODELS:
            raise ValueError("Unsupported text checkpoint")
        if self.settings.audio_model not in AUDIO_MODELS:
            raise ValueError("Unsupported acoustic checkpoint")
        self.slot = ModelSlot()

    @property
    def models_ready(self):
        return {FACE, self.settings.audio_model}.issubset(self.slot.verified)

    def get_readiness(self):
        def state(name):
            return dict(loaded=self.slot.name == name, checkpoint_loaded=name in self.slot.verified,
                        checkpoint_path=name, model_name=name, is_mock=False)
        return dict(models_ready=self.models_ready, thresholds_calibrated=False,
                    models_loaded=self.slot.name is not None, video_model=state(FACE),
                    audio_model=state(self.settings.audio_model), fusion_model=state(self.settings.strategy))

    def predict_text(self, text_content):
        start = time.perf_counter()
        result = response("text_only", self.settings.audio_model)
        result["text"] = analyze_text(text_content, self.slot, self.settings.text_model)
        direct(result, "text", self.settings.text_band)
        return finish(result, start)

    def _speech(self, path, caption):
        cross = dict(transcript=None, transcript_status="ASR disabled", reason=None,
                     extracted_text_detected=bool(caption), ocr_word_count=0)
        try:
            wav = decode(path)
        except NoAudioStreamError:
            cross["transcript_status"] = "No audio track"
            return empty(self.settings.audio_model, "No audio track detected in video."), cross, None
        except (OSError, ValueError, RuntimeError, EOFError, subprocess.SubprocessError, wave.Error) as exc:
            cross["transcript_status"] = "No decodable audio"
            return empty(self.settings.audio_model, f"Audio decoding failed: {exc}"), cross, None
        evidence = acoustic(wav, self.slot, self.settings.window_seconds, self.settings.audio_model)
        evidence["band"] = self.settings.audio_band
        text = None
        if self.settings.asr_enabled and evidence["usable_seconds"]:
            try:
                transcript, lang = transcribe(wav, self.slot)
                cross.update(transcript=transcript or None, language=lang,
                             transcript_status="Transcribed" if transcript else "No speech transcribed")
                if transcript:
                    text = analyze_text(transcript, self.slot, self.settings.text_model)
                    text["band"] = self.settings.text_band
                    if caption:
                        ref = caption_reference(caption, len(wav)/16000)
                        cross.update(compare_caption(ref, transcript))
                        if not ref:
                            cross["reason"] = "No reference words in the analyzed interval."
            except Exception as exc:
                cross.update(transcript_status="Speech analysis incomplete", reason=str(exc))
        elif self.settings.asr_enabled:
            cross["transcript_status"] = "Insufficient active audio"
        return evidence, cross, text

    def signals(self, result, caption=None, video=True):
        audio, cross = result["audio"], result["cross_modal"]
        branches = [Signal("acoustic", audio["score"], min(1, audio.get("usable_seconds",0)/5),
                           self.settings.audio_band)]
        if video:
            vis = result["video"]
            branches.insert(0, Signal("visual", vis["score"], vis.get("faces_detected",0)/8,
                                      self.settings.video_band))
        if self.settings.asr_enabled and caption and caption.strip():
            words = min(len(caption_reference(caption, audio.get("duration_seconds",30)).split()), len((cross.get("transcript") or "").split()))
            branches.append(Signal("caption mismatch", cross.get("mismatch"), min(1, words/20),
                                   self.settings.text_band))
        return branches

    def predict_audio(self, audio_path, text_content=None, strategy=None):
        start = time.perf_counter()
        result = response("audio_only", self.settings.audio_model)
        result["audio"], result["cross_modal"], result["text"] = self._speech(audio_path, text_content)
        direct(result, "audio", self.settings.audio_band)
        if text_content and text_content.strip():
            result.update(fuse(self.signals(result,text_content,False), strategy or self.settings.strategy,
                               self.settings.fusion_band))
        return finish(result, start)

    def predict(self, video_path, text_content=None, strategy=None):
        from src.data.video_preprocessing import extract_frames
        start = time.perf_counter()
        result = response("fusion", self.settings.audio_model)
        frames = extract_frames(video_path, max_frames=8, max_side=1920)
        try:
            if not frames:
                raise ValueError("No decodable video frames")
            with self.slot.use(FACE, lambda: load_visual(FACE)) as detector:
                evidence = detector.predict_video_frames([np.asarray(f) for f in frames])
            del detector
            result["video"] = dict(**evidence, available=evidence["score"] is not None,
                                   checkpoint_loaded=True)
        except Exception as exc:
            result["video"] = empty(FACE, f"Visual inference unavailable: {exc}")
            result["video"]["frames_analyzed"] = len(frames)
        del frames
        result["video"]["band"] = self.settings.video_band
        result["audio"], result["cross_modal"], result["text"] = self._speech(video_path, text_content)
        signals = self.signals(result, text_content)
        result.update(fuse(signals, strategy or self.settings.strategy, self.settings.fusion_band))
        vis, audio = result["video"]["available"], result["audio"]["available"]
        result["models_ready"] = vis and audio
        result["prediction_path"] = ("video_text_fusion" if text_content else
                                     "fusion" if vis and audio else "video_only" if vis else
                                     "audio_only" if audio else "no_modalities")
        return finish(result, start)

    def predict_video_text(self, video_path, text_content=None, strategy=None):
        return self.predict(video_path, text_content, strategy)

    def predict_image(self, image_path, check_type="face_swap"):
        start = time.perf_counter()
        if check_type not in {"sdxl", "general_ai", "face_swap"}:
            raise ValueError("Unknown image check")
        name = FACE if check_type == "face_swap" else IMAGE
        try:
            with Image.open(Path(image_path)) as im:
                width, height = im.size
                if width*height > 24_000_000:
                    raise ValueError("Image exceeds the CPU decoder limit of 24 megapixels")
                im.draft("RGB", (960, 960))
                im.thumbnail((960, 960))
                rgb = im.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Could not decode image") from exc
        result = response("image_only", self.settings.audio_model)
        try:
            with self.slot.use(name, lambda: load_visual(name)) as detector:
                ev = detector.predict_image(np.asarray(rgb) if name == FACE else rgb)
        except Exception as exc:
            ev = empty(name, f"Image inference unavailable: {exc}")
        result["image"] = dict(score=ev["score"], model=name, reason=ev.get("reason"),
                               available=ev["score"] is not None, check_type=check_type,
                               faces_detected=int(ev.get("has_face",False)),
                               resolution=f"{width}x{height}",
                               sdxl_score=ev["score"] if name == IMAGE else None)
        direct(result, "image", (.5, .5))
        return finish(result, start)
