"""VeriSight v2 — Deliverable B: Whisper ASR Batch Transcription & Audio-Text Cache.

Laptop 2 (Audio/ASR/AV-Sync Engineer) owns this file.
DO NOT MODIFY from Laptop 1 (Piyush / Vision-Text-Fusion-UI branch).

Runs faster-whisper on a directory of audio/video clips → produces:
  data/cache/asr_transcripts_l2.json

Schema of asr_transcripts_l2.json:
{
  "<clip_id>": {
    "file": "<relative path to clip>",
    "language": "en" | "hi" | ...,
    "transcript": "<full transcript text>",
    "word_timestamps": [
      {"word": "hello", "start": 0.24, "end": 0.52, "probability": 0.98},
      ...
    ],
    "duration_seconds": 5.3,
    "whisper_model": "base"
  },
  ...
}

Usage:
  python -m src.pipelines.asr_batch --input Dataset_L2/ASVspoof2019_LA --output data/cache/asr_transcripts_l2.json
  python -m src.pipelines.asr_batch --input Dataset_L2/video_ood --output data/cache/asr_transcripts_l2_ood.json --ext .mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Supported audio/video extensions
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
ALL_EXTS = AUDIO_EXTS | VIDEO_EXTS


def _load_whisper_model(model_size: str = "base", compute_type: str = "float16", device: str = "cuda"):
    """Load faster-whisper model, fall back to 'int8' on VRAM-limited devices."""
    try:
        from faster_whisper import WhisperModel
        logger.info(f"Loading faster-whisper '{model_size}' on {device} ({compute_type}) ...")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        return model
    except Exception as exc:
        logger.warning(f"faster-whisper load failed ({exc}). Trying CPU int8 fallback ...")
        try:
            from faster_whisper import WhisperModel
            return WhisperModel(model_size, device="cpu", compute_type="int8")
        except Exception as exc2:
            raise RuntimeError(
                "Could not load faster-whisper. Install with: pip install faster-whisper"
            ) from exc2


def transcribe_file(
    model,
    file_path: Path,
    beam_size: int = 5,
    language: Optional[str] = None,
) -> Dict:
    """Transcribe a single audio/video file. Returns a transcript dict entry."""
    try:
        segments, info = model.transcribe(
            str(file_path),
            beam_size=beam_size,
            language=language,
            word_timestamps=True,
        )

        transcript_parts = []
        word_timestamps: List[Dict] = []

        for segment in segments:
            transcript_parts.append(segment.text.strip())
            if segment.words:
                for word in segment.words:
                    word_timestamps.append({
                        "word": word.word.strip(),
                        "start": round(word.start, 3),
                        "end": round(word.end, 3),
                        "probability": round(word.probability, 4),
                    })

        return {
            "file": str(file_path),
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "transcript": " ".join(transcript_parts),
            "word_timestamps": word_timestamps,
            "duration_seconds": round(info.duration, 3),
            "whisper_model": getattr(model, "model_size_or_path", "base"),
        }

    except Exception as exc:
        logger.warning(f"Transcription failed for {file_path}: {exc}")
        return {
            "file": str(file_path),
            "language": None,
            "language_probability": None,
            "transcript": "",
            "word_timestamps": [],
            "duration_seconds": None,
            "whisper_model": None,
            "error": str(exc),
        }


def run_batch_asr(
    input_dir: str | Path,
    output_json: str | Path,
    model_size: str = "base",
    compute_type: str = "float16",
    device: str = "cuda",
    language: Optional[str] = None,
    ext_filter: Optional[List[str]] = None,
    max_files: Optional[int] = None,
    resume: bool = True,
) -> Dict:
    """Run faster-whisper ASR on all audio/video files in input_dir.

    Args:
        input_dir:    Directory containing audio/video clips (searched recursively).
        output_json:  Path to save the transcript cache JSON.
        model_size:   Whisper model size ("tiny", "base", "small", "medium", "large-v3").
        compute_type: "float16" for CUDA, "int8" for CPU.
        device:       "cuda" or "cpu".
        language:     Force language (None = auto-detect, "en", "hi", etc.).
        ext_filter:   List of extensions to include (default: all audio+video).
        max_files:    Limit total number of files processed (for quick tests).
        resume:       If True, skip files already in output_json.

    Returns:
        dict: Full transcript cache (also written to output_json).
    """
    input_dir = Path(input_dir)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    allowed_exts = set(ext_filter) if ext_filter else ALL_EXTS

    # Discover files
    files = sorted([
        f for f in input_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in allowed_exts
    ])
    if max_files:
        files = files[:max_files]

    logger.info(f"Found {len(files)} files in {input_dir}")

    # Resume: load existing cache
    cache: Dict = {}
    if resume and output_json.exists():
        with open(output_json, "r", encoding="utf-8") as f:
            cache = json.load(f)
        logger.info(f"Resuming — {len(cache)} files already transcribed.")

    skip_keys = set(cache.keys())

    # Load model once
    model = _load_whisper_model(model_size=model_size, compute_type=compute_type, device=device)

    done = 0
    t0 = time.time()
    for i, fpath in enumerate(files, 1):
        clip_id = fpath.stem
        # Make unique by parent folder name if duplicate stems
        clip_id = f"{fpath.parent.name}__{fpath.stem}"
        if clip_id in skip_keys:
            continue

        logger.info(f"[{i}/{len(files)}] Transcribing: {fpath.name}")
        result = transcribe_file(model, fpath, language=language)
        cache[clip_id] = result
        done += 1

        # Save incrementally every 50 files
        if done % 50 == 0:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
            elapsed = time.time() - t0
            logger.info(f"Checkpoint saved ({done} new | {len(cache)} total) — {elapsed:.0f}s elapsed")

    # Final save
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    total_time = time.time() - t0
    logger.info(
        f"ASR complete: {done} new transcriptions | {len(cache)} total cached | "
        f"Saved → {output_json} | Time: {total_time:.1f}s"
    )
    return cache


def compute_audio_text_consistency(
    asr_transcript: str,
    caption: str,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> float:
    """Layer-2 Cross-Modal Check: Audio-Text (ASR transcript vs user caption) consistency.

    Uses multilingual sentence embedding cosine similarity.
    Returns score in [0, 1] where 1 = identical semantic content.
    """
    if not asr_transcript.strip() or not caption.strip():
        return 0.5  # Neutral when either is empty

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer(model_name)
        embs = model.encode([asr_transcript.strip(), caption.strip()], normalize_embeddings=True)
        score = float(np.dot(embs[0], embs[1]))
        return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))
    except Exception:
        # Fallback: simple word-overlap (Jaccard)
        a_words = set(asr_transcript.lower().split())
        b_words = set(caption.lower().split())
        if not a_words and not b_words:
            return 1.0
        return len(a_words & b_words) / len(a_words | b_words) if (a_words | b_words) else 0.5


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="VeriSight v2 — Whisper ASR Batch Transcription")
    parser.add_argument("--input", required=True, help="Directory of audio/video clips")
    parser.add_argument("--output", default="data/cache/asr_transcripts_l2.json", help="Output JSON path")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Whisper model size")
    parser.add_argument("--compute_type", default="float16", choices=["float16", "int8", "float32"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--language", default=None, help="Force language (en/hi/None for auto)")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--ext", nargs="+", default=None, help="Extensions to include (.wav .mp4 ...)")
    parser.add_argument("--no_resume", action="store_true", help="Restart from scratch (ignore cache)")
    args = parser.parse_args()

    run_batch_asr(
        input_dir=args.input,
        output_json=args.output,
        model_size=args.model,
        compute_type=args.compute_type,
        device=args.device,
        language=args.language,
        ext_filter=args.ext,
        max_files=args.max_files,
        resume=not args.no_resume,
    )
