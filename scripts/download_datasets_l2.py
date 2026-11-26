"""VeriSight v2 — Laptop 2 Dataset Downloader & Organizer.

Downloads Laptop 2 datasets into ./Dataset_L2/ (gitignored):
  1. ASVspoof 2019 LA       -> Dataset_L2/ASVspoof2019_LA/
  2. OOD Audio (WaveFake / Deepfake Audio Detection) -> Dataset_L2/audio_ood/{real,fake}/
  3. OOD Video (Cross-dataset / compression tests)   -> Dataset_L2/video_ood/{real,fake}/

Usage:
  python scripts/download_datasets_l2.py --dataset all
  python scripts/download_datasets_l2.py --dataset asvspoof
  python scripts/download_datasets_l2.py --dataset audio_ood --max_ood_audio 600
  python scripts/download_datasets_l2.py --dataset video_ood --max_ood_video 60
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)

ASVSPOOF_URLS = [
    "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip",
    "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3336/LA.zip",
]


def _download_with_resume(urls: list[str], dest: Path, chunk_size: int = 1024 * 1024) -> Path:
    """Download a large archive with HTTP Range resume support."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest.with_suffix(dest.suffix + ".part")

    last_err: Optional[Exception] = None
    for url in urls:
        try:
            existing = part_path.stat().st_size if part_path.exists() else 0
            req = urllib.request.Request(url, headers={"User-Agent": "VeriSight-L2-Downloader/1.0"})
            if existing > 0:
                req.add_header("Range", f"bytes={existing}-")

            with urllib.request.urlopen(req, timeout=60) as resp:
                status = getattr(resp, "status", 200)
                if existing > 0 and status != 206:
                    # Server ignored Range header; restart cleanly.
                    existing = 0
                    mode = "wb"
                else:
                    mode = "ab" if existing > 0 else "wb"

                content_len = resp.headers.get("Content-Length")
                total = (int(content_len) + existing) if content_len else None

                logger.info("Downloading %s -> %s", url, dest)
                with open(part_path, mode) as out_f, tqdm(
                    total=total,
                    initial=existing,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=dest.name,
                ) as pbar:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        out_f.write(chunk)
                        pbar.update(len(chunk))

            shutil.move(str(part_path), str(dest))
            return dest
        except Exception as exc:
            logger.warning("Download failed from %s: %s", url, exc)
            last_err = exc

    raise RuntimeError(f"All download mirrors failed for {dest.name}: {last_err}")


def download_asvspoof(dest_root: Path) -> None:
    """Download and extract ASVspoof 2019 LA (~3.5-7 GB zip)."""
    dest_root.mkdir(parents=True, exist_ok=True)
    if (dest_root / "ASVspoof2019_LA_cm_protocols").exists() or (
        dest_root / "LA" / "ASVspoof2019_LA_cm_protocols"
    ).exists():
        logger.info("ASVspoof 2019 LA already present at %s — skipping.", dest_root)
        return

    zip_path = dest_root / "LA.zip"
    if not zip_path.exists():
        _download_with_resume(ASVSPOOF_URLS, zip_path)

    logger.info("Extracting %s -> %s ...", zip_path, dest_root)
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.infolist()
        for member in tqdm(members, desc="Extracting LA.zip", unit="file"):
            zf.extract(member, dest_root)
    logger.info("ASVspoof 2019 LA extracted successfully to %s", dest_root)


def _decode_audio_entry(audio_obj) -> tuple:
    """Extract (numpy float32 array, sample_rate) from HuggingFace dataset audio column."""
    import io
    import numpy as np
    import soundfile as sf

    if isinstance(audio_obj, dict) and "array" in audio_obj:
        return np.asarray(audio_obj["array"], dtype=np.float32), int(audio_obj["sampling_rate"])
    if isinstance(audio_obj, dict) and audio_obj.get("bytes"):
        data, sr = sf.read(io.BytesIO(audio_obj["bytes"]), dtype="float32", always_2d=False)
        return data, int(sr)
    if isinstance(audio_obj, dict) and audio_obj.get("path"):
        data, sr = sf.read(audio_obj["path"], dtype="float32", always_2d=False)
        return data, int(sr)
    raise ValueError(f"Unsupported audio object structure: {type(audio_obj)}")


def _normalize_binary_label(raw_label) -> Optional[int]:
    if isinstance(raw_label, (int, float)):
        return 1 if int(raw_label) == 1 else 0
    s = str(raw_label).strip().lower()
    if s in {"real", "bonafide", "human", "genuine", "0"}:
        return 0
    if s in {"fake", "spoof", "synthetic", "ai", "1"}:
        return 1
    return None


def download_audio_ood(dest_root: Path, max_samples: int = 600) -> None:
    """Download balanced out-of-domain audio benchmark into real/ and fake/ folders."""
    real_dir = dest_root / "real"
    fake_dir = dest_root / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    existing_count = len(list(real_dir.glob("*.wav"))) + len(list(fake_dir.glob("*.wav")))
    if existing_count >= max_samples:
        logger.info("OOD audio already has %d files in %s — skipping.", existing_count, dest_root)
        return

    import soundfile as sf
    from datasets import Audio, load_dataset

    repo_id = "garystafford/deepfake-audio-detection"
    logger.info("Streaming OOD audio from HuggingFace: %s (target=%d clips) ...", repo_id, max_samples)
    ds = load_dataset(repo_id, split="train", streaming=True)
    # Disable torchcodec decoding so we can read raw bytes/arrays cleanly via soundfile.
    if "audio" in getattr(ds, "features", {}):
        ds = ds.cast_column("audio", Audio(decode=False))

    per_class = max_samples // 2
    n_real, n_fake = len(list(real_dir.glob("*.wav"))), len(list(fake_dir.glob("*.wav")))

    for idx, row in enumerate(tqdm(ds, desc="Downloading OOD audio")):
        if n_real >= per_class and n_fake >= per_class:
            break

        raw_lbl = row.get("label", row.get("is_fake", row.get("class")))
        lbl = _normalize_binary_label(raw_lbl)
        if lbl is None:
            continue
        if lbl == 0 and n_real >= per_class:
            continue
        if lbl == 1 and n_fake >= per_class:
            continue

        audio_col = row.get("audio")
        if audio_col is None:
            continue
        try:
            arr, sr = _decode_audio_entry(audio_col)
            out_dir = real_dir if lbl == 0 else fake_dir
            out_path = out_dir / f"ood_{idx:05d}.wav"
            sf.write(str(out_path), arr, sr)
            if lbl == 0:
                n_real += 1
            else:
                n_fake += 1
        except Exception as exc:
            logger.debug("Skipping OOD sample %d: %s", idx, exc)

    logger.info("OOD audio ready -> %s (real=%d, fake=%d)", dest_root, n_real, n_fake)


def download_video_ood(dest_root: Path, max_videos: int = 60) -> None:
    """Download balanced OOD video subset for compression & AV-sync testing."""
    real_dir = dest_root / "real"
    fake_dir = dest_root / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    existing = list(real_dir.glob("*.mp4")) + list(fake_dir.glob("*.mp4"))
    if len(existing) >= max_videos:
        logger.info("OOD video already has %d clips in %s — skipping.", len(existing), dest_root)
        return

    from huggingface_hub import HfApi, hf_hub_download

    repo_id = "gonnerthetooner/deepfake_video_data"
    api = HfApi()
    logger.info("Listing video files in HuggingFace repo: %s ...", repo_id)
    try:
        repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except Exception as exc:
        logger.warning("Could not list %s (%s). Trying fallback repo ...", repo_id, exc)
        return

    video_files = [f for f in repo_files if f.lower().endswith((".mp4", ".avi", ".mkv"))]
    real_files = [f for f in video_files if "real" in f.lower() or "original" in f.lower()]
    fake_files = [f for f in video_files if "fake" in f.lower() or "manipulated" in f.lower()]
    if not real_files and not fake_files:
        # Unlabeled layout: place half in real/ and half in fake/ or flat
        selected = [(f, real_dir) for f in video_files[:max_videos]]
    else:
        half = max_videos // 2
        selected = [(f, real_dir) for f in real_files[:half]] + [(f, fake_dir) for f in fake_files[:half]]

    for rf, target_dir in tqdm(selected, desc="Downloading OOD videos"):
        local_dest = target_dir / Path(rf).name
        if local_dest.exists():
            continue
        cached = hf_hub_download(repo_id=repo_id, filename=rf, repo_type="dataset")
        shutil.copy2(cached, local_dest)

    logger.info(
        "OOD video ready -> %s (real=%d, fake=%d)",
        dest_root,
        len(list(real_dir.glob("*.*"))),
        len(list(fake_dir.glob("*.*"))),
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="VeriSight v2 — Laptop 2 Dataset Downloader")
    parser.add_argument("--dataset", default="all", choices=["all", "asvspoof", "audio_ood", "video_ood"])
    parser.add_argument("--max_ood_audio", type=int, default=600)
    parser.add_argument("--max_ood_video", type=int, default=60)
    args = parser.parse_args()

    base = Path("Dataset_L2")
    if args.dataset in ("all", "audio_ood"):
        download_audio_ood(base / "audio_ood", max_samples=args.max_ood_audio)
    if args.dataset in ("all", "video_ood"):
        download_video_ood(base / "video_ood", max_videos=args.max_ood_video)
    if args.dataset in ("all", "asvspoof"):
        download_asvspoof(base / "ASVspoof2019_LA")


if __name__ == "__main__":
    main()
