"""VeriSight v2 — Laptop 2 Dataset Downloader & Organizer.

Downloads Laptop 2 datasets into ./Dataset_L2/ (gitignored):
  1. ASVspoof 2019 LA       -> Dataset_L2/ASVspoof2019_LA/
     Default (--source hf): fast HuggingFace CDN mirror (Bisher/ASVspoof_2019_LA),
     unpacked into official ASVspoof2019_LA_{train,dev,eval}/flac + cm_protocols.
     Fallback (--source datashare): Edinburgh DataShare LA.zip.
  2. OOD Audio              -> Dataset_L2/audio_ood/{real,fake}/
  3. OOD Video (AV + faces) -> Dataset_L2/video_ood/{real,fake}/

Usage:
  python scripts/download_datasets_l2.py --dataset all
  python scripts/download_datasets_l2.py --dataset asvspoof --source hf
  python scripts/download_datasets_l2.py --dataset audio_ood --max_ood_audio 600
  python scripts/download_datasets_l2.py --dataset video_ood --max_ood_video 40
"""

from __future__ import annotations

import argparse
import logging
import shutil
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


def _asvspoof_present(dest_root: Path) -> bool:
    for base in (dest_root, dest_root / "LA"):
        proto_dir = base / "ASVspoof2019_LA_cm_protocols"
        if (proto_dir / "ASVspoof2019.LA.cm.train.trn.txt").exists() and (
            proto_dir / "ASVspoof2019.LA.cm.eval.trl.txt"
        ).exists():
            return True
    return False


def download_asvspoof_hf(dest_root: Path) -> None:
    """Download ASVspoof 2019 LA from fast HuggingFace CDN & write official folder layout."""
    if _asvspoof_present(dest_root):
        logger.info("ASVspoof 2019 LA already present at %s — skipping.", dest_root)
        return

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    repo_id = "Bisher/ASVspoof_2019_LA"
    split_specs = [
        ("data/train-00000-of-00001.parquet", "ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt"),
        ("data/validation-00000-of-00001.parquet", "ASVspoof2019_LA_dev", "ASVspoof2019.LA.cm.dev.trl.txt"),
        ("data/test-00000-of-00001.parquet", "ASVspoof2019_LA_eval", "ASVspoof2019.LA.cm.eval.trl.txt"),
    ]

    proto_dir = dest_root / "ASVspoof2019_LA_cm_protocols"
    proto_dir.mkdir(parents=True, exist_ok=True)

    for parquet_rel, flac_split_dir, proto_name in split_specs:
        logger.info("Downloading %s from %s ...", parquet_rel, repo_id)
        local_pq = hf_hub_download(repo_id=repo_id, filename=parquet_rel, repo_type="dataset")

        flac_dir = dest_root / flac_split_dir / "flac"
        flac_dir.mkdir(parents=True, exist_ok=True)
        proto_path = proto_dir / proto_name

        pf = pq.ParquetFile(local_pq)
        lines: list[str] = []
        logger.info("Unpacking %s (%d rows) -> %s ...", parquet_rel, pf.metadata.num_rows, flac_dir)

        with tqdm(total=pf.metadata.num_rows, desc=flac_split_dir, unit="clip") as pbar:
            for batch in pf.iter_batches(
                batch_size=256,
                columns=["speaker_id", "audio_file_name", "audio", "system_id", "key"],
            ):
                d = batch.to_pydict()
                for spk, fname, aud, sys_id, key in zip(
                    d["speaker_id"], d["audio_file_name"], d["audio"], d["system_id"], d["key"]
                ):
                    out_flac = flac_dir / f"{fname}.flac"
                    if not out_flac.exists() and isinstance(aud, dict) and aud.get("bytes"):
                        out_flac.write_bytes(aud["bytes"])
                    lbl_str = "bonafide" if int(key) == 0 else "spoof"
                    lines.append(f"{spk} {fname} - {sys_id} {lbl_str}\n")
                    pbar.update(1)

        proto_path.write_text("".join(lines), encoding="utf-8")
        logger.info("Wrote %d protocol entries -> %s", len(lines), proto_path)


def download_asvspoof_datashare(dest_root: Path) -> None:
    """Download and extract ASVspoof 2019 LA from Edinburgh DataShare."""
    dest_root.mkdir(parents=True, exist_ok=True)
    if _asvspoof_present(dest_root):
        logger.info("ASVspoof 2019 LA already present at %s — skipping.", dest_root)
        return

    zip_path = dest_root / "LA.zip"
    if not zip_path.exists():
        _download_with_resume(ASVSPOOF_URLS, zip_path)

    logger.info("Extracting %s -> %s ...", zip_path, dest_root)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in tqdm(zf.infolist(), desc="Extracting LA.zip", unit="file"):
            zf.extract(member, dest_root)


def _decode_audio_entry(audio_obj) -> tuple:
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
    if "audio" in getattr(ds, "features", {}):
        ds = ds.cast_column("audio", Audio(decode=False))

    per_class = max_samples // 2
    n_real, n_fake = len(list(real_dir.glob("*.wav"))), len(list(fake_dir.glob("*.wav")))

    for idx, row in enumerate(tqdm(ds, desc="Downloading OOD audio")):
        if n_real >= per_class and n_fake >= per_class:
            break
        raw_lbl = row.get("label", row.get("is_fake", row.get("class")))
        lbl = _normalize_binary_label(raw_lbl)
        if lbl is None or (lbl == 0 and n_real >= per_class) or (lbl == 1 and n_fake >= per_class):
            continue
        audio_col = row.get("audio")
        if audio_col is None:
            continue
        try:
            arr, sr = _decode_audio_entry(audio_col)
            out_path = (real_dir if lbl == 0 else fake_dir) / f"ood_{idx:05d}.wav"
            sf.write(str(out_path), arr, sr)
            if lbl == 0:
                n_real += 1
            else:
                n_fake += 1
        except Exception as exc:
            logger.debug("Skipping OOD sample %d: %s", idx, exc)

    logger.info("OOD audio ready -> %s (real=%d, fake=%d)", dest_root, n_real, n_fake)


def _has_audio_and_face(video_path: Path) -> bool:
    """Return True if video has at least one audio stream."""
    try:
        import av
        with av.open(str(video_path)) as c:
            return len(c.streams.audio) > 0 and len(c.streams.video) > 0
    except Exception:
        return False


def _mux_video_with_fake_audio(video_src: Path, audio_src: Path, out_path: Path) -> bool:
    """Create a RealVideo-FakeAudio dubbing clip via PyAV for AV-sync/compression eval."""
    try:
        import av
        with av.open(str(video_src)) as vc, av.open(str(audio_src)) as ac, av.open(str(out_path), mode="w") as out_c:
            in_v = vc.streams.video[0]
            fps = in_v.average_rate or 25
            out_v = out_c.add_stream("libx264", rate=fps)
            out_v.width = in_v.width - (in_v.width % 2)
            out_v.height = in_v.height - (in_v.height % 2)
            out_v.pix_fmt = "yuv420p"
            out_v.options = {"crf": "20"}

            out_a = out_c.add_stream("aac", rate=16000)
            resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=16000)

            max_frames = int(float(fps) * 6.0)
            for idx, frame in enumerate(vc.decode(video=0)):
                if idx >= max_frames:
                    break
                if frame.width != out_v.width or frame.height != out_v.height:
                    frame = frame.reformat(width=out_v.width, height=out_v.height, format="yuv420p")
                for pkt in out_v.encode(frame):
                    out_c.mux(pkt)
            for pkt in out_v.encode(None):
                out_c.mux(pkt)

            for frame in ac.decode(audio=0):
                for rframe in resampler.resample(frame):
                    for pkt in out_a.encode(rframe):
                        out_c.mux(pkt)
            for rframe in resampler.resample(None):
                for pkt in out_a.encode(rframe):
                    out_c.mux(pkt)
            for pkt in out_a.encode(None):
                out_c.mux(pkt)
        return out_path.exists()
    except Exception as exc:
        logger.debug("Mux failed: %s", exc)
        return False


def download_video_ood(dest_root: Path, max_videos: int = 40) -> None:
    """Download talking-head videos WITH audio streams for AV-sync, ASR & H.264 tests."""
    real_dir = dest_root / "real"
    fake_dir = dest_root / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    per_class = max_videos // 2

    # 1. Real talking-head AV clips from FreedomIntelligence/TalkVid (videos_w_audios) + UniDataPro
    existing_real = [p for p in real_dir.glob("*.*") if _has_audio_and_face(p)]
    if len(existing_real) < per_class:
        try:
            items = list(
                api.list_repo_tree(
                    "FreedomIntelligence/TalkVid",
                    path_in_repo="TalkVid-bench/age/videos_w_audios",
                    repo_type="dataset",
                    recursive=True,
                )
            )
            mp4s = [x.path for x in items if x.path.lower().endswith(".mp4")]
            for rf in tqdm(mp4s[: per_class + 5], desc="Downloading TalkVid real AV"):
                if len(list(real_dir.glob("*.mp4"))) >= per_class:
                    break
                cached = Path(hf_hub_download("FreedomIntelligence/TalkVid", filename=rf, repo_type="dataset"))
                if _has_audio_and_face(cached):
                    shutil.copy2(cached, real_dir / Path(rf).name)
        except Exception as exc:
            logger.warning("TalkVid download failed: %s", exc)

        try:
            u_real = list(api.list_repo_tree("UniDataPro/deepfake-videos-dataset", path_in_repo="video", repo_type="dataset"))
            for item in u_real:
                if len(list(real_dir.glob("*.*"))) >= per_class:
                    break
                if item.path.lower().endswith((".mp4", ".mov")):
                    cached = Path(hf_hub_download("UniDataPro/deepfake-videos-dataset", filename=item.path, repo_type="dataset"))
                    if _has_audio_and_face(cached):
                        shutil.copy2(cached, real_dir / f"unidata_{Path(item.path).name}")
        except Exception as exc:
            logger.warning("UniDataPro real download failed: %s", exc)

    # 2. Deepfake AV clips from UniDataPro + FakeAudio-RealVideo dubs if needed
    existing_fake = [p for p in fake_dir.glob("*.*") if _has_audio_and_face(p)]
    if len(existing_fake) < per_class:
        try:
            items = list(api.list_repo_tree("UniDataPro/deepfake-videos-dataset", path_in_repo="deepfake", repo_type="dataset"))
            for item in items:
                if len(list(fake_dir.glob("*.*"))) >= per_class:
                    break
                if item.path.lower().endswith((".mp4", ".mov")):
                    cached = Path(hf_hub_download("UniDataPro/deepfake-videos-dataset", filename=item.path, repo_type="dataset"))
                    if _has_audio_and_face(cached):
                        shutil.copy2(cached, fake_dir / f"unidata_{Path(item.path).name}")
        except Exception as exc:
            logger.warning("UniDataPro fake download failed: %s", exc)

        # Supplement remaining fake AV slots with RealVideo + FakeSpeech dubs (FakeAVCeleb-style).
        real_clips = sorted(real_dir.glob("*.mp4"))
        fake_wavs = sorted(Path("Dataset_L2/audio_ood/fake").glob("*.wav"))
        idx = 0
        while len(list(fake_dir.glob("*.mp4"))) + len(list(fake_dir.glob("*.mov"))) < per_class and idx < len(real_clips) and idx < len(fake_wavs):
            out_fake = fake_dir / f"avdub_fake_{idx:03d}.mp4"
            if not out_fake.exists():
                _mux_video_with_fake_audio(real_clips[idx], fake_wavs[idx], out_fake)
            idx += 1

    logger.info(
        "OOD AV video ready -> %s (real=%d, fake=%d)",
        dest_root,
        len(list(real_dir.glob("*.*"))),
        len(list(fake_dir.glob("*.*"))),
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="VeriSight v2 — Laptop 2 Dataset Downloader")
    parser.add_argument("--dataset", default="all", choices=["all", "asvspoof", "audio_ood", "video_ood"])
    parser.add_argument("--source", default="hf", choices=["hf", "datashare"], help="ASVspoof source mirror")
    parser.add_argument("--max_ood_audio", type=int, default=600)
    parser.add_argument("--max_ood_video", type=int, default=40)
    args = parser.parse_args()

    base = Path("Dataset_L2")
    if args.dataset in ("all", "audio_ood"):
        download_audio_ood(base / "audio_ood", max_samples=args.max_ood_audio)
    if args.dataset in ("all", "video_ood"):
        download_video_ood(base / "video_ood", max_videos=args.max_ood_video)
    if args.dataset in ("all", "asvspoof"):
        if args.source == "hf":
            download_asvspoof_hf(base / "ASVspoof2019_LA")
        else:
            download_asvspoof_datashare(base / "ASVspoof2019_LA")


if __name__ == "__main__":
    main()
