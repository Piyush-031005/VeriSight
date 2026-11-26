"""VeriSight v2 — Deliverable D: Experiment 3 Generalization & Social-Media Compression Stress Suite.

Laptop 2 (Audio/ASR/AV-Sync Engineer) — EXPERIMENT SCRIPT.

USAGE:
  python scripts/run_exp3_robustness.py --config configs/v2_cuda.yaml

OUTPUTS:
  results/exp3_robustness.json
  results/av_sync_eval_l2.json
  results/plots/exp3_compression_curves.png
  results/plots/exp3_audio_cross_dataset.png
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_audio_xlsr_v2 import load_audio_tensor, resolve_asvspoof_root

logger = logging.getLogger(__name__)

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov"}


# ---------------------------------------------------------------------------
# Signal Degradation & Container Utilities
# ---------------------------------------------------------------------------

def add_gaussian_noise_snr(waveform: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise to a waveform at a given SNR (dB)."""
    signal_power = float(np.mean(waveform ** 2))
    if signal_power < 1e-12:
        return waveform
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.randn(*waveform.shape).astype(np.float32) * np.sqrt(noise_power)
    return (waveform + noise).clip(-1.0, 1.0)


def compress_video_h264(input_path: Path, output_path: Path, crf: int) -> bool:
    """Re-encode video with H.264 at the given CRF using ffmpeg CLI or PyAV fallback."""
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-c:v", "libx264", "-crf", str(crf),
            "-c:a", "aac", "-b:a", "128k",
            "-loglevel", "error",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode == 0:
            return True

    # PyAV fallback (uses bundled FFmpeg libraries without requiring ffmpeg.exe on PATH)
    try:
        import av
        with av.open(str(input_path)) as in_c, av.open(str(output_path), mode="w") as out_c:
            in_v = in_c.streams.video[0]
            fps = in_v.average_rate or 25
            out_v = out_c.add_stream("libx264", rate=fps)
            out_v.width = in_v.width - (in_v.width % 2)
            out_v.height = in_v.height - (in_v.height % 2)
            out_v.pix_fmt = "yuv420p"
            out_v.options = {"crf": str(crf)}

            in_a = in_c.streams.audio[0] if in_c.streams.audio else None
            out_a = out_c.add_stream("aac", rate=16000) if in_a else None
            resampler = av.audio.resampler.AudioResampler(format="fltp", layout="mono", rate=16000) if out_a else None

            for packet in in_c.demux([s for s in (in_v, in_a) if s is not None]):
                for frame in packet.decode():
                    if isinstance(frame, av.VideoFrame):
                        if frame.width != out_v.width or frame.height != out_v.height:
                            frame = frame.reformat(width=out_v.width, height=out_v.height, format="yuv420p")
                        for pkt in out_v.encode(frame):
                            out_c.mux(pkt)
                    elif isinstance(frame, av.AudioFrame) and out_a and resampler:
                        for rframe in resampler.resample(frame):
                            for pkt in out_a.encode(rframe):
                                out_c.mux(pkt)
            for pkt in out_v.encode(None):
                out_c.mux(pkt)
            if out_a:
                for pkt in out_a.encode(None):
                    out_c.mux(pkt)
        return output_path.exists()
    except Exception as exc:
        logger.warning("H.264 re-encode failed for %s (CRF %d): %s", input_path, crf, exc)
        return False


def extract_wav_from_video(video_path: Path, wav_out: Path, sample_rate: int = 16000) -> bool:
    """Extract mono 16 kHz WAV using ffmpeg CLI or PyAV + soundfile fallback."""
    if shutil.which("ffmpeg"):
        res = subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ar", str(sample_rate), "-ac", "1",
             "-loglevel", "error", str(wav_out)],
            capture_output=True, timeout=60,
        )
        if res.returncode == 0 and wav_out.exists():
            return True

    try:
        import soundfile as sf
        wav_t, ok = load_audio_tensor(video_path, sample_rate=sample_rate, max_samples=sample_rate * 30)
        if not ok:
            return False
        sf.write(str(wav_out), wav_t.numpy(), sample_rate)
        return wav_out.exists()
    except Exception as exc:
        logger.debug("extract_wav_from_video failed for %s: %s", video_path, exc)
        return False


def compress_frame_jpeg(frame_rgb: np.ndarray, quality: int) -> np.ndarray:
    """Encode and decode a frame via JPEG compression at the given quality."""
    import cv2
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def reencode_video_jpeg_frames(video_path: Path, out_video_path: Path, quality: int) -> bool:
    """Re-encode video frames after JPEG compression at given quality (audio unchanged if passed separately)."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        degraded_rgb = compress_frame_jpeg(rgb, quality=quality)
        writer.write(cv2.cvtColor(degraded_rgb, cv2.COLOR_RGB2BGR))
    cap.release()
    writer.release()
    return out_video_path.exists()


# ---------------------------------------------------------------------------
# Dataset Label Resolvers & Audio Model Inference
# ---------------------------------------------------------------------------

def collect_asvspoof_eval_labeled(data_root: Path, max_files: int = 200) -> List[Tuple[Path, int]]:
    """Collect balanced bonafide(0)/spoof(1) pairs from ASVspoof 2019 LA eval split."""
    root = resolve_asvspoof_root(data_root)
    proto = root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"
    flac_dir = root / "ASVspoof2019_LA_eval" / "flac"
    if not proto.exists() or not flac_dir.exists():
        return []

    file_index = {p.stem: p for p in flac_dir.iterdir() if p.is_file() and p.suffix.lower() in {".flac", ".wav"}}
    real, fake = [], []
    with open(proto, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            p = file_index.get(parts[1])
            if p is None:
                continue
            lbl = 0 if parts[4].lower() == "bonafide" else 1
            (real if lbl == 0 else fake).append((p, lbl))

    half = max_files // 2
    selected = real[:half] + fake[: max_files - min(half, len(real))]
    return selected


def collect_folder_labeled(root_dir: Path, exts: set, max_files: int = 200) -> List[Tuple[Path, int]]:
    """Collect labeled media files from real/bonafide (0) and fake/spoof (1) subfolders."""
    if not root_dir.exists():
        return []
    real, fake = [], []
    for p in sorted(root_dir.rglob("*")):
        if not (p.is_file() and p.suffix.lower() in exts):
            continue
        parts = {part.lower() for part in p.relative_to(root_dir).parts[:-1]}
        if parts & {"real", "bonafide", "human", "genuine"}:
            real.append((p, 0))
        elif parts & {"fake", "spoof", "synthetic", "ai"}:
            fake.append((p, 1))
    half = max_files // 2
    return real[:half] + fake[: max_files - min(half, len(real))]


def load_audio_model(ckpt_path: Path, cfg: dict, device: torch.device):
    """Load AudioV2Detector with saved adapter weights."""
    from src.models.v2.audio_model import AudioV2Detector
    backbone = cfg["models_v2"]["audio"]["backbone"]
    model = AudioV2Detector(
        backbone_name=backbone,
        freeze_backbone=True,
        embedding_dim=cfg["models_v2"]["embedding_dim"],
        dropout=0.0,
    ).to(device)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        state = ckpt.get("state_dict", ckpt)
        missing, _ = model.load_state_dict(state, strict=False)
        logger.info(f"Checkpoint loaded from {ckpt_path} | Missing frozen keys: {len(missing)}")
    else:
        logger.warning(f"Checkpoint not found: {ckpt_path} — using untrained head weights.")
    model.eval()
    return model


@torch.no_grad()
def predict_audio_waveform(
    model,
    waveform: np.ndarray,
    sample_rate: int = 16000,
    max_duration: float = 5.0,
    device: torch.device = torch.device("cpu"),
) -> Optional[float]:
    try:
        max_samp = int(max_duration * sample_rate)
        wav = torch.from_numpy(waveform.astype(np.float32))
        if wav.shape[0] > max_samp:
            wav = wav[:max_samp]
        elif wav.shape[0] < max_samp:
            wav = torch.nn.functional.pad(wav, (0, max_samp - wav.shape[0]))
        logits, _ = model(wav.unsqueeze(0).to(device))
        return float(torch.sigmoid(logits).item())
    except Exception as exc:
        logger.debug("predict_audio_waveform failed: %s", exc)
        return None


@torch.no_grad()
def predict_audio_file(
    model,
    file_path: Path,
    sample_rate: int = 16000,
    max_duration: float = 5.0,
    device: torch.device = torch.device("cpu"),
) -> Optional[float]:
    wav_t, ok = load_audio_tensor(file_path, sample_rate=sample_rate, max_samples=int(max_duration * sample_rate))
    if not ok:
        return None
    logits, _ = model(wav_t.unsqueeze(0).to(device))
    return float(torch.sigmoid(logits).item())


def compute_binary_metrics(labels: List[int], probs: List[float]) -> Dict:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    if not probs:
        return {"n_files": 0, "acc": None, "f1": None, "auc": None}
    preds = [1 if p >= 0.5 else 0 for p in probs]
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else None
    return {
        "n_files": len(probs),
        "acc": round(float(accuracy_score(labels, preds)), 4),
        "f1": round(float(f1_score(labels, preds, zero_division=0)), 4),
        "auc": round(float(auc), 4) if auc is not None else None,
    }


# ---------------------------------------------------------------------------
# Experiment 3 Sub-Tests
# ---------------------------------------------------------------------------

def run_audio_noise_stress(
    model,
    labeled_files: List[Tuple[Path, int]],
    snr_levels: List[float],
    device: torch.device,
    sample_rate: int = 16000,
) -> Dict:
    """Test audio model robustness to additive Gaussian noise at SNR 20/10/5 dB."""
    if not labeled_files:
        return {"skipped": "No labeled ASVspoof eval files found."}

    logger.info(f"Audio noise stress: {len(labeled_files)} files × {len(snr_levels)} SNR levels")
    max_samp = int(5.0 * sample_rate)
    results = {}
    for snr in snr_levels:
        probs, labels = [], []
        for fpath, lbl in labeled_files:
            wav_t, ok = load_audio_tensor(fpath, sample_rate=sample_rate, max_samples=max_samp)
            if not ok:
                continue
            noisy = add_gaussian_noise_snr(wav_t.numpy(), snr_db=snr)
            p = predict_audio_waveform(model, noisy, sample_rate=sample_rate, device=device)
            if p is not None:
                probs.append(p)
                labels.append(lbl)
        m = compute_binary_metrics(labels, probs)
        results[f"snr_{int(snr)}dB"] = m
        logger.info(f"  SNR {snr:4.0f} dB | n={m['n_files']} acc={m['acc']} f1={m['f1']} auc={m['auc']}")
    return results


def run_h264_stress(
    model,
    video_dir: Path,
    crf_levels: List[int],
    device: torch.device,
    sample_rate: int = 16000,
    max_files: int = 50,
) -> Dict:
    """Test audio/AV degradation across H.264 CRF 23, 32, 40."""
    from src.models.av_sync import compute_av_sync_score
    labeled = collect_folder_labeled(video_dir, VIDEO_EXTS, max_files=max_files)
    if not labeled:
        raw_videos = [f for f in sorted(video_dir.rglob("*")) if f.suffix.lower() in VIDEO_EXTS][:max_files]
        labeled = [(v, 1) for v in raw_videos]

    if not labeled:
        return {"skipped": "No video files found in video_ood directory."}

    logger.info(f"H.264 stress: {len(labeled)} videos × {len(crf_levels)} CRF levels")
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for crf in crf_levels:
            probs, labels, av_scores = [], [], []
            for fpath, lbl in labeled:
                compressed_path = tmp_path / f"crf{crf}_{fpath.stem}.mp4"
                if not compress_video_h264(fpath, compressed_path, crf=crf):
                    continue
                p = predict_audio_file(model, compressed_path, sample_rate=sample_rate, device=device)
                if p is not None:
                    probs.append(p)
                    labels.append(lbl)
                av_res = compute_av_sync_score(compressed_path)
                if av_res.av_sync_score is not None:
                    av_scores.append(av_res.av_sync_score)
            m = compute_binary_metrics(labels, probs)
            m["mean_av_sync"] = round(float(np.mean(av_scores)), 4) if av_scores else None
            results[f"crf_{crf}"] = m
            logger.info(f"  CRF {crf} | n={m['n_files']} acc={m['acc']} auc={m['auc']} av_sync={m['mean_av_sync']}")
    return results


def run_jpeg_stress(
    video_dir: Path,
    qualities: List[int],
    max_files: int = 30,
) -> Dict:
    """Test AV-sync & face-detection retention under JPEG quality 90, 60, 30."""
    from src.models.av_sync import compute_av_sync_score
    videos = [f for f in sorted(video_dir.rglob("*")) if f.suffix.lower() in VIDEO_EXTS][:max_files]
    if not videos:
        return {"skipped": "No video files found for JPEG compression test."}

    logger.info(f"JPEG stress: {len(videos)} videos × qualities {qualities}")
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for q in qualities:
            av_scores, face_frames_list = [], []
            for idx, vpath in enumerate(videos):
                audio_wav = tmp_path / f"orig_{idx}.wav"
                degraded_vid = tmp_path / f"jpeg{q}_{idx}.mp4"
                if not extract_wav_from_video(vpath, audio_wav):
                    continue
                if not reencode_video_jpeg_frames(vpath, degraded_vid, quality=q):
                    continue
                res = compute_av_sync_score(degraded_vid, audio_path=audio_wav)
                if res.av_sync_score is not None:
                    av_scores.append(res.av_sync_score)
                    face_frames_list.append(res.n_frames_analyzed)
            results[f"jpeg_q{q}"] = {
                "n_clips": len(av_scores),
                "mean_av_sync": round(float(np.mean(av_scores)), 4) if av_scores else None,
                "mean_face_frames": round(float(np.mean(face_frames_list)), 2) if face_frames_list else None,
            }
            logger.info(f"  JPEG Q={q} | clips={len(av_scores)} av_sync={results[f'jpeg_q{q}']['mean_av_sync']}")
    return results


def run_av_sync_conflict_test(video_dir: Path, max_pairs: int = 30) -> Dict:
    """Test AV-sync score on aligned vs artificially swapped-audio pairs."""
    from src.models.av_sync import compute_av_sync_score
    videos = [f for f in sorted(video_dir.rglob("*")) if f.suffix.lower() in VIDEO_EXTS][: max_pairs * 2]
    if len(videos) < 2:
        return {"skipped": "Need at least 2 video files for AV swap test."}

    aligned_scores, swapped_scores = [], []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pairs = list(zip(videos[:max_pairs], videos[1: max_pairs + 1]))
        for i, (vid_a, vid_b) in enumerate(pairs):
            res_aligned = compute_av_sync_score(vid_a)
            if res_aligned.av_sync_score is not None:
                aligned_scores.append(res_aligned.av_sync_score)

            audio_b = tmp_path / f"audio_b_{i}.wav"
            if extract_wav_from_video(vid_b, audio_b):
                res_swapped = compute_av_sync_score(vid_a, audio_path=audio_b)
                if res_swapped.av_sync_score is not None:
                    swapped_scores.append(res_swapped.av_sync_score)

    result = {
        "n_aligned_pairs": len(aligned_scores),
        "n_swapped_pairs": len(swapped_scores),
        "aligned_mean_score": round(float(np.mean(aligned_scores)), 4) if aligned_scores else None,
        "aligned_std": round(float(np.std(aligned_scores)), 4) if aligned_scores else None,
        "swapped_mean_score": round(float(np.mean(swapped_scores)), 4) if swapped_scores else None,
        "swapped_std": round(float(np.std(swapped_scores)), 4) if swapped_scores else None,
        "contradiction_detection_rate": (
            round(sum(1 for s in swapped_scores if s < 0.40) / len(swapped_scores), 4)
            if swapped_scores else None
        ),
    }
    logger.info(
        f"AV sync | aligned={result['aligned_mean_score']} | "
        f"swapped={result['swapped_mean_score']} | "
        f"contradiction_rate={result['contradiction_detection_rate']}"
    )
    return result


def run_cross_dataset_audio(
    model,
    indomain_files: List[Tuple[Path, int]],
    ood_dir: Path,
    device: torch.device,
    max_files: int = 200,
) -> Dict:
    """Evaluate audio model on in-domain (ASVspoof) vs OOD (WaveFake / Deepfake-Eval-2024)."""
    logger.info("=== Cross-Dataset Audio Generalization ===")
    in_probs, in_labels = [], []
    for p, lbl in indomain_files:
        prob = predict_audio_file(model, p, device=device)
        if prob is not None:
            in_probs.append(prob)
            in_labels.append(lbl)
    indomain_metrics = compute_binary_metrics(in_labels, in_probs)
    logger.info(f"  In-domain (ASVspoof): {indomain_metrics}")

    ood_files = collect_folder_labeled(ood_dir, AUDIO_EXTS, max_files=max_files)
    ood_probs, ood_labels = [], []
    for p, lbl in ood_files:
        prob = predict_audio_file(model, p, device=device)
        if prob is not None:
            ood_probs.append(prob)
            ood_labels.append(lbl)
    ood_metrics = compute_binary_metrics(ood_labels, ood_probs)
    logger.info(f"  OOD ({ood_dir}): {ood_metrics}")

    return {
        "indomain_ASVspoof_eval": indomain_metrics,
        "ood_WaveFake_or_DeepfakeEval": ood_metrics,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_compression_curves(results: Dict, out_path: Path):
    """Plot degradation across H.264 CRF, JPEG quality, and Audio SNR."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        fig.suptitle("VeriSight v2 — Experiment 3: Compression & Noise Robustness", fontsize=13, fontweight="bold")

        ax = axes[0]
        h264 = results.get("h264_compression", {})
        crfs, h_vals = [], []
        for k, v in sorted(h264.items()):
            if k.startswith("crf_") and isinstance(v, dict):
                crfs.append(int(k.split("_")[1]))
                h_vals.append((v.get("auc") or v.get("acc") or 0))
        if crfs:
            ax.plot(crfs, h_vals, "b-o", linewidth=2, markersize=7)
            ax.set_xlabel("H.264 CRF (higher = heavier)")
            ax.set_ylabel("AUC / Accuracy")
            ax.set_title("H.264 Compression")
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No H.264 video data", ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_title("H.264 Compression (Skipped)")

        ax = axes[1]
        jpeg = results.get("jpeg_compression", {})
        qs, j_vals = [], []
        for k, v in sorted(jpeg.items(), key=lambda x: -int(x[0].split("_q")[1]) if "_q" in x[0] else 0):
            if k.startswith("jpeg_q") and isinstance(v, dict):
                qs.append(int(k.split("_q")[1]))
                j_vals.append(v.get("mean_av_sync") or 0)
        if qs:
            ax.plot(qs, j_vals, "g-^", linewidth=2, markersize=7)
            ax.invert_xaxis()
            ax.set_xlabel("JPEG Quality (lower = heavier)")
            ax.set_ylabel("Mean AV-Sync Score")
            ax.set_title("JPEG Frame Compression")
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No JPEG video data", ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_title("JPEG Compression (Skipped)")

        ax = axes[2]
        noise = results.get("audio_noise_stress", {})
        snrs, n_vals = [], []
        for k, v in sorted(noise.items(), key=lambda x: -float(x[0].split("_")[1].replace("dB", "")) if k.startswith("snr_") else 0):
            if k.startswith("snr_") and isinstance(v, dict):
                snrs.append(float(k.split("_")[1].replace("dB", "")))
                n_vals.append(v.get("auc") or v.get("acc") or 0)
        if snrs:
            ax.plot(snrs, n_vals, "r-s", linewidth=2, markersize=7)
            ax.invert_xaxis()
            ax.set_xlabel("Audio SNR (dB, lower = noisier)")
            ax.set_ylabel("ROC-AUC / Accuracy")
            ax.set_title("Gaussian Audio Noise")
            ax.set_ylim(0, 1.05)
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No noise data", ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_title("Audio Noise (Skipped)")

        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Compression curves saved -> {out_path}")
    except Exception as exc:
        logger.warning(f"Plotting failed: {exc}")


def plot_cross_dataset(results: Dict, out_path: Path):
    """Bar chart comparing in-domain vs OOD performance."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cross = results.get("cross_dataset", {})
        indomain = cross.get("indomain_ASVspoof_eval", {})
        ood = cross.get("ood_WaveFake_or_DeepfakeEval", {})

        metrics = ["acc", "f1", "auc"]
        labels = ["Accuracy", "F1 Score", "ROC-AUC"]
        indomain_vals = [indomain.get(m) or 0 for m in metrics]
        ood_vals = [ood.get(m) or 0 for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        bars1 = ax.bar(x - width / 2, indomain_vals, width, label="In-Domain (ASVspoof 2019 LA)", color="steelblue")
        bars2 = ax.bar(x + width / 2, ood_vals, width, label="OOD (WaveFake / Deepfake-Eval-2024)", color="salmon")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.set_title("VeriSight v2 — Exp 3: Cross-Dataset Audio Generalization")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.08)
        ax.legend()
        ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=9)
        ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Cross-dataset plot saved -> {out_path}")
    except Exception as exc:
        logger.warning(f"Cross-dataset plotting failed: {exc}")


def main():
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/exp3_robustness.log", mode="a", encoding="utf-8"),
        ],
    )

    parser = argparse.ArgumentParser(description="VeriSight v2 — Experiment 3 Robustness Suite")
    parser.add_argument("--config", default="configs/v2_cuda.yaml")
    parser.add_argument("--data_root", default="Dataset_L2/ASVspoof2019_LA")
    parser.add_argument("--ood_dir", default="Dataset_L2/audio_ood")
    parser.add_argument("--video_dir", default="Dataset_L2/video_ood")
    parser.add_argument("--max_files", type=int, default=200)
    parser.add_argument("--skip_h264", action="store_true")
    parser.add_argument("--skip_jpeg", action="store_true")
    parser.add_argument("--skip_avsync", action="store_true")
    args = parser.parse_args()

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | Experiment 3 — Robustness & Compression Stress")

    ckpt_path = Path(cfg["models_v2"]["audio"]["checkpoint"])
    model = load_audio_model(ckpt_path, cfg, device)
    sample_rate = cfg["models_v2"]["audio"]["sample_rate"]

    indomain_labeled = collect_asvspoof_eval_labeled(Path(args.data_root), max_files=args.max_files)
    t_total = time.time()

    results: Dict = {
        "config": args.config,
        "checkpoint": str(ckpt_path),
        "device": str(device),
    }

    logger.info("=== 1/5: Audio Gaussian Noise Stress (SNR 20/10/5 dB) ===")
    results["audio_noise_stress"] = run_audio_noise_stress(
        model, indomain_labeled, snr_levels=[20.0, 10.0, 5.0], device=device, sample_rate=sample_rate
    )

    logger.info("=== 2/5: H.264 Compression Stress (CRF 23/32/40) ===")
    results["h264_compression"] = (
        {"skipped": True}
        if args.skip_h264
        else run_h264_stress(model, Path(args.video_dir), [23, 32, 40], device, sample_rate, min(args.max_files, 50))
    )

    logger.info("=== 3/5: JPEG Frame Compression Stress (Q=90/60/30) ===")
    results["jpeg_compression"] = (
        {"skipped": True}
        if args.skip_jpeg
        else run_jpeg_stress(Path(args.video_dir), [90, 60, 30], max_files=min(args.max_files, 30))
    )

    logger.info("=== 4/5: Cross-Dataset Audio Generalization ===")
    results["cross_dataset"] = run_cross_dataset_audio(
        model, indomain_labeled, Path(args.ood_dir), device=device, max_files=args.max_files
    )

    logger.info("=== 5/5: AV Sync Contradiction Test (aligned vs swapped audio) ===")
    av_sync_res = (
        {"skipped": True}
        if args.skip_avsync
        else run_av_sync_conflict_test(Path(args.video_dir), max_pairs=30)
    )
    results["av_sync_contradiction"] = av_sync_res

    results["total_time_seconds"] = round(time.time() - t_total, 1)

    out_json = Path("results/exp3_robustness.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved -> {out_json}")

    av_json = Path("results/av_sync_eval_l2.json")
    with open(av_json, "w", encoding="utf-8") as f:
        json.dump(av_sync_res, f, indent=2)
    logger.info(f"AV-sync results saved -> {av_json}")

    plot_compression_curves(results, Path("results/plots/exp3_compression_curves.png"))
    plot_cross_dataset(results, Path("results/plots/exp3_audio_cross_dataset.png"))
    logger.info("=== Experiment 3 DONE! ===")


if __name__ == "__main__":
    main()
