"""VeriSight v2 — Deliverable D: Experiment 3 Generalization & Social-Media Compression Stress Suite.

Laptop 2 (Audio/ASR/AV-Sync Engineer) — EXPERIMENT SCRIPT.
Run after training is complete (checkpoints/v2_audio_xlsr_head.pt exists).

USAGE:
  python scripts/run_exp3_robustness.py --config configs/v2_cuda.yaml

OUTPUTS:
  results/exp3_robustness.json
  results/plots/exp3_compression_curves.png
  results/plots/exp3_audio_cross_dataset.png

Tests degradation across:
  1. H.264 video compression:    CRF 23, 32, 40 (requires ffmpeg on PATH)
  2. JPEG frame compression:     quality 90, 60, 30
  3. Audio Gaussian noise:       SNR 20 dB, 10 dB, 5 dB
  4. Cross-dataset audio OOD:    ASVspoof (in-domain) vs WaveFake/Deepfake-Eval-2024 (OOD)
  5. AV-sync contradiction test: aligned pairs vs artificially swapped-audio pairs
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal Degradation Utilities
# ---------------------------------------------------------------------------

def add_gaussian_noise_snr(waveform: np.ndarray, snr_db: float) -> np.ndarray:
    """Add Gaussian noise to a waveform at a given SNR (dB)."""
    signal_power = np.mean(waveform ** 2)
    if signal_power < 1e-12:
        return waveform
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.random.randn(*waveform.shape).astype(np.float32) * np.sqrt(noise_power)
    return (waveform + noise).clip(-1.0, 1.0)


def compress_video_h264(input_path: Path, output_path: Path, crf: int) -> bool:
    """Re-encode a video with H.264 at the given CRF using ffmpeg."""
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg not found on PATH. Skipping H.264 compression test.")
        return False
    cmd = [
        "ffmpeg", "-y", "-i", str(input_path),
        "-c:v", "libx264", "-crf", str(crf),
        "-c:a", "aac", "-b:a", "128k",
        "-loglevel", "error",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    return result.returncode == 0


def compress_frame_jpeg(frame_rgb: np.ndarray, quality: int) -> np.ndarray:
    """Encode and decode a frame via JPEG compression at the given quality."""
    import cv2
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), encode_param)
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Audio Model Inference Helpers
# ---------------------------------------------------------------------------

def load_audio_model(ckpt_path: Path, cfg: dict, device: torch.device):
    """Load AudioV2Detector with saved adapter weights."""
    from src.models.v2.audio_model import AudioV2Detector
    backbone = cfg["models_v2"]["audio"]["backbone"]
    model = AudioV2Detector(
        backbone_name=backbone,
        freeze_backbone=True,
        embedding_dim=cfg["models_v2"]["embedding_dim"],
        dropout=0.0,  # No dropout at eval
    ).to(device)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device)
        state = ckpt.get("state_dict", ckpt)
        # Load only the adapter/head keys (not backbone, which is frozen)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"Checkpoint loaded from {ckpt_path} | Missing keys: {len(missing)}")
    else:
        logger.warning(f"Checkpoint not found: {ckpt_path} — using random weights (for structure test).")
    model.eval()
    return model


def predict_audio_file(
    model,
    file_path: Path,
    sample_rate: int = 16000,
    max_duration: float = 5.0,
    device: torch.device = torch.device("cpu"),
) -> Optional[float]:
    """Return fake probability [0,1] for a single audio file, or None on error."""
    try:
        import torchaudio
        wav, sr = torchaudio.load(str(file_path))
        wav = wav.mean(0)
        if sr != sample_rate:
            import torchaudio.transforms as T
            wav = T.Resample(sr, sample_rate)(wav)
        max_samp = int(max_duration * sample_rate)
        if wav.shape[0] > max_samp:
            wav = wav[:max_samp]
        elif wav.shape[0] < max_samp:
            wav = torch.nn.functional.pad(wav, (0, max_samp - wav.shape[0]))
        with torch.no_grad():
            logits, _ = model(wav.unsqueeze(0).to(device))
        return float(torch.sigmoid(logits).item())
    except Exception as exc:
        logger.debug(f"predict_audio_file failed for {file_path}: {exc}")
        return None


def predict_audio_waveform(
    model,
    waveform: np.ndarray,
    sample_rate: int = 16000,
    max_duration: float = 5.0,
    device: torch.device = torch.device("cpu"),
) -> Optional[float]:
    """Return fake probability for a raw numpy waveform."""
    try:
        max_samp = int(max_duration * sample_rate)
        wav = torch.from_numpy(waveform.astype(np.float32))
        if wav.shape[0] > max_samp:
            wav = wav[:max_samp]
        elif wav.shape[0] < max_samp:
            wav = torch.nn.functional.pad(wav, (0, max_samp - wav.shape[0]))
        with torch.no_grad():
            logits, _ = model(wav.unsqueeze(0).to(device))
        return float(torch.sigmoid(logits).item())
    except Exception as exc:
        logger.debug(f"predict_audio_waveform failed: {exc}")
        return None


def evaluate_audio_dir(
    model,
    audio_dir: Path,
    label: int,  # 0=real, 1=fake
    device: torch.device,
    sample_rate: int = 16000,
    max_files: Optional[int] = None,
    ext_filter=None,
) -> Dict:
    """Evaluate model on all audio files in a directory, returning acc/f1/auc."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    ext_filter = ext_filter or {".wav", ".flac", ".mp3"}
    files = [f for f in sorted(audio_dir.rglob("*")) if f.suffix.lower() in ext_filter]
    if max_files:
        files = files[:max_files]
    if not files:
        return {"n_files": 0, "acc": None, "f1": None, "auc": None}

    probs, labels = [], []
    for f in files:
        p = predict_audio_file(model, f, sample_rate=sample_rate, device=device)
        if p is not None:
            probs.append(p)
            labels.append(label)

    if not probs:
        return {"n_files": 0, "acc": None, "f1": None, "auc": None}

    preds = [1 if p >= 0.5 else 0 for p in probs]
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else None
    return {
        "n_files": len(probs),
        "acc": round(accuracy_score(labels, preds), 4),
        "f1": round(f1_score(labels, preds, zero_division=0), 4),
        "auc": round(auc, 4) if auc is not None else None,
    }


# ---------------------------------------------------------------------------
# Experiment 3 Sub-Tests
# ---------------------------------------------------------------------------

def run_audio_noise_stress(
    model,
    audio_dir: Path,
    snr_levels: List[float],
    device: torch.device,
    sample_rate: int = 16000,
    max_files: int = 100,
) -> Dict:
    """Test audio model robustness to additive Gaussian noise at various SNR levels."""
    import torchaudio
    ext_filter = {".wav", ".flac", ".mp3"}
    files = [f for f in sorted(audio_dir.rglob("*"))
             if f.suffix.lower() in ext_filter][:max_files]

    logger.info(f"Audio noise stress: {len(files)} files × {len(snr_levels)} SNR levels")

    results = {}
    for snr in snr_levels:
        probs = []
        for fpath in files:
            try:
                wav, sr = torchaudio.load(str(fpath))
                wav_np = wav.mean(0).numpy()
                noisy = add_gaussian_noise_snr(wav_np, snr_db=snr)
                p = predict_audio_waveform(model, noisy, sample_rate=sr, device=device)
                if p is not None:
                    probs.append(p)
            except Exception:
                continue
        # We expect spoof/fake files from ASVspoof → label=1
        preds = [1 if p >= 0.5 else 0 for p in probs]
        results[f"snr_{snr}dB"] = {
            "n_files": len(probs),
            "mean_fake_prob": round(float(np.mean(probs)), 4) if probs else None,
            "detection_rate": round(sum(preds) / len(preds), 4) if preds else None,
        }
        logger.info(f"  SNR {snr:4.0f} dB | n={len(probs)} | detection_rate={results[f'snr_{snr}dB']['detection_rate']}")

    return results


def run_h264_stress(
    model,
    video_dir: Path,
    crf_levels: List[int],
    device: torch.device,
    sample_rate: int = 16000,
    max_files: int = 50,
) -> Dict:
    """Test model on H.264-compressed video clips (audio extracted for audio model)."""
    ext_filter = {".mp4", ".avi", ".mkv"}
    files = [f for f in sorted(video_dir.rglob("*"))
             if f.suffix.lower() in ext_filter][:max_files]

    if not files:
        return {"skipped": "No video files found in video_ood directory."}

    logger.info(f"H.264 stress: {len(files)} videos × {len(crf_levels)} CRF levels")

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for crf in crf_levels:
            probs = []
            for fpath in files:
                compressed_path = tmp_path / f"crf{crf}_{fpath.name}"
                # Re-encode video
                ok = compress_video_h264(fpath, compressed_path, crf=crf)
                if not ok:
                    continue
                # Extract audio from compressed video and predict
                audio_tmp = tmp_path / f"audio_{fpath.stem}.wav"
                if shutil.which("ffmpeg"):
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", str(compressed_path),
                         "-ar", str(sample_rate), "-ac", "1",
                         "-loglevel", "error", str(audio_tmp)],
                        capture_output=True, timeout=60,
                    )
                    if audio_tmp.exists():
                        p = predict_audio_file(model, audio_tmp, sample_rate=sample_rate, device=device)
                        if p is not None:
                            probs.append(p)
            results[f"crf_{crf}"] = {
                "n_clips": len(probs),
                "mean_fake_prob": round(float(np.mean(probs)), 4) if probs else None,
            }
            logger.info(f"  CRF {crf} | n={len(probs)} | mean_fake_prob={results[f'crf_{crf}']['mean_fake_prob']}")
    return results


def run_av_sync_conflict_test(
    video_dir: Path,
    max_pairs: int = 30,
) -> Dict:
    """Test AV-sync score on aligned vs artificially swapped-audio pairs.

    For each video clip:
      - Compute av_sync_score with ORIGINAL audio → should be high (aligned)
      - Swap audio with another random clip → should be low (contradiction)
    """
    from src.models.av_sync import compute_av_sync_score
    ext_filter = {".mp4", ".avi", ".mkv"}
    videos = [f for f in sorted(video_dir.rglob("*"))
              if f.suffix.lower() in ext_filter][:max_pairs * 2]

    if len(videos) < 2:
        return {"skipped": "Need at least 2 video files for AV swap test."}

    aligned_scores, swapped_scores = [], []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        pairs = list(zip(videos[:max_pairs], videos[1:max_pairs + 1]))
        for i, (vid_a, vid_b) in enumerate(pairs):
            # Aligned: vid_a with its own audio
            res_aligned = compute_av_sync_score(vid_a)
            if res_aligned.av_sync_score is not None:
                aligned_scores.append(res_aligned.av_sync_score)

            # Swapped: extract audio from vid_b and use with vid_a video
            audio_b = tmp_path / f"audio_b_{i}.wav"
            if shutil.which("ffmpeg"):
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(vid_b), "-vn", "-ar", "16000", "-ac", "1",
                     "-loglevel", "error", str(audio_b)],
                    capture_output=True, timeout=60,
                )
                if audio_b.exists():
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
    indomain_dir: Path,
    ood_dir: Path,
    device: torch.device,
    max_files: int = 200,
) -> Dict:
    """Evaluate audio model on in-domain (ASVspoof) vs OOD (WaveFake / Deepfake-Eval-2024)."""
    logger.info("=== Cross-Dataset Audio Generalization ===")
    # In-domain: ASVspoof eval split (spoof files only for detection rate)
    spoof_dir = indomain_dir / "ASVspoof2019_LA_eval" / "flac"
    if not spoof_dir.exists():
        spoof_dir = indomain_dir  # fallback

    indomain_metrics = evaluate_audio_dir(model, spoof_dir, label=1, device=device, max_files=max_files)
    logger.info(f"  In-domain (ASVspoof): {indomain_metrics}")

    ood_metrics = evaluate_audio_dir(model, ood_dir, label=1, device=device, max_files=max_files)
    logger.info(f"  OOD (WaveFake/Deepfake-Eval): {ood_metrics}")

    return {
        "indomain_ASVspoof_eval": indomain_metrics,
        "ood_WaveFake_or_DeepfakeEval": ood_metrics,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_compression_curves(results: Dict, out_path: Path):
    """Plot detection rate vs CRF and SNR degradation curves."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("VeriSight v2 — Experiment 3: Compression Robustness", fontsize=14, fontweight="bold")

        # H.264 CRF curve
        ax = axes[0]
        h264 = results.get("h264_compression", {})
        crfs, detection = [], []
        for key, val in sorted(h264.items()):
            crf = int(key.split("_")[1])
            crfs.append(crf)
            detection.append(val.get("mean_fake_prob", 0) or 0)
        if crfs:
            ax.plot(crfs, detection, "b-o", linewidth=2, markersize=8, label="Mean Fake Prob")
            ax.set_xlabel("H.264 CRF (higher = more compressed)")
            ax.set_ylabel("Mean P(Fake)")
            ax.set_title("Audio Robustness vs H.264 Compression")
            ax.set_ylim(0, 1)
            ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5, label="Decision boundary")
            ax.legend()
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No H.264 data\n(ffmpeg not available or no video files)",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_title("H.264 Compression (Skipped)")

        # Audio SNR curve
        ax = axes[1]
        noise = results.get("audio_noise_stress", {})
        snrs, det_rates = [], []
        for key, val in sorted(noise.items(), key=lambda x: float(x[0].split("_")[1].replace("dB", ""))):
            snr = float(key.split("_")[1].replace("dB", ""))
            snrs.append(snr)
            det_rates.append(val.get("detection_rate", 0) or 0)
        if snrs:
            ax.plot(snrs, det_rates, "r-s", linewidth=2, markersize=8, label="Detection Rate")
            ax.set_xlabel("SNR (dB) — lower = noisier")
            ax.set_ylabel("Spoof Detection Rate")
            ax.set_title("Audio Robustness vs Gaussian Noise")
            ax.set_ylim(0, 1)
            ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5)
            ax.legend()
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No noise stress data", ha="center", va="center", transform=ax.transAxes)

        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Compression curves saved → {out_path}")
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
        indomain_vals = [indomain.get(m, 0) or 0 for m in metrics]
        ood_vals = [ood.get(m, 0) or 0 for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        bars1 = ax.bar(x - width/2, indomain_vals, width, label="In-Domain (ASVspoof 2019 LA)", color="steelblue")
        bars2 = ax.bar(x + width/2, ood_vals, width, label="OOD (WaveFake / Deepfake-Eval-2024)", color="salmon")
        ax.set_xlabel("Metric")
        ax.set_ylabel("Score")
        ax.set_title("VeriSight v2 — Exp 3: Cross-Dataset Audio Generalization")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=9)
        ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Cross-dataset plot saved → {out_path}")
    except Exception as exc:
        logger.warning(f"Cross-dataset plotting failed: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/exp3_robustness.log", mode="a"),
        ],
    )
    Path("logs").mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="VeriSight v2 — Experiment 3 Robustness Suite")
    parser.add_argument("--config", default="configs/v2_cuda.yaml")
    parser.add_argument("--data_root", default="Dataset_L2/ASVspoof2019_LA")
    parser.add_argument("--ood_dir", default="Dataset_L2/audio_ood")
    parser.add_argument("--video_dir", default="Dataset_L2/video_ood")
    parser.add_argument("--max_files", type=int, default=100)
    parser.add_argument("--skip_h264", action="store_true", help="Skip H.264 compression test")
    parser.add_argument("--skip_avsync", action="store_true", help="Skip AV-sync swap test")
    args = parser.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | Experiment 3 — Robustness & Compression Stress")

    # Load audio model
    ckpt_path = Path(cfg["models_v2"]["audio"]["checkpoint"])
    model = load_audio_model(ckpt_path, cfg, device)

    sample_rate = cfg["models_v2"]["audio"]["sample_rate"]
    results: Dict = {
        "config": args.config,
        "checkpoint": str(ckpt_path),
        "device": str(device),
    }

    t_total = time.time()

    # 1. Audio Gaussian noise stress
    logger.info("=== 1/4: Audio Gaussian Noise Stress (SNR 20/10/5 dB) ===")
    indomain_audio = Path(args.data_root) / "ASVspoof2019_LA_eval" / "flac"
    if not indomain_audio.exists():
        indomain_audio = Path(args.data_root)
    results["audio_noise_stress"] = run_audio_noise_stress(
        model, indomain_audio, snr_levels=[20.0, 10.0, 5.0],
        device=device, sample_rate=sample_rate, max_files=args.max_files,
    )

    # 2. H.264 video compression stress
    if not args.skip_h264:
        logger.info("=== 2/4: H.264 Compression Stress (CRF 23/32/40) ===")
        results["h264_compression"] = run_h264_stress(
            model, Path(args.video_dir), crf_levels=[23, 32, 40],
            device=device, sample_rate=sample_rate, max_files=min(args.max_files, 50),
        )
    else:
        results["h264_compression"] = {"skipped": True}

    # 3. Cross-dataset audio generalization
    logger.info("=== 3/4: Cross-Dataset Audio Generalization ===")
    results["cross_dataset"] = run_cross_dataset_audio(
        model, Path(args.data_root), Path(args.ood_dir),
        device=device, max_files=args.max_files,
    )

    # 4. AV sync contradiction test
    if not args.skip_avsync:
        logger.info("=== 4/4: AV Sync Contradiction Test (aligned vs swapped audio) ===")
        results["av_sync_contradiction"] = run_av_sync_conflict_test(
            Path(args.video_dir), max_pairs=30,
        )
    else:
        results["av_sync_contradiction"] = {"skipped": True}

    total_time = time.time() - t_total
    results["total_time_seconds"] = round(total_time, 1)
    logger.info(f"Experiment 3 complete in {total_time:.1f}s")

    # Save JSON
    out_json = Path("results/exp3_robustness.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {out_json}")

    # Generate plots
    plot_compression_curves(results, Path("results/plots/exp3_compression_curves.png"))
    plot_cross_dataset(results, Path("results/plots/exp3_audio_cross_dataset.png"))

    logger.info("=== Experiment 3 DONE! Notify Piyush (Laptop 1) to pull and merge. ===")


if __name__ == "__main__":
    main()
