#!/usr/bin/env python3
"""
VeriSight CUDA Training Pipeline
---------------------------------
Trains the Video (Celeb-DF v2), Audio (Deepfake-Audio), and Text (HC3 + Fakeddit)
models using CUDA GPU when available.

Usage:
    python scripts/train_all.py --config configs/cuda.yaml
    python scripts/train_all.py --config configs/cuda.yaml --stage video
    python scripts/train_all.py --config configs/cuda.yaml --stage audio
    python scripts/train_all.py --config configs/cuda.yaml --stage text
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.prepare_datasets import main as prepare_datasets_main
from src.config import load_config
from src.training.train_video import train_video_model
from src.training.train_audio import train_audio_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_all")


def check_cuda(requested_device: str) -> str:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        logger.info(f"CUDA GPU detected: {props.name} ({props.total_memory / 1e9:.1f} GB VRAM)")
        return "cuda" if requested_device == "cuda" else requested_device
    logger.warning("No CUDA GPU detected — falling back to CPU.")
    return "cpu"


def train_text_stage(cfg_path: str, device: str, epochs: int = 5):
    """Train a fast TF-IDF + PyTorch MLP text classifier on HC3 + Fakeddit on GPU."""
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import accuracy_score, f1_score

    train_csv = ROOT / "data" / "text_train.csv"
    val_csv = ROOT / "data" / "text_val.csv"
    if not train_csv.exists() or not val_csv.exists():
        logger.warning("text_train.csv / text_val.csv not found; skipping text stage.")
        return

    logger.info("=" * 60)
    logger.info("STAGE 3 — Text AI/Misinformation Classifier (HC3 + Fakeddit)")
    logger.info("=" * 60)

    tr_df = pd.read_csv(train_csv)
    va_df = pd.read_csv(val_csv)

    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_tr = torch.tensor(vec.fit_transform(tr_df["text"].astype(str)).toarray(), dtype=torch.float32)
    y_tr = torch.tensor(tr_df["label"].values, dtype=torch.float32)
    X_va = torch.tensor(vec.transform(va_df["text"].astype(str)).toarray(), dtype=torch.float32)
    y_va = torch.tensor(va_df["label"].values, dtype=torch.float32)

    tr_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
    model = nn.Sequential(
        nn.Linear(X_tr.shape[1], 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, 1),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()

    best_f1 = 0.0
    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(yb)

        model.eval()
        with torch.no_grad():
            logits = model(X_va.to(device)).squeeze(-1)
            preds = (torch.sigmoid(logits).cpu().numpy() >= 0.5).astype(int)
            acc = accuracy_score(y_va.numpy(), preds)
            f1 = f1_score(y_va.numpy(), preds, zero_division=0)
        logger.info(f"Text Epoch {ep:02d}/{epochs:02d} | Loss: {total_loss/len(tr_df):.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")
        if f1 >= best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), ckpt_dir / "text_mlp_best.pt")

    logger.info(f"Saved best text model (Val F1={best_f1:.4f}) to checkpoints/text_mlp_best.pt")


def main():
    parser = argparse.ArgumentParser(description="VeriSight CUDA training pipeline")
    parser.add_argument("--config", default="configs/cuda.yaml", help="YAML config path")
    parser.add_argument("--stage", choices=["video", "audio", "text", "all"], default="all")
    args = parser.parse_args()

    config_path = str(ROOT / args.config)
    cfg = load_config(config_path)
    device = check_cuda(cfg.device)

    # Always ensure data splits are prepared from Dataset/
    if not (ROOT / "data" / "video_train.csv").exists() or not (ROOT / "data" / "audio_train.csv").exists():
        logger.info("Preparing train/val/test CSV splits from Dataset/...")
        prepare_datasets_main()

    stages = ["video", "audio", "text"] if args.stage == "all" else [args.stage]
    total_start = time.perf_counter()

    if "video" in stages:
        t0 = time.perf_counter()
        logger.info("=" * 60)
        logger.info("STAGE 1 — Video Deepfake Detector (EfficientNet-B0 on Celeb-DF v2)")
        logger.info("=" * 60)
        train_video_model(
            config_path=config_path,
            overrides={
                "device": device,
                "paths": {
                    "train_split": str(ROOT / "data" / "video_train.csv"),
                    "val_split": str(ROOT / "data" / "video_val.csv"),
                },
            },
        )
        logger.info(f"Stage 1 (Video) finished in {time.perf_counter() - t0:.2f}s")

    if "audio" in stages:
        t0 = time.perf_counter()
        logger.info("=" * 60)
        logger.info("STAGE 2 — Audio Deepfake Detector (wav2vec2-base on Deepfake-Audio)")
        logger.info("=" * 60)
        train_audio_model(
            config_path=config_path,
            overrides={
                "device": device,
                "paths": {
                    "train_split": str(ROOT / "data" / "audio_train.csv"),
                    "val_split": str(ROOT / "data" / "audio_val.csv"),
                },
            },
        )
        logger.info(f"Stage 2 (Audio) finished in {time.perf_counter() - t0:.2f}s")

    if "text" in stages:
        t0 = time.perf_counter()
        train_text_stage(config_path, device=device, epochs=5)
        logger.info(f"Stage 3 (Text) finished in {time.perf_counter() - t0:.2f}s")

    total_sec = time.perf_counter() - total_start
    logger.info(f"All requested stages finished in {total_sec:.2f}s ({total_sec/60:.2f} min)")
    logger.info("Checkpoints saved in: %s", ROOT / cfg.paths.checkpoints_dir)


if __name__ == "__main__":
    main()
