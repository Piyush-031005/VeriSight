#!/usr/bin/env python3
"""
Laptop 1 Training Script: VeriSight v2 Multilingual Text Branch (XLM-RoBERTa)
Trains on English (HC3 + Fakeddit) + evaluates Hindi/Hinglish generalization.
Saves compact adapter weights (< 10 MB) to checkpoints/v2_text_head.pt
and exports L2-normalized 256-d embeddings to data/cache/text_l1_embeddings.npz.

Usage:
    python scripts/train_v2_text.py --config configs/v2_cuda.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.v2.text_model import TextV2Detector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_v2_text")


class TokenizedTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df["label"].astype(float).tolist()
        self.langs = df["language"].astype(str).tolist() if "language" in df.columns else ["en"] * len(df)
        self.enc = tokenizer(
            self.texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.enc["input_ids"][idx],
            self.enc["attention_mask"][idx],
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def save_adapter_state(model: TextV2Detector, path: Path) -> None:
    state = {
        "token_attn": model.token_attn.state_dict(),
        "adapter": model.adapter.state_dict(),
        "classifier": model.classifier.state_dict(),
        "embedding_dim": model.embedding_dim,
        "backbone_name": model.backbone_name,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main():
    parser = argparse.ArgumentParser(description="Train VeriSight v2 Multilingual Text Branch")
    parser.add_argument("--config", default="configs/v2_cuda.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info(f"CUDA GPU: {props.name} ({props.total_memory / 1e9:.1f} GB VRAM)")

    t_cfg = cfg["models_v2"]["text"]
    epochs = args.epochs or cfg["training_v2"]["epochs"]

    train_csv = ROOT / "data" / "text_v2_train.csv"
    val_csv = ROOT / "data" / "text_v2_val.csv"
    hi_csv = ROOT / "data" / "text_v2_hindi_test.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("Run `python scripts/prepare_datasets_v2.py` first.")

    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizer & XLM-R model: {t_cfg['backbone']}...")
    tokenizer = AutoTokenizer.from_pretrained(t_cfg["backbone"])

    tr_df = pd.read_csv(train_csv)
    va_df = pd.read_csv(val_csv)
    hi_df = pd.read_csv(hi_csv) if hi_csv.exists() else None

    train_ds = TokenizedTextDataset(tr_df, tokenizer, max_length=t_cfg["max_length"])
    val_ds = TokenizedTextDataset(va_df, tokenizer, max_length=t_cfg["max_length"])
    hi_ds = TokenizedTextDataset(hi_df, tokenizer, max_length=t_cfg["max_length"]) if hi_df is not None and len(hi_df) > 0 else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TextV2Detector(
        backbone_name=t_cfg["backbone"],
        freeze_backbone=t_cfg["freeze_backbone"],
        embedding_dim=cfg["models_v2"]["embedding_dim"],
        dropout=cfg["models_v2"]["dropout"],
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable):,} (XLM-R frozen)")

    optimizer = torch.optim.AdamW(trainable, lr=cfg["training_v2"]["learning_rate"], weight_decay=cfg["training_v2"]["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    ckpt_path = ROOT / t_cfg["checkpoint"]
    best_f1 = -1.0
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        model.train()
        tr_loss = 0.0
        for ids, mask, y in train_loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits, _ = model(ids, mask)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item() * len(y)

        model.eval()
        val_loss = 0.0
        probs, labels = [], []
        with torch.no_grad():
            for ids, mask, y in val_loader:
                ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits, _ = model(ids, mask)
                    loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                labels.extend(y.cpu().numpy().astype(int).tolist())

        tr_loss /= max(len(train_ds), 1)
        val_loss /= max(len(val_ds), 1)
        preds = [1 if p >= 0.5 else 0 for p in probs]
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)
        auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else 0.0

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} ({time.perf_counter() - ep_t0:.1f}s) | "
            f"Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {acc:.4f} | Val F1: {f1:.4f} | Val AUC: {auc:.4f}"
        )

        if f1 >= best_f1:
            best_f1 = f1
            save_adapter_state(model, ckpt_path)
            logger.info(f"  -> Saved best text adapter: {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.2f} MB)")

    # Evaluate Hindi/Hinglish generalization (Exp 3 Cross-Lingual)
    eval_summary = {"en_val_f1": round(float(best_f1), 4)}
    if hi_ds is not None:
        hi_loader = DataLoader(hi_ds, batch_size=args.batch_size, shuffle=False)
        hi_probs, hi_labels = [], []
        model.eval()
        with torch.no_grad():
            for ids, mask, y in hi_loader:
                logits, _ = model(ids.to(device), mask.to(device))
                hi_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                hi_labels.extend(y.numpy().astype(int).tolist())
        hi_preds = [1 if p >= 0.5 else 0 for p in hi_probs]
        hi_acc = accuracy_score(hi_labels, hi_preds)
        hi_f1 = f1_score(hi_labels, hi_preds, zero_division=0)
        eval_summary["hindi_hinglish_acc"] = round(float(hi_acc), 4)
        eval_summary["hindi_hinglish_f1"] = round(float(hi_f1), 4)
        logger.info(f"[Exp 3 Cross-Lingual EN -> Hindi/Hinglish] Acc: {hi_acc:.4f} | F1: {hi_f1:.4f}")

    # Export cached text embeddings
    cache_path = ROOT / cfg["paths"]["cache_dir"] / "text_l1_embeddings.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out_npz = {}
    model.eval()
    with torch.no_grad():
        for name, ds in [("train", train_ds), ("val", val_ds), ("hindi", hi_ds)]:
            if ds is None or len(ds) == 0:
                continue
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
            embs, scs, lbs = [], [], []
            for ids, mask, y in loader:
                logits, emb = model(ids.to(device), mask.to(device))
                embs.append(emb.cpu().numpy())
                scs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                lbs.extend(y.numpy().tolist())
            out_npz[f"{name}_embeddings"] = np.concatenate(embs, axis=0)
            out_npz[f"{name}_scores"] = np.array(scs, dtype=np.float32)
            out_npz[f"{name}_labels"] = np.array(lbs, dtype=np.float32)
    np.savez(cache_path, **out_npz)

    res_path = ROOT / cfg["paths"]["results_dir"] / "text_eval_l1.json"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    eval_summary["training_time_sec"] = round(time.perf_counter() - t0, 2)
    res_path.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
    logger.info(f"Text v2 training finished in {eval_summary['training_time_sec']}s")


if __name__ == "__main__":
    main()
