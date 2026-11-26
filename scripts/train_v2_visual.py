#!/usr/bin/env python3
"""
Laptop 1 Training Script: VeriSight v2 Visual Branch (CLIP ViT-B/32 + EfficientNet-B0)
Saves compact trainable adapter weights (< 10 MB) to checkpoints/v2_visual_head.pt
and exports L2-normalized 256-d embeddings to data/cache/visual_l1_embeddings.npz.

Usage:
    python scripts/train_v2_visual.py --config configs/v2_cuda.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.dataset import VideoDataset
from src.models.v2.visual_model import VisualV2Detector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_v2_visual")


def save_adapter_state(model: VisualV2Detector, path: Path) -> None:
    """Save only trainable adapter & classifier weights (keeps checkpoint < 10 MB)."""
    state = {
        "adapter": model.adapter.state_dict(),
        "classifier": model.classifier.state_dict(),
        "embedding_dim": model.embedding_dim,
        "use_clip": model.use_clip,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main():
    parser = argparse.ArgumentParser(description="Train VeriSight v2 Visual Branch")
    parser.add_argument("--config", default="configs/v2_cuda.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    with open(ROOT / args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        logger.info(f"CUDA GPU: {props.name} ({props.total_memory / 1e9:.1f} GB VRAM)")
    else:
        logger.warning("CUDA not detected — running on CPU.")

    epochs = args.epochs or cfg["training_v2"]["epochs"]
    lr = cfg["training_v2"]["learning_rate"]
    v_cfg = cfg["models_v2"]["visual"]

    train_csv = ROOT / "data" / "video_train.csv"
    val_csv = ROOT / "data" / "video_val.csv"
    test_csv = ROOT / "data" / "video_test.csv"
    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("Run `python scripts/prepare_datasets_v2.py` first to generate video splits.")

    train_ds = VideoDataset(train_csv, max_frames=v_cfg["max_frames"], image_size=v_cfg["image_size"])
    val_ds = VideoDataset(val_csv, max_frames=v_cfg["max_frames"], image_size=v_cfg["image_size"])
    test_ds = VideoDataset(test_csv, max_frames=v_cfg["max_frames"], image_size=v_cfg["image_size"]) if test_csv.exists() else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    logger.info(f"Loading VisualV2Detector ({v_cfg['artifact_backbone']} spatial artifact branch; CLIP disabled for cross-identity generalization)...")
    model = VisualV2Detector(
        embedding_dim=cfg["models_v2"]["embedding_dim"],
        dropout=cfg["models_v2"]["dropout"],
        freeze_backbones=v_cfg["freeze_backbones"],
        use_clip=False,
        clip_name=v_cfg["clip_backbone"],
        artifact_backbone=v_cfg["artifact_backbone"],
        pretrained=True,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,} (backbones frozen)")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=cfg["training_v2"]["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    @torch.no_grad()
    def extract_split_features(loader):
        model.eval()
        feats_list, labels_list = [], []
        for x, y in loader:
            b, t, c, h, w = x.shape
            f_flat, _ = model._encode_frames(x.to(device).view(b * t, c, h, w))
            feats_list.append(f_flat.view(b, t, -1).cpu())
            labels_list.append(y.cpu())
        return torch.cat(feats_list, dim=0), torch.cat(labels_list, dim=0)

    logger.info("Extracting frozen CLIP + EffNet-B0 features once across train/val/test...")
    tr_feats, tr_labels = extract_split_features(train_loader)
    va_feats, va_labels = extract_split_features(val_loader)
    te_feats, te_labels = extract_split_features(DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)) if test_ds is not None else (None, None)

    feat_tr_loader = DataLoader(torch.utils.data.TensorDataset(tr_feats, tr_labels), batch_size=args.batch_size, shuffle=True)
    feat_va_loader = DataLoader(torch.utils.data.TensorDataset(va_feats, va_labels), batch_size=args.batch_size, shuffle=False)

    ckpt_path = ROOT / v_cfg["checkpoint"]
    best_val_loss = float("inf")
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        ep_t0 = time.perf_counter()
        model.adapter.train()
        model.classifier.train()
        tr_loss = 0.0
        n_tr = 0
        for f_bt, y in feat_tr_loader:
            f_bt, y = f_bt.to(device), y.to(device)
            b, t, d = f_bt.shape
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                emb_flat = model.adapter(f_bt.view(b * t, d))
                logits = model.classifier(emb_flat).squeeze(-1).view(b, t).mean(dim=1)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item() * len(y)
            n_tr += len(y)

        model.eval()
        val_loss = 0.0
        n_val = 0
        all_probs, all_labels = [], []
        with torch.no_grad():
            for f_bt, y in feat_va_loader:
                f_bt, y = f_bt.to(device), y.to(device)
                b, t, d = f_bt.shape
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    emb_flat = model.adapter(f_bt.view(b * t, d))
                    logits = model.classifier(emb_flat).squeeze(-1).view(b, t).mean(dim=1)
                    loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
                n_val += len(y)
                all_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                all_labels.extend(y.cpu().numpy().astype(int).tolist())

        tr_loss /= max(n_tr, 1)
        val_loss /= max(n_val, 1)
        preds = [1 if p >= 0.5 else 0 for p in all_probs]
        val_acc = accuracy_score(all_labels, preds)
        val_f1 = f1_score(all_labels, preds, zero_division=0)
        try:
            val_auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
        except Exception:
            val_auc = 0.0

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} ({time.perf_counter() - ep_t0:.2f}s) | "
            f"Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_adapter_state(model, ckpt_path)
            logger.info(f"  -> Saved best adapter checkpoint: {ckpt_path} ({ckpt_path.stat().st_size / 1e6:.2f} MB)")

    # Reload best weights before exporting cached embeddings
    best_st = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.adapter.load_state_dict(best_st["adapter"])
    model.classifier.load_state_dict(best_st["classifier"])

    # Export cached embeddings for Fusion + Exp 1-6
    cache_path = ROOT / cfg["paths"]["cache_dir"] / "visual_l1_embeddings.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    splits_out = {}
    model.eval()
    with torch.no_grad():
        for name, f_split, y_split in [("train", tr_feats, tr_labels), ("val", va_feats, va_labels), ("test", te_feats, te_labels)]:
            if f_split is None or len(f_split) == 0:
                continue
            embs, scores, labels = [], [], []
            for start in range(0, len(f_split), args.batch_size):
                f_bt = f_split[start:start + args.batch_size].to(device)
                b, t, d = f_bt.shape
                emb_flat = model.adapter(f_bt.view(b * t, d))
                logits = model.classifier(emb_flat).squeeze(-1).view(b, t).mean(dim=1)
                emb = torch.nn.functional.normalize(emb_flat.view(b, t, model.embedding_dim).mean(dim=1), p=2, dim=-1)
                embs.append(emb.cpu().numpy())
                scores.extend(torch.sigmoid(logits).cpu().numpy().tolist())
                labels.extend(y_split[start:start + args.batch_size].numpy().tolist())
            splits_out[f"{name}_embeddings"] = np.concatenate(embs, axis=0)
            splits_out[f"{name}_scores"] = np.array(scores, dtype=np.float32)
            splits_out[f"{name}_labels"] = np.array(labels, dtype=np.float32)
    np.savez(cache_path, **splits_out)

    elapsed = time.perf_counter() - t0
    logger.info(f"Visual v2 training + embedding export finished in {elapsed:.2f}s ({elapsed/60:.2f} min)")
    logger.info(f"Embeddings cached to: {cache_path}")


if __name__ == "__main__":
    main()
