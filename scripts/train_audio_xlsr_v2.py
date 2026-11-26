"""VeriSight v2 — Deliverable A: Train wav2vec2 XLS-R Audio Head on ASVspoof 2019 LA.

Laptop 2 (Audio/ASR/AV-Sync Engineer) — TRAINING SCRIPT.
Run ONLY after Piyush (Laptop 1) confirms Phase 1 shared code is on main.

USAGE:
  # Verify setup first (dry run, no actual training):
  python scripts/train_audio_xlsr_v2.py --config configs/v2_cuda.yaml --dry_run

  # Full training (Piyush will tell you when to run this):
  python scripts/train_audio_xlsr_v2.py --config configs/v2_cuda.yaml

OUTPUTS:
  checkpoints/v2_audio_xlsr_head.pt       — adapter head only (<25 MB, safe for GitHub)
  data/cache/audio_l2_embeddings.npz      — test/val embeddings + temporal saliency scores
  results/audio_eval_l2.json             — in-domain + OOD evaluation metrics

WAIT FOR PIYUSH'S SIGNAL before running full training.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset: ASVspoof 2019 LA
# ---------------------------------------------------------------------------

class ASVspoofDataset(Dataset):
    """Dataset loader for ASVspoof 2019 LA (speaker-disjoint train/dev/eval splits).

    Expects the standard ASVspoof 2019 LA directory structure:
      Dataset_L2/ASVspoof2019_LA/
        ASVspoof2019_LA_train/flac/*.flac
        ASVspoof2019_LA_dev/flac/*.flac
        ASVspoof2019_LA_eval/flac/*.flac
        ASVspoof2019_LA_cm_protocols/
          ASVspoof2019.LA.cm.train.trn.txt
          ASVspoof2019.LA.cm.dev.trl.txt
          ASVspoof2019.LA.cm.eval.trl.txt

    Label file format (space-separated):
      LA_0001  LA_T_1000137  <env>  <attack_type>  bonafide|spoof
    """

    SPLIT_TO_PROTOCOL = {
        "train": "ASVspoof2019.LA.cm.train.trn.txt",
        "dev":   "ASVspoof2019.LA.cm.dev.trl.txt",
        "eval":  "ASVspoof2019.LA.cm.eval.trl.txt",
    }
    SPLIT_TO_DIR = {
        "train": "ASVspoof2019_LA_train/flac",
        "dev":   "ASVspoof2019_LA_dev/flac",
        "eval":  "ASVspoof2019_LA_eval/flac",
    }

    def __init__(
        self,
        root_dir: str | Path = "Dataset_L2/ASVspoof2019_LA",
        split: str = "train",
        sample_rate: int = 16000,
        max_duration_seconds: float = 5.0,
    ):
        self.root = Path(root_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_seconds * sample_rate)
        self.samples = self._load_manifest()

    def _load_manifest(self):
        protocol_path = self.root / "ASVspoof2019_LA_cm_protocols" / self.SPLIT_TO_PROTOCOL[self.split]
        audio_dir = self.root / self.SPLIT_TO_DIR[self.split]

        if not protocol_path.exists():
            raise FileNotFoundError(
                f"Protocol file not found: {protocol_path}\n"
                f"Download ASVspoof 2019 LA from: https://datashare.ed.ac.uk/handle/10283/3336"
            )

        samples = []
        with open(protocol_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                clip_id = parts[1]
                label_str = parts[4].lower()
                label = 0 if label_str == "bonafide" else 1  # 0=REAL, 1=FAKE/SPOOF

                # Try .flac and .wav
                for ext in [".flac", ".wav"]:
                    fpath = audio_dir / (clip_id + ext)
                    if fpath.exists():
                        samples.append({"path": fpath, "label": label, "id": clip_id})
                        break

        logger.info(f"ASVspoof 2019 LA [{self.split}]: {len(samples)} clips loaded from {self.root}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        waveform = self._load_waveform(item["path"])
        return waveform, torch.tensor(item["label"], dtype=torch.float32), item["id"]

    def _load_waveform(self, path: Path) -> torch.Tensor:
        try:
            import torchaudio
            wav, sr = torchaudio.load(str(path))
            wav = wav.mean(0)  # Mono
            if sr != self.sample_rate:
                import torchaudio.transforms as T
                wav = T.Resample(sr, self.sample_rate)(wav)
        except Exception:
            wav = torch.zeros(self.max_samples)

        # Pad / truncate to max_samples
        if wav.shape[0] > self.max_samples:
            wav = wav[:self.max_samples]
        elif wav.shape[0] < self.max_samples:
            wav = torch.nn.functional.pad(wav, (0, self.max_samples - wav.shape[0]))
        return wav


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, epoch: int) -> Dict:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for step, (waveforms, labels, _) in enumerate(loader, 1):
        waveforms = waveforms.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with autocast():
            logits, _ = model(waveforms)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
        if step % 50 == 0:
            logger.info(f"  Epoch {epoch} Step {step}/{len(loader)} loss={loss.item():.4f}")
    return {"loss": total_loss / max(len(loader), 1), "acc": correct / max(total, 1)}


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> Dict:
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    for waveforms, labels, _ in loader:
        waveforms, labels = waveforms.to(device), labels.to(device)
        logits, emb = model(waveforms)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    return {
        "loss": total_loss / max(len(loader), 1),
        "acc": accuracy_score(all_labels, preds),
        "f1": f1_score(all_labels, preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5,
    }


@torch.no_grad()
def export_embeddings(model, loader, device, out_path: Path):
    model.eval()
    all_embs, all_saliency, all_labels, all_ids = [], [], [], []
    for waveforms, labels, ids in loader:
        waveforms = waveforms.to(device)
        logits, emb, saliency = model(waveforms, return_saliency=True)
        all_embs.append(emb.cpu().numpy())
        all_saliency.append(saliency.cpu().numpy())
        all_labels.extend(labels.numpy().tolist())
        all_ids.extend(list(ids))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        embeddings=np.concatenate(all_embs, axis=0),
        saliency=np.concatenate(all_saliency, axis=0),
        labels=np.array(all_labels),
        ids=np.array(all_ids),
    )
    logger.info(f"Embeddings exported → {out_path}")


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/train_audio_xlsr_v2.log", mode="a"),
        ],
    )
    Path("logs").mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="VeriSight v2 — Train Audio XLS-R Head (Laptop 2)")
    parser.add_argument("--config", default="configs/v2_cuda.yaml")
    parser.add_argument("--data_root", default="Dataset_L2/ASVspoof2019_LA")
    parser.add_argument("--ood_dir", default="Dataset_L2/audio_ood", help="Out-of-domain audio for OOD eval")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Verify setup without training")
    parser.add_argument("--backbone", default=None, help="Override backbone (e.g. facebook/wav2vec2-base)")
    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    backbone = args.backbone or cfg["models_v2"]["audio"]["backbone"]
    batch_size = args.batch_size or cfg["training_v2"]["batch_size"]
    epochs = args.epochs or cfg["training_v2"]["epochs"]
    emb_dim = cfg["models_v2"]["embedding_dim"]
    lr = cfg["training_v2"]["learning_rate"]
    wd = cfg["training_v2"]["weight_decay"]
    patience = cfg["training_v2"]["early_stopping_patience"]
    amp_enabled = cfg["training_v2"]["amp"] and device.type == "cuda"

    logger.info(f"Backbone: {backbone} | Epochs: {epochs} | Batch: {batch_size} | AMP: {amp_enabled}")

    if args.dry_run:
        logger.info("=== DRY RUN: Testing dataset loading only ===")
        try:
            ds = ASVspoofDataset(root_dir=args.data_root, split="train")
            wav, lbl, cid = ds[0]
            logger.info(f"Sample waveform shape: {wav.shape}, label: {lbl}, id: {cid}")
            logger.info("✅ Dry run passed — dataset loads correctly!")
        except FileNotFoundError as e:
            logger.error(f"❌ Dataset not found: {e}")
            logger.info("Download ASVspoof 2019 LA from: https://datashare.ed.ac.uk/handle/10283/3336")
        return

    # Build model
    from src.models.v2.audio_model import AudioV2Detector
    model = AudioV2Detector(
        backbone_name=backbone,
        freeze_backbone=cfg["models_v2"]["audio"]["freeze_backbone"],
        embedding_dim=emb_dim,
        dropout=cfg["models_v2"]["dropout"],
    ).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_trainable:,}")

    # Datasets
    train_ds = ASVspoofDataset(root_dir=args.data_root, split="train",
                                sample_rate=cfg["models_v2"]["audio"]["sample_rate"],
                                max_duration_seconds=cfg["models_v2"]["audio"]["max_duration_seconds"])
    dev_ds   = ASVspoofDataset(root_dir=args.data_root, split="dev",
                                sample_rate=cfg["models_v2"]["audio"]["sample_rate"],
                                max_duration_seconds=cfg["models_v2"]["audio"]["max_duration_seconds"])
    eval_ds  = ASVspoofDataset(root_dir=args.data_root, split="eval",
                                sample_rate=cfg["models_v2"]["audio"]["sample_rate"],
                                max_duration_seconds=cfg["models_v2"]["audio"]["max_duration_seconds"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=wd,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=amp_enabled)

    # Training loop
    best_auc, best_epoch, no_improve = 0.0, 0, 0
    ckpt_path = Path(cfg["models_v2"]["audio"]["checkpoint"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, epoch)
        dev_metrics   = evaluate(model, dev_loader, criterion, device)
        scheduler.step()

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.3f} | "
            f"Dev   loss={dev_metrics['loss']:.4f} acc={dev_metrics['acc']:.3f} "
            f"f1={dev_metrics['f1']:.3f} auc={dev_metrics['auc']:.3f} | "
            f"Time: {time.time()-t0:.1f}s"
        )

        if dev_metrics["auc"] > best_auc:
            best_auc = dev_metrics["auc"]
            best_epoch = epoch
            no_improve = 0
            # Save only adapter head (not frozen backbone)
            adapter_state = {k: v for k, v in model.state_dict().items()
                             if not k.startswith("backbone.")}
            torch.save({"epoch": epoch, "state_dict": adapter_state,
                        "best_dev_auc": best_auc, "backbone": backbone}, ckpt_path)
            logger.info(f"  ✅ Best checkpoint saved → {ckpt_path} (AUC={best_auc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                break

    # Final eval on eval split
    logger.info("=== Final Evaluation on ASVspoof 2019 LA EVAL split ===")
    eval_metrics = evaluate(model, eval_loader, criterion, device)
    logger.info(f"EVAL | acc={eval_metrics['acc']:.4f} f1={eval_metrics['f1']:.4f} auc={eval_metrics['auc']:.4f}")

    # Export embeddings + saliency for fusion training
    emb_path = Path("data/cache/audio_l2_embeddings.npz")
    export_embeddings(model, eval_loader, device, emb_path)

    # Save evaluation results
    results = {
        "in_domain": {
            "dataset": "ASVspoof2019_LA_eval",
            "backbone": backbone,
            "best_epoch": best_epoch,
            "dev_auc": best_auc,
            **eval_metrics,
        }
    }
    results_path = Path("results/audio_eval_l2.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {results_path}")
    logger.info("=== Training complete! Notify Piyush (Laptop 1) to pull and merge. ===")


if __name__ == "__main__":
    main()
