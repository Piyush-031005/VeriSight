"""VeriSight v2 — Deliverable A: Train wav2vec2 XLS-R Audio Head on ASVspoof 2019 LA.

Laptop 2 (Audio/ASR/AV-Sync Engineer) — TRAINING SCRIPT.

USAGE:
  # Verify setup first (dry run, no actual training):
  python scripts/train_audio_xlsr_v2.py --config configs/v2_cuda.yaml --dry_run

  # Full training (run after dataset download & confirmation):
  python scripts/train_audio_xlsr_v2.py --config configs/v2_cuda.yaml

OUTPUTS:
  checkpoints/v2_audio_xlsr_head.pt       — adapter head only (<25 MB, safe for GitHub)
  data/cache/audio_l2_embeddings.npz      — dev/eval embeddings + scores + temporal saliency
  results/audio_eval_l2.json              — in-domain + OOD evaluation metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

AUDIO_EXTS = {".flac", ".wav", ".mp3", ".ogg", ".m4a"}


def resolve_asvspoof_root(root_dir: str | Path) -> Path:
    """Support both Dataset_L2/ASVspoof2019_LA and nested Dataset_L2/ASVspoof2019_LA/LA."""
    root = Path(root_dir)
    if (root / "ASVspoof2019_LA_cm_protocols").exists():
        return root
    if (root / "LA" / "ASVspoof2019_LA_cm_protocols").exists():
        return root / "LA"
    return root


def load_audio_tensor(path: Path, sample_rate: int = 16000, max_samples: int = 80000) -> Tuple[torch.Tensor, bool]:
    """Decode waveform using soundfile -> PyAV -> torchaudio; pad/truncate to max_samples."""
    wav_np: Optional[np.ndarray] = None

    try:
        import soundfile as sf
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != sample_rate:
            import scipy.signal as ss
            data = ss.resample_poly(data, sample_rate, sr).astype(np.float32)
        wav_np = data.astype(np.float32, copy=False)
    except Exception:
        pass

    if wav_np is None:
        try:
            import av
            with av.open(str(path)) as container:
                resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=sample_rate)
                chunks = []
                for frame in container.decode(audio=0):
                    for rframe in resampler.resample(frame):
                        chunks.append(rframe.to_ndarray().reshape(-1))
                for rframe in resampler.resample(None):
                    chunks.append(rframe.to_ndarray().reshape(-1))
            if chunks:
                wav_np = np.concatenate(chunks).astype(np.float32, copy=False)
        except Exception:
            pass

    if wav_np is None:
        try:
            import torchaudio
            wav_t, sr = torchaudio.load(str(path))
            wav_t = wav_t.mean(0)
            if sr != sample_rate:
                import torchaudio.transforms as T
                wav_t = T.Resample(sr, sample_rate)(wav_t)
            wav_np = wav_t.numpy().astype(np.float32, copy=False)
        except Exception as exc:
            logger.warning("Failed to decode %s (%s); substituting silence.", path, exc)
            return torch.zeros(max_samples, dtype=torch.float32), False

    wav = torch.from_numpy(wav_np)
    if wav.shape[0] > max_samples:
        wav = wav[:max_samples]
    elif wav.shape[0] < max_samples:
        wav = torch.nn.functional.pad(wav, (0, max_samples - wav.shape[0]))
    return wav, True


class ASVspoofDataset(Dataset):
    """Dataset loader for ASVspoof 2019 LA (speaker-disjoint train/dev/eval splits)."""

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
        max_samples_limit: Optional[int] = None,
    ):
        self.root = resolve_asvspoof_root(root_dir)
        self.split = split
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_seconds * sample_rate)
        self.max_samples_limit = max_samples_limit
        self.decode_failures = 0
        self.samples = self._load_manifest()

    def _load_manifest(self) -> List[Dict]:
        protocol_path = self.root / "ASVspoof2019_LA_cm_protocols" / self.SPLIT_TO_PROTOCOL[self.split]
        audio_dir = self.root / self.SPLIT_TO_DIR[self.split]

        if not protocol_path.exists():
            raise FileNotFoundError(
                f"Protocol file not found: {protocol_path}\n"
                f"Download ASVspoof 2019 LA from: https://datashare.ed.ac.uk/handle/10283/3336"
            )

        # Index audio files once in O(N) instead of calling stat() 25k+ times.
        file_index: Dict[str, Path] = {}
        if audio_dir.exists():
            for p in audio_dir.iterdir():
                if p.is_file() and p.suffix.lower() in {".flac", ".wav"}:
                    file_index[p.stem] = p

        samples: List[Dict] = []
        with open(protocol_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                clip_id = parts[1]
                label = 0 if parts[4].lower() == "bonafide" else 1
                fpath = file_index.get(clip_id)
                if fpath is not None:
                    samples.append({"path": fpath, "label": label, "id": clip_id})

        if self.max_samples_limit and len(samples) > self.max_samples_limit:
            # Stratified cap so both bonafide and spoof remain represented.
            real = [s for s in samples if s["label"] == 0]
            fake = [s for s in samples if s["label"] == 1]
            half = self.max_samples_limit // 2
            samples = real[:half] + fake[: self.max_samples_limit - min(half, len(real))]

        logger.info(f"ASVspoof 2019 LA [{self.split}]: {len(samples)} clips loaded from {self.root}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        waveform, ok = load_audio_tensor(item["path"], self.sample_rate, self.max_samples)
        if not ok:
            self.decode_failures += 1
        return waveform, torch.tensor(item["label"], dtype=torch.float32), item["id"]


class FolderAudioDataset(Dataset):
    """Out-of-domain audio dataset reading real/bonafide (0) vs fake/spoof (1) subfolders."""

    def __init__(
        self,
        root_dir: str | Path,
        sample_rate: int = 16000,
        max_duration_seconds: float = 5.0,
        max_files: Optional[int] = None,
    ):
        self.root = Path(root_dir)
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_seconds * sample_rate)
        self.samples: List[Dict] = []

        if self.root.exists():
            for p in sorted(self.root.rglob("*")):
                if not (p.is_file() and p.suffix.lower() in AUDIO_EXTS):
                    continue
                parts_lower = {part.lower() for part in p.relative_to(self.root).parts[:-1]}
                if parts_lower & {"real", "bonafide", "human", "genuine"}:
                    label = 0
                elif parts_lower & {"fake", "spoof", "synthetic", "ai"}:
                    label = 1
                else:
                    continue
                self.samples.append({"path": p, "label": label, "id": f"{p.parent.name}__{p.stem}"})

        if max_files and len(self.samples) > max_files:
            real = [s for s in self.samples if s["label"] == 0]
            fake = [s for s in self.samples if s["label"] == 1]
            half = max_files // 2
            self.samples = real[:half] + fake[: max_files - min(half, len(real))]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        waveform, _ = load_audio_tensor(item["path"], self.sample_rate, self.max_samples)
        return waveform, torch.tensor(item["label"], dtype=torch.float32), item["id"]


def _make_amp_helpers(amp_enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        autocast_ctx = lambda: torch.amp.autocast("cuda", enabled=amp_enabled)
    else:
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler(enabled=amp_enabled)
        autocast_ctx = lambda: autocast(enabled=amp_enabled)
    return scaler, autocast_ctx


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, autocast_ctx, epoch: int) -> Dict:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for step, (waveforms, labels, _) in enumerate(loader, 1):
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            logits, _ = model(waveforms)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()
        if step % 100 == 0:
            logger.info(f"  Epoch {epoch} Step {step}/{len(loader)} loss={loss.item():.4f}")
    return {"loss": total_loss / max(len(loader), 1), "acc": correct / max(total, 1)}


@torch.no_grad()
def evaluate(model, loader, criterion, device, autocast_ctx) -> Dict:
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0
    for waveforms, labels, _ in loader:
        waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_ctx():
            logits, _ = model(waveforms)
            loss = criterion(logits, labels)
        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits.float()).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else None
    return {
        "n_samples": len(all_labels),
        "loss": round(total_loss / max(len(loader), 1), 4),
        "acc": round(float(accuracy_score(all_labels, preds)), 4),
        "f1": round(float(f1_score(all_labels, preds, zero_division=0)), 4),
        "auc": round(float(auc), 4) if auc is not None else None,
    }


@torch.no_grad()
def export_embeddings(model, loaders_with_splits: List[Tuple[str, DataLoader]], device, autocast_ctx, out_path: Path):
    model.eval()
    all_embs, all_probs, all_saliency, all_labels, all_ids, all_splits = [], [], [], [], [], []
    for split_name, loader in loaders_with_splits:
        for waveforms, labels, ids in loader:
            waveforms = waveforms.to(device, non_blocking=True)
            with autocast_ctx():
                logits, emb, saliency = model(waveforms, return_saliency=True)
            all_embs.append(emb.float().cpu().numpy())
            all_probs.extend(torch.sigmoid(logits.float()).cpu().tolist())
            all_saliency.append(saliency.float().cpu().numpy())
            all_labels.extend(labels.numpy().tolist())
            all_ids.extend(list(ids))
            all_splits.extend([split_name] * len(ids))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        embeddings=np.concatenate(all_embs, axis=0),
        probs=np.array(all_probs, dtype=np.float32),
        saliency=np.concatenate(all_saliency, axis=0),
        labels=np.array(all_labels, dtype=np.int64),
        ids=np.array(all_ids),
        splits=np.array(all_splits),
    )
    logger.info(f"Embeddings exported -> {out_path} ({len(all_ids)} total samples)")


def main():
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/train_audio_xlsr_v2.log", mode="a", encoding="utf-8"),
        ],
    )

    parser = argparse.ArgumentParser(description="VeriSight v2 — Train Audio XLS-R Head (Laptop 2)")
    parser.add_argument("--config", default="configs/v2_cuda.yaml")
    parser.add_argument("--data_root", default="Dataset_L2/ASVspoof2019_LA")
    parser.add_argument("--ood_dir", default="Dataset_L2/audio_ood", help="Out-of-domain audio for OOD eval")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=(0 if sys.platform == "win32" else 2))
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Verify setup without training")
    parser.add_argument("--backbone", default=None, help="Override backbone (e.g. facebook/wav2vec2-base)")
    args = parser.parse_args()

    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
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
    amp_enabled = bool(cfg["training_v2"]["amp"] and device.type == "cuda")
    sample_rate = cfg["models_v2"]["audio"]["sample_rate"]
    max_duration = cfg["models_v2"]["audio"]["max_duration_seconds"]

    logger.info(f"Backbone: {backbone} | Epochs: {epochs} | Batch: {batch_size} | AMP: {amp_enabled}")

    if args.dry_run:
        logger.info("=== DRY RUN: Testing dataset & model readiness ===")
        try:
            ds = ASVspoofDataset(root_dir=args.data_root, split="train", sample_rate=sample_rate, max_duration_seconds=max_duration)
            if len(ds) == 0:
                logger.error("Protocol found, but 0 audio files matched.")
            else:
                wav, lbl, cid = ds[0]
                logger.info(f"Sample waveform shape: {wav.shape}, label: {lbl}, id: {cid}")
                logger.info("Dry run passed — dataset loads correctly!")
        except FileNotFoundError as e:
            logger.error(f"Dataset not found: {e}")
            logger.info("Run: python scripts/download_datasets_l2.py --dataset asvspoof")
        return

    from src.models.v2.audio_model import AudioV2Detector
    model = AudioV2Detector(
        backbone_name=backbone,
        freeze_backbone=cfg["models_v2"]["audio"]["freeze_backbone"],
        embedding_dim=emb_dim,
        dropout=cfg["models_v2"]["dropout"],
    ).to(device)
    if getattr(model, "use_mock_backbone", False):
        raise RuntimeError(
            f"Backbone '{backbone}' fell back to mock backbone! "
            "Check internet/HuggingFace access and ensure transformers>=4.48.0,<5 is installed."
        )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_trainable:,}")

    train_ds = ASVspoofDataset(args.data_root, "train", sample_rate, max_duration, args.max_train_samples)
    dev_ds   = ASVspoofDataset(args.data_root, "dev",   sample_rate, max_duration, args.max_eval_samples)
    eval_ds  = ASVspoofDataset(args.data_root, "eval",  sample_rate, max_duration, args.max_eval_samples)

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=pin)
    dev_loader   = DataLoader(dev_ds,   batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    eval_loader  = DataLoader(eval_ds,  batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    # Class-balanced pos_weight (bonafide is minority ~10% in ASVspoof LA).
    n_pos = sum(1 for s in train_ds.samples if s["label"] == 1)
    n_neg = max(1, len(train_ds.samples) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    logger.info(f"Train class counts -> real(0)={n_neg}, fake(1)={n_pos}, pos_weight={float(pos_weight):.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler, autocast_ctx = _make_amp_helpers(amp_enabled)

    best_auc, best_epoch, no_improve = -1.0, 0, 0
    ckpt_path = Path(cfg["models_v2"]["audio"]["checkpoint"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    t_train_start = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, autocast_ctx, epoch)
        dev_metrics   = evaluate(model, dev_loader, criterion, device, autocast_ctx)
        scheduler.step()

        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.3f} | "
            f"Dev   loss={dev_metrics['loss']:.4f} acc={dev_metrics['acc']:.3f} "
            f"f1={dev_metrics['f1']:.3f} auc={dev_metrics['auc']} | "
            f"Time: {time.time()-t0:.1f}s"
        )

        cur_auc = dev_metrics["auc"] if dev_metrics["auc"] is not None else dev_metrics["acc"]
        if cur_auc > best_auc:
            best_auc = cur_auc
            best_epoch = epoch
            no_improve = 0
            adapter_state = {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")}
            torch.save(
                {"epoch": epoch, "state_dict": adapter_state, "best_dev_auc": best_auc, "backbone": backbone},
                ckpt_path,
            )
            logger.info(f"  Best checkpoint saved -> {ckpt_path} (AUC={best_auc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
                break

    total_train_time = round(time.time() - t_train_start, 1)

    # Reload best weights before final eval & embedding export.
    if ckpt_path.exists():
        best_ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        model.load_state_dict(best_ckpt.get("state_dict", best_ckpt), strict=False)
        logger.info(f"Reloaded best checkpoint from epoch {best_epoch} for final evaluation.")

    logger.info("=== Final Evaluation on ASVspoof 2019 LA EVAL split ===")
    eval_metrics = evaluate(model, eval_loader, criterion, device, autocast_ctx)
    logger.info(f"EVAL | acc={eval_metrics['acc']} f1={eval_metrics['f1']} auc={eval_metrics['auc']}")

    ood_metrics = None
    ood_ds = FolderAudioDataset(args.ood_dir, sample_rate=sample_rate, max_duration_seconds=max_duration)
    if len(ood_ds) > 0:
        ood_loader = DataLoader(ood_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
        ood_metrics = evaluate(model, ood_loader, criterion, device, autocast_ctx)
        logger.info(f"OOD ({args.ood_dir}) | n={len(ood_ds)} acc={ood_metrics['acc']} f1={ood_metrics['f1']} auc={ood_metrics['auc']}")
    else:
        logger.info(f"No labeled OOD audio found under {args.ood_dir} (expect real/ and fake/ subfolders).")

    export_embeddings(
        model,
        [("dev", dev_loader), ("eval", eval_loader)],
        device,
        autocast_ctx,
        Path("data/cache/audio_l2_embeddings.npz"),
    )

    peak_vram_gb = round(torch.cuda.max_memory_allocated(device) / 1e9, 3) if device.type == "cuda" else 0.0
    ckpt_size_mb = round(ckpt_path.stat().st_size / (1024 * 1024), 2) if ckpt_path.exists() else None

    results = {
        "in_domain": {
            "dataset": "ASVspoof2019_LA_eval",
            "backbone": backbone,
            "best_epoch": best_epoch,
            "best_dev_auc": best_auc,
            "training_time_seconds": total_train_time,
            "peak_vram_gb": peak_vram_gb,
            "checkpoint_size_mb": ckpt_size_mb,
            **eval_metrics,
        },
        "out_of_domain": ood_metrics,
    }
    results_path = Path("results/audio_eval_l2.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved -> {results_path}")


if __name__ == "__main__":
    main()
