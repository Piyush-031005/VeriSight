"""VeriSight v2 Uncertainty Estimation: Temperature Scaling, MC-Dropout, ECE & Abstention."""

from __future__ import annotations

from typing import Callable, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling for calibrated probabilities."""

    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor(float(np.log(max(init_temp, 1e-3)))))

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temp).clamp(min=0.5, max=5.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 50) -> float:
        """Optimize temperature scalar on validation logits using L-BFGS."""
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.log_temp], lr=0.05, max_iter=max_iter)
        logits_d = val_logits.detach()
        labels_d = val_labels.detach().float()

        def closure():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits_d), labels_d)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.item())


def binary_entropy(probs: np.ndarray | torch.Tensor) -> np.ndarray:
    """Compute normalized binary predictive entropy in [0, 1]."""
    p = np.asarray(probs, dtype=np.float64).clip(1e-7, 1.0 - 1e-7)
    ent = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return ent


def mc_dropout_predict(
    forward_fn: Callable[[], torch.Tensor],
    model: nn.Module,
    n_samples: int = 15,
    temperature: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Run MC-Dropout stochastic forward passes to estimate mean prob & entropy."""
    was_training = model.training
    model.eval()
    # Enable Dropout layers only during MC sampling
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    probs_list = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = forward_fn() / max(temperature, 1e-4)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)

    if not was_training:
        model.eval()

    stacked = np.stack(probs_list, axis=0)  # (S, B)
    mean_prob = stacked.mean(axis=0)
    std_prob = stacked.std(axis=0)
    entropy = binary_entropy(mean_prob)
    return {
        "mean_prob": mean_prob,
        "std_prob": std_prob,
        "entropy": entropy,
    }


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) for binary probabilities."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    conf = np.maximum(p, 1.0 - p)
    preds = (p >= 0.5).astype(np.float64)
    acc = (preds == y).astype(np.float64)

    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (conf >= bins[i]) & (conf <= bins[i + 1])
        else:
            mask = (conf >= bins[i]) & (conf < bins[i + 1])
        if np.any(mask):
            ece += (np.sum(mask) / len(p)) * abs(float(np.mean(acc[mask])) - float(np.mean(conf[mask])))
    return float(ece)


def abstain_verdict(
    prob_fake: float,
    entropy: float,
    entropy_threshold: float = 0.45,
    conf_threshold: float = 0.60,
) -> Tuple[str, float]:
    """Return 3-way verdict ('REAL' | 'FAKE' | 'UNCERTAIN') and confidence."""
    confidence = float(max(prob_fake, 1.0 - prob_fake))
    if entropy > entropy_threshold or confidence < conf_threshold:
        return "UNCERTAIN", confidence
    return ("FAKE" if prob_fake >= 0.5 else "REAL"), confidence
