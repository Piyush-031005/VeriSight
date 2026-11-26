# MERGE_HANDOFF_L2.md — VeriSight v2 Laptop 2 Handoff to Piyush (Laptop 1)

> **Status:** ⏳ In Progress — Fill in metrics after training completes.
> Author: Laptop 2 (Audio/ASR/AV-Sync Engineer)
> Branch: `feat/laptop2-audio-asr-robustness`
> Target merge into: `main` (by Piyush — Laptop 1)

---

## 1. Files Added by This Branch (DO NOT TOUCH on Laptop 1)

| File | Description |
|------|-------------|
| `src/models/av_sync.py` | Layer-2 AV Sync / Lip-Sync module (`AVSyncDetector`, `compute_av_sync_score`) |
| `src/pipelines/__init__.py` | Package init for pipelines |
| `src/pipelines/asr_batch.py` | faster-whisper ASR batch transcription + `compute_audio_text_consistency()` |
| `scripts/train_audio_xlsr_v2.py` | Training script for AudioV2Detector (XLS-R adapter head) |
| `scripts/run_exp3_robustness.py` | Experiment 3: H.264/SNR/cross-dataset robustness suite |
| `tests/test_av_sync.py` | Unit tests for av_sync + asr_batch modules |
| `Dataset_L2/` | Local-only dataset folder (gitignored) |
| `MERGE_HANDOFF_L2.md` | This file |

---

## 2. Deliverable A — Audio Model Checkpoint

**Checkpoint path:** `checkpoints/v2_audio_xlsr_head.pt`

**Input/Output tensor shapes:**
```
Input:  waveform  — shape (B, N_samples)   where N_samples = 16000 * max_duration_seconds = 80000
Output: logits    — shape (B,)              raw logit (apply sigmoid for fake probability)
        embedding — shape (B, 256)          L2-normalized 256-d embedding for fusion
        saliency  — shape (B, T_frames)     temporal attention weights (returned when return_saliency=True)
```

**Backbone:** `facebook/wav2vec2-xls-r-300m` (frozen) + lightweight adapter head (~25 MB saved)

**Checkpoint file size:** _____ MB  ← Fill after training

**Training details:**
| Setting | Value |
|---------|-------|
| Dataset | ASVspoof 2019 LA (speaker-disjoint train/dev/eval) |
| Epochs trained | _____ |
| Best epoch | _____ |
| Batch size | 16 |
| AMP (fp16) | Yes |
| RTX 4050 VRAM peak | _____ GB |
| Training time | _____ hours |
| Best Dev AUC | _____ |

---

## 3. Deliverable B — ASR Transcript Cache

**Cache path:** `data/cache/asr_transcripts_l2.json`

**Schema:**
```json
{
  "<parent_dir>__<clip_stem>": {
    "file": "<absolute or relative path to clip>",
    "language": "en",
    "language_probability": 0.998,
    "transcript": "The prime minister addressed the nation...",
    "word_timestamps": [
      {"word": "The", "start": 0.24, "end": 0.40, "probability": 0.99},
      ...
    ],
    "duration_seconds": 5.3,
    "whisper_model": "base"
  }
}
```

**Whisper model used:** faster-whisper `base` (compute_type=`float16` on CUDA)
**Number of clips transcribed:** _____
**Total JSON file size:** _____ MB

---

## 4. Deliverable C — AV Sync Module Interface

**Module:** `src/models/av_sync.py`

**Public API:**
```python
from src.models.av_sync import compute_av_sync_score, AVSyncResult

result: AVSyncResult = compute_av_sync_score(
    video_path="path/to/clip.mp4",
    audio_path=None,            # None = extract audio from video itself
    sample_fps=10.0,            # Frames per second to sample
    contradiction_threshold=0.40,
)

# Fields:
result.av_sync_score          # float in [0,1] or None if no face/audio
result.is_av_contradiction    # bool — True if score < 0.40
result.reason                 # str explanation if None or contradiction
result.mfcc_correlation       # Pearson r of mouth-motion vs audio-RMS
result.n_frames_analyzed      # int
```

**AV Sync Benchmark on Dataset_L2/video_ood:**
| Condition | Mean Score | Std | Contradiction Rate |
|-----------|-----------|-----|-------------------|
| Aligned pairs (original audio) | _____ | _____ | _____ |
| Swapped audio (deepfake simulation) | _____ | _____ | _____ |

Full benchmark saved → `results/av_sync_eval_l2.json`

---

## 5. Deliverable D — Experiment 3 Summary

**Results file:** `results/exp3_robustness.json`
**Plots:**
- `results/plots/exp3_compression_curves.png`
- `results/plots/exp3_audio_cross_dataset.png`

### Audio Gaussian Noise Robustness (In-Domain ASVspoof 2019 LA Eval)
| SNR | Detection Rate | Mean P(Fake) |
|-----|---------------|--------------|
| 20 dB | _____ | _____ |
| 10 dB | _____ | _____ |
|  5 dB | _____ | _____ |

### H.264 Video Compression Robustness
| CRF | Mean P(Fake) | Notes |
|-----|-------------|-------|
| CRF 23 (light) | _____ | |
| CRF 32 (medium) | _____ | |
| CRF 40 (heavy) | _____ | |

### Cross-Dataset Audio Generalization
| Dataset | Acc | F1 | ROC-AUC |
|---------|----|----|---------|
| In-domain (ASVspoof 2019 LA eval) | _____ | _____ | _____ |
| OOD (WaveFake / Deepfake-Eval-2024) | _____ | _____ | _____ |

### AV Sync Contradiction Test
| Condition | Mean Score | Contradiction Detected |
|-----------|-----------|----------------------|
| Aligned (original audio) | _____ | _____ % |
| Swapped audio (simulated deepfake) | _____ | _____ % |

---

## 6. How Piyush (Laptop 1) Should Use These Outputs

### Merge steps:
```bash
# On Laptop 1 (Piyush):
git fetch origin
git checkout main
git merge feat/laptop2-audio-asr-robustness
# Resolve any conflicts (there should be none — zero file overlaps)
git push origin main
```

### Using AudioV2Detector in fusion training:
```python
from src.models.v2.audio_model import AudioV2Detector
import torch

device = torch.device("cuda")
model = AudioV2Detector(backbone_name="facebook/wav2vec2-xls-r-300m", freeze_backbone=True)
ckpt = torch.load("checkpoints/v2_audio_xlsr_head.pt", map_location=device)
model.load_state_dict(ckpt["state_dict"], strict=False)
model.to(device).eval()

# Get 256-d embedding for fusion (shape: (B, 256))
logits, embedding = model(waveform_tensor)
```

### Using pre-computed embeddings:
```python
import numpy as np
data = np.load("data/cache/audio_l2_embeddings.npz")
embeddings = data["embeddings"]   # shape: (N, 256)
saliency   = data["saliency"]     # shape: (N, T_frames)
labels     = data["labels"]       # shape: (N,)
ids        = data["ids"]          # shape: (N,)
```

### Using AV Sync in Layer-2 Consistency:
```python
from src.models.av_sync import compute_av_sync_score
result = compute_av_sync_score("path/to/video.mp4")
av_sync_score = result.av_sync_score or 0.5   # Use 0.5 as neutral fallback
```

### Using ASR transcript cache:
```python
import json
with open("data/cache/asr_transcripts_l2.json") as f:
    transcripts = json.load(f)
# transcripts["<dir>__<clip_id>"]["transcript"]  → spoken text
# transcripts["<dir>__<clip_id>"]["language"]    → "en" / "hi"
```

---

## 7. Tests

```bash
# Run Laptop 2 unit tests (should all pass):
pytest tests/test_av_sync.py -v
```

**All tests passing:** ✅ / ❌  ← Fill after running

---

## 8. Notes / Issues for Piyush

- _(Write any known issues, dataset quirks, or integration notes here)_
- Layer-2 `av_sync_score` is `None` when no face detected (< 3 frames with face) → Piyush's fusion should use 0.5 as neutral fallback
- `compute_audio_text_consistency()` in `asr_batch.py` requires `sentence-transformers` for best quality; falls back to Jaccard overlap if not installed
