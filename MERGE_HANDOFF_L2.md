# MERGE_HANDOFF_L2.md — VeriSight v2 Laptop 2 Handoff to Piyush (Laptop 1)

> **Status:** ✅ Complete — Trained, evaluated, tested (21/21 unit tests passing), and ready for merge.
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
| `scripts/train_audio_xlsr_v2.py` | Training script for `AudioV2Detector` (XLS-R 300M adapter head) |
| `scripts/run_exp3_robustness.py` | Experiment 3: H.264/JPEG/Gaussian noise/cross-dataset/AV-sync suite |
| `scripts/download_datasets_l2.py` | Dataset downloader for Laptop 2 (`ASVspoof2019_LA`, `audio_ood`, `video_ood`) |
| `tests/test_av_sync.py` | Unit tests for `av_sync` + `asr_batch` modules |
| `checkpoints/v2_audio_xlsr_head.pt` | Trained audio adapter head weights (`1.26 MB`) |
| `data/cache/audio_l2_embeddings.npz` | Balanced dev/eval 256-d audio embeddings, scores & temporal saliency (`3.99 MB`) |
| `data/cache/asr_transcripts_l2.json` | Whisper ASR transcripts for 840 clips (`1.43 MB`) |
| `results/audio_eval_l2.json` | Full in-domain (`71,237` clips) & OOD (`600` clips) evaluation metrics |
| `results/av_sync_eval_l2.json` | AV-sync aligned vs swapped-audio benchmark |
| `results/exp3_robustness.json` | Complete Experiment 3 JSON metrics |
| `results/plots/exp3_compression_curves.png` | H.264 CRF, JPEG Q, and Audio SNR robustness curves |
| `results/plots/exp3_audio_cross_dataset.png` | In-domain vs OOD audio generalization bar chart |
| `Dataset_L2/` | Local-only dataset folder (gitignored) |
| `MERGE_HANDOFF_L2.md` | This file |

---

## 2. Deliverable A — Audio Model Checkpoint

**Checkpoint path:** `checkpoints/v2_audio_xlsr_head.pt`

**Input/Output tensor shapes:**
```
Input:  waveform  — shape (B, N_samples)   where N_samples = 16000 * 5.0 = 80000
Output: logits    — shape (B,)              raw logit (apply torch.sigmoid for fake probability)
        embedding — shape (B, 256)          L2-normalized 256-d embedding for fusion
        saliency  — shape (B, 249)          temporal attention weights (when return_saliency=True)
```

**Backbone:** `facebook/wav2vec2-xls-r-300m` (frozen) + lightweight trainable head (`328,834` trainable parameters)

**Checkpoint file size:** `1.26 MB`

**Training & Evaluation Details (`results/audio_eval_l2.json`):**
| Setting / Split | Value |
|-----------------|-------|
| Dataset | ASVspoof 2019 LA (`25,380` train / `24,844` dev / `71,237` eval) |
| Epochs trained | `10` (`early_stopping_patience=3`, cosine LR) |
| Best epoch | `9` (`Best Dev ROC-AUC = 0.9989`) |
| Batch size | `16` (backbone cached once) / `64` (head epochs) |
| AMP (`fp16`) | Yes (`torch 2.5.1+cu121`) |
| RTX 4050 VRAM peak | `3.146 GB` |
| Total training + 71k eval time | `3,226.2 s` (~`53.8 min`) |
| Full Eval (`71,237` clips) | Acc `0.9685` \| F1 `0.9821` \| **ROC-AUC `0.9975`** |
| OOD Audio (`600` clips, `garystafford/deepfake-audio-detection`) | Acc `0.5283` \| F1 `0.6222` \| ROC-AUC `0.5757` |

**Pre-computed Embeddings (`data/cache/audio_l2_embeddings.npz`, `3.99 MB`):**
- Keys: `['embeddings', 'probs', 'saliency', 'labels', 'ids', 'splits']`
- Shapes: `embeddings=(5000, 256)` (`float16`), `saliency=(5000, 249)` (`float16`), `probs=(5000,)` (`float32`), `labels=(5000,)` (`int64`), `splits={'dev': 2500, 'eval': 2500}` (50/50 balanced bonafide/spoof per split)

---

## 3. Deliverable B — ASR Transcript Cache

**Cache path:** `data/cache/asr_transcripts_l2.json`

**Schema:**
```json
{
  "<parent_dir>__<clip_stem>": {
    "file": "<path to clip>",
    "language": "en",
    "language_probability": 0.998,
    "transcript": "The prime minister addressed the nation...",
    "word_timestamps": [
      {"word": "The", "start": 0.24, "end": 0.40, "probability": 0.99}
    ],
    "duration_seconds": 5.3,
    "whisper_model": "base"
  }
}
```

- **Whisper model used:** faster-whisper `base` (`compute_type="float16"` on CUDA)
- **Number of clips transcribed:** `840` (`40` from `video_ood`, `600` from `audio_ood`, `200` from `ASVspoof2019_LA_eval`)
- **Detected languages:** `{'en', 'hi', 'zh', 'ko', 'es', 'nn', 'cy', 'la'}`
- **Total JSON file size:** `1.43 MB`

---

## 4. Deliverable C — AV Sync Module Interface

**Module:** `src/models/av_sync.py`

**Public API:**
```python
from src.models.av_sync import compute_av_sync_score, AVSyncResult

result: AVSyncResult = compute_av_sync_score(
    video_path="path/to/clip.mp4",
    audio_path=None,            # None = extract audio from video via PyAV
    sample_fps=10.0,
    contradiction_threshold=0.40,
)

# Fields:
result.av_sync_score          # float in [0,1] or None if <3 face frames or silent/missing audio
result.is_av_contradiction    # bool — True if score < 0.40
result.reason                 # str explanation if None or contradiction
result.mfcc_correlation       # Pearson r of mouth-motion vs blended audio RMS+vocal-band envelope
result.n_frames_analyzed      # int
```

**AV Sync Benchmark on `Dataset_L2/video_ood/real` (`results/av_sync_eval_l2.json`):**
| Condition | Pairs | Mean Score | Std | Contradiction Rate (`<0.40`) |
|-----------|-------|------------|-----|------------------------------|
| Aligned pairs (original audio) | `20` | `0.5302` | `0.0757` | — |
| Swapped audio (simulated deepfake) | `20` | `0.4621` | `0.0899` | `15.0%` (`0.15`) |

---

## 5. Deliverable D — Experiment 3 Summary (`results/exp3_robustness.json`)

**Plots:**
- `results/plots/exp3_compression_curves.png`
- `results/plots/exp3_audio_cross_dataset.png`

### Audio Gaussian Noise Robustness (`ASVspoof2019_LA_eval`, `n=600` balanced)
| SNR | Acc | F1 | ROC-AUC |
|-----|-----|-----|---------|
| 20 dB | `0.7517` | `0.7732` | `0.8528` |
| 10 dB | `0.7800` | `0.7708` | `0.8602` |
|  5 dB | `0.7550` | `0.7721` | `0.8437` |

### H.264 Video Compression Robustness (`video_ood`, `n=40`)
| CRF | Acc | F1 | ROC-AUC | Mean AV-Sync |
|-----|-----|----|---------|--------------|
| CRF 23 (light) | `0.7500` | `0.7826` | `0.8075` | `0.5126` |
| CRF 32 (medium) | `0.7500` | `0.7826` | `0.8075` | `0.5340` |
| CRF 40 (heavy) | `0.7500` | `0.7826` | `0.8075` | `0.5194` |

### JPEG Frame Compression Robustness (`video_ood`, `n=30`)
| JPEG Quality | Mean AV-Sync | Mean Face Frames |
|--------------|--------------|------------------|
| Q=90 | `0.4460` | `62.00` |
| Q=60 | `0.4645` | `62.03` |
| Q=30 | `0.4553` | `62.23` |

### Cross-Dataset Audio Generalization (`n=600` each)
| Dataset | Acc | F1 | ROC-AUC |
|---------|-----|----|---------|
| In-domain (`ASVspoof 2019 LA eval`) | `0.9733` | `0.9731` | `0.9972` |
| OOD (`garystafford/deepfake-audio-detection`) | `0.5317` | `0.6258` | `0.5747` |

---

## 6. How Piyush (Laptop 1) Should Use These Outputs

### Merge steps:
```bash
# On Laptop 1 (Piyush):
git fetch origin
git checkout main
git merge origin/feat/laptop2-audio-asr-robustness
git push origin main
```

### Using `AudioV2Detector` in fusion / inference:
```python
from src.models.v2.audio_model import AudioV2Detector
import torch

device = torch.device("cuda")
model = AudioV2Detector(backbone_name="facebook/wav2vec2-xls-r-300m", freeze_backbone=True)
ckpt = torch.load("checkpoints/v2_audio_xlsr_head.pt", map_location=device, weights_only=True)
model.load_state_dict(ckpt["state_dict"], strict=False)
model.to(device).eval()

logits, embedding = model(waveform_tensor)  # embedding: (B, 256)
```

### Using pre-computed embeddings:
```python
import numpy as np
data = np.load("data/cache/audio_l2_embeddings.npz")
embeddings = data["embeddings"].astype(np.float32)  # shape: (5000, 256)
saliency   = data["saliency"].astype(np.float32)    # shape: (5000, 249)
probs      = data["probs"]                          # shape: (5000,)
labels     = data["labels"]                         # shape: (5000,)
splits     = data["splits"]                         # 'dev' (2500) | 'eval' (2500)
```

---

## 7. Tests

```bash
pytest tests/test_av_sync.py tests/test_v2_models.py -v
```
**All tests passing:** ✅ **21/21 passed** (`tests/test_av_sync.py` + `tests/test_v2_models.py`)

---

## 8. Notes for Piyush

- Layer-2 `av_sync_score` returns `None` when `<3` frames contain a detected face or audio is missing/silent → use `0.5` as neutral fallback in fusion.
- Note the strong cross-dataset drop (`0.9972` in-domain AUC -> `0.5747` OOD AUC on modern ElevenLabs/Polly/Kokoro TTS), which directly validates Paper #3 (`Deepfake-Eval-2024`) and Paper #23 (`Mueller et al. 2022`) for Experiment 3 in the paper!
- `checkpoints/v2_audio_xlsr_head.pt` requires `transformers==4.49.0` (with `torch 2.5.1`) or `torch >= 2.6` because `facebook/wav2vec2-xls-r-300m` uses `.bin` weights on HuggingFace.
