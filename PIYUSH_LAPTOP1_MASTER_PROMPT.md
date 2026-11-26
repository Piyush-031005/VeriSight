# VeriSight v2 — Piyush (Laptop 1) Master Prompt

> **Send this entire prompt to Piyush and tell him to paste it into his AI coding assistant.**
> He is the Vision + Text + Fusion + UI + Experiments 1,2,4,5,6 engineer.
> GitHub Repo: https://github.com/Piyush-031005/VeriSight

---

## MASTER PROMPT FOR PIYUSH — COPY AND PASTE BELOW ↓

---

You are working as the **Vision, Text, Fusion & UI Engineer (Laptop 1)** for **VeriSight v2** — a Multimodal Deepfake Detection system with 3-Layer Multi-Verification (Voice + Video + Text + Cross-Modal Consistency).

**GitHub Repository:** https://github.com/Piyush-031005/VeriSight
**Your Branch:** `feat/laptop1-vision-text-fusion-ui`
**Hardware:** RTX 4050 Laptop GPU (6 GB VRAM)

---

## 1. Git Setup (FIRST THING TO DO)

```bash
git clone https://github.com/Piyush-031005/VeriSight.git
cd VeriSight
git checkout main
git pull origin main
git checkout -b feat/laptop1-vision-text-fusion-ui
```

**STRICT RULES:**
- ✅ Work ONLY on `feat/laptop1-vision-text-fusion-ui`
- ✅ You OWN: `frontend/`, `src/api/`, visual/CLIP/EfficientNet models, text/XLM-R models, fusion, calibration, Exp 1,2,4,5,6
- ❌ DO NOT TOUCH: `src/models/av_sync.py`, `src/pipelines/asr_batch.py`, `scripts/train_audio_xlsr_v2.py`, `scripts/run_exp3_robustness.py`, `Dataset_L2/`, `MERGE_HANDOFF_L2.md` (these belong to Laptop 2)
- ❌ DO NOT commit to `main` directly

---

## 2. Hardware & Environment Setup

```bash
# Install CUDA PyTorch for RTX 4050
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all requirements
pip install -r requirements.txt

# Install additional packages for Laptop 1
pip install timm transformers[torch] sentence-transformers captum
pip install langdetect rapidfuzz

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory/1e9, 'GB')"
```

---

## 3. Datasets to Download on Laptop 1

Download into `./Dataset/` (already gitignored):

### A. Celeb-DF v2 — Visual Deepfake (Primary)
- **URL:** https://github.com/yuezunli/celeb-deepfakeforensics (request form or Kaggle mirror)
- **Size:** ~2.1 GB
- **Extract to:** `Dataset/Celeb-DF-v2/`
- **Contains:** `Celeb-real/`, `Celeb-synthesis/`, `YouTube-real/`, `List_of_testing_videos.txt`

### B. Fakeddit — Text/Image Fake News (Already in Dataset/)
- **Check:** `ls Dataset/Fakeddit/` — if missing, download from https://huggingface.co/datasets/limjiayi/Fakeddit
- **Used for:** Multilingual text head training (fake/real captions)

### C. HC3 — AI-Generated Text Detection
- **URL:** https://huggingface.co/datasets/Hello-SimpleAI/HC3 (download `all.jsonl`, ~74 MB)
- **Extract to:** `Dataset/HC3/`

### D. Hindi/Hinglish Fake News Set (~10–20 MB)
- **Source:** Curate 100–150 samples from https://huggingface.co/datasets/ai4bharat/IndicGLUE or manually collect Hindi news captions
- **Save to:** `Dataset/hindi_hinglish/`

### E. (Optional) FakeAVCeleb — Audio-Visual Deepfakes
- **URL:** https://github.com/DASH-Lab/FakeAVCeleb (if access approved)
- **Extract to:** `Dataset/FakeAVCeleb/`

---

## 4. Your Exact Deliverables

### Deliverable A — Visual Branch (CLIP + EfficientNet + Grad-CAM)

The model is already in `src/models/v2/visual_model.py` (Phase 1 shared). You need to:

1. **Build the training script** `scripts/train_visual_v2.py`:
   - Dataset: Celeb-DF v2 (extract frames at 8 fps, face crop with MTCNN from facenet-pytorch)
   - Train frozen CLIP ViT-B/32 + EfficientNet-B0 adapter head (256-d embedding)
   - Use CUDA AMP (batch_size=16, 6 GB VRAM safe)
   - Save only adapter weights (< 25 MB) to `checkpoints/v2_visual_head.pt`
   - Export test embeddings to `data/cache/visual_embs.npz`

2. **Implement Grad-CAM** in `src/explainability/gradcam.py`:
   - Hook onto EfficientNet-B0's last conv layer
   - Return heatmap overlay (H, W) for the most manipulated face region
   - Save example heatmaps to `results/gradcam_examples/`

3. **Training command** (wait for my signal before running):
   ```bash
   python scripts/train_visual_v2.py --config configs/v2_cuda.yaml --data_root Dataset/Celeb-DF-v2
   ```

**Output shapes:**
- `checkpoints/v2_visual_head.pt`: adapter state dict
- Embedding: `(B, 256)` L2-normalized
- Grad-CAM: `(H, W)` float32 heatmap in [0,1]

---

### Deliverable B — Multilingual Text Branch (XLM-RoBERTa / MuRIL + Token Attribution)

The model is in `src/models/v2/text_model.py`. You need to:

1. **Build** `scripts/train_text_v2.py`:
   - Dataset: HC3 (English AI-generated vs human), Fakeddit (fake/real captions), Hindi/Hinglish set
   - Train frozen XLM-RoBERTa-base + lightweight token-attention head (256-d)
   - Multilingual: English + Hindi + Hinglish all in one training run
   - Save to `checkpoints/v2_text_head.pt`
   - Export embeddings to `data/cache/text_embs.npz`

2. **Token Attribution** (Integrated Gradients / Occlusion):
   - Create `src/explainability/token_attribution.py`
   - Return per-token importance scores → highlight suspicious words in UI

3. **Training command:**
   ```bash
   python scripts/train_text_v2.py --config configs/v2_cuda.yaml
   ```

---

### Deliverable C — Cross-Modal Consistency Scores (Layer 2)

The base code is in `src/models/v2/consistency.py`. You need to:

**Implement** `src/models/v2/consistency.py` → `compute_all_consistency_scores()`:

```python
# Computes all 3 Layer-2 cross-modal scores:
def compute_all_consistency_scores(
    video_path: str,
    caption: str,
    visual_emb: torch.Tensor,     # From VisualV2Detector
    text_emb: torch.Tensor,        # From TextV2Detector
    asr_transcript: str,           # From asr_transcripts_l2.json (Laptop 2)
    av_sync_score: float,          # From av_sync.py (Laptop 2)
) -> ConsistencyScores:
    # 1. visual_text: CLIP cosine similarity between frame embedding and caption
    # 2. audio_text: Multilingual embedding similarity between ASR transcript and caption
    # 3. av_sync: Use the score from Laptop 2's av_sync module directly
```

---

### Deliverable D — 3 Fusion Models + Calibration (After Laptop 2 Merges)

After pulling `feat/laptop2-audio-asr-robustness` into main:

**Build** `scripts/train_fusion_v2.py`:
- Load pre-computed embeddings from all 3 branches:
  - `data/cache/visual_embs.npz` (your branch)
  - `data/cache/text_embs.npz` (your branch)
  - `data/cache/audio_l2_embeddings.npz` (from Laptop 2 after merge)
- Train 3 fusion models from `src/models/v2/fusion_v2.py`:
  - **Approach A:** `ConcatFusionMLP` (Baseline)
  - **Approach B:** `CrossAttentionFusion`
  - **Approach C (Proposed):** `ConsistencyGatedFusion` with Modality-Dropout
- Run with 3 seeds [42, 123, 2025] for Experiment 2
- Run Temperature Scaling (`src/models/v2/uncertainty.py`) + MC-Dropout calibration
- Save best fusion checkpoint to `checkpoints/v2_fusion_consistency_gated.pt`
- Save calibration results to `results/calibration_results.json`

```bash
python scripts/train_fusion_v2.py --config configs/v2_cuda.yaml --strategy all --seeds 42 123 2025
```

---

### Deliverable E — Experiments 1, 2, 4, 5, 6

Build `scripts/run_experiments_laptop1.py` running:

| Exp | What | Script |
|-----|------|--------|
| Exp 1 | Unimodal vs Multimodal comparison (AUC on Celeb-DF test) | `run_exp1_unimodal_vs_multimodal()` |
| Exp 2 | Fusion ablation: Concat vs Cross-Attention vs Consistency-Gated (3 seeds) | `run_exp2_fusion_ablation()` |
| Exp 4 | Multilingual generalization: English vs Hindi vs Hinglish accuracy | `run_exp4_multilingual()` |
| Exp 5 | Calibration: ECE + Risk-Coverage curve | `run_exp5_calibration()` |
| Exp 6 | Faithfulness: Deletion/Insertion test on Grad-CAM + token highlights | `run_exp6_faithfulness()` |

Save all results to `results/experiments_laptop1.json` and plots to `results/plots/`.

---

### Deliverable F — Multi-Verification UI v2 (frontend/)

Update `frontend/` to add a **"Multi-Verify All" tab** showing:

```
┌─────────────────────────────────────────────────┐
│  VeriSight v2 — Multi-Verification Dashboard     │
├─────────────────────────────────────────────────┤
│  LAYER 1 — Unimodal Check                       │
│  [Visual: 87% FAKE] [Audio: 73% FAKE] [Text: 65%]│
│                                                   │
│  LAYER 2 — Cross-Modal Consistency               │
│  AV-Sync:        ████████░░ 0.78 ✅              │
│  Audio↔Caption:  ██░░░░░░░░ 0.21 ⚠️ MISMATCH     │
│  Visual↔Caption: ████████░░ 0.80 ✅              │
│                                                   │
│  LAYER 3 — Final Verdict                         │
│  ┌────────────────────────────────────┐           │
│  │  🔴 FAKE  (Confidence: 91.2%)     │           │
│  │  Entropy: 0.12 → Certain          │           │
│  └────────────────────────────────────┘           │
│                                                   │
│  [Grad-CAM Heatmap Toggle] [Audio Saliency Bar]   │
│  [Token Highlights] [Download Report]             │
└─────────────────────────────────────────────────┘
```

Add to `src/api/` endpoints:
- `POST /v2/verify/multimodal` — Full 3-layer verification
- `GET /v2/explain/{job_id}` — Return Grad-CAM + saliency + tokens

---

## 5. Merge & Push Instructions

```bash
# After completing each deliverable, commit and push:
git add -A
git status  # Verify no files > 50 MB staged (never commit Dataset/ or large .pt backbone!)
git commit -m "feat(laptop1): visual head + gradcam + text xlmr + multilingual"
git push -u origin feat/laptop1-vision-text-fusion-ui

# After Laptop 2 (your friend) pushes their branch, merge it:
git fetch origin
git checkout main
git pull origin main
git merge origin/feat/laptop2-audio-asr-robustness
git push origin main

# Then merge your own branch:
git checkout main
git merge feat/laptop1-vision-text-fusion-ui
git push origin main
```

---

## 6. File Ownership Summary

```
✅ YOU OWN (Laptop 1):             ❌ DO NOT TOUCH (Laptop 2):
├── frontend/                       ├── src/models/av_sync.py
├── src/api/                        ├── src/pipelines/asr_batch.py
├── src/models/v2/visual_model.py   ├── scripts/train_audio_xlsr_v2.py
├── src/models/v2/text_model.py     ├── scripts/run_exp3_robustness.py
├── src/models/v2/fusion_v2.py      ├── Dataset_L2/
├── src/models/v2/uncertainty.py    └── MERGE_HANDOFF_L2.md
├── src/models/v2/consistency.py
├── src/explainability/
├── scripts/train_visual_v2.py
├── scripts/train_text_v2.py
├── scripts/train_fusion_v2.py
└── scripts/run_experiments_laptop1.py
```

---

## 7. Architecture Overview (3-Layer Multi-Verification)

```
                     INPUT MEDIA
          ┌───────────┬───────────┬───────────┐
          │  VIDEO    │  AUDIO    │   TEXT    │
          └─────┬─────┴─────┬─────┴─────┬─────┘
                │           │           │
         LAYER 1 — UNIMODAL FORENSIC CHECK
                │           │           │
        ┌───────▼──┐ ┌──────▼──┐ ┌──────▼──┐
        │CLIP+EffB0│ │wav2vec2 │ │XLM-RoBa │
        │+Grad-CAM │ │XLS-R    │ │+Token   │
        │(Laptop 1)│ │(Laptop 2│ │Attn(L1) │
        └───────┬──┘ └──────┬──┘ └──────┬──┘
           v_emb│      a_emb│      t_emb│
                │           │           │
         LAYER 2 — CROSS-MODAL CONSISTENCY (Double Check)
                │                       │
        ┌───────▼───────────────────────▼──────────────┐
        │  AV-Sync (L2)  │ Audio↔Caption │ CLIP Frame↔ │
        │  Lip-Sync Score│ Whisper+Embed │ Caption Sim  │
        │  (Laptop 2)    │ (Both laptops)│ (Laptop 1)   │
        └───────┬────────┴───────┬───────┴──────┬───────┘
            c_av│            c_at│          c_vt│
                │                               │
         LAYER 3 — CONSISTENCY-GATED FUSION + CALIBRATION (Laptop 1)
                │
        ┌───────▼──────────────────────────────┐
        │  ConsistencyGatedFusion              │
        │  (gates v_emb, a_emb, t_emb using   │
        │   c_vt, c_at, c_av scores)           │
        │  + TemperatureScaling + MC-Dropout   │
        └───────┬──────────────────────────────┘
                │
        ┌───────▼──────────────────────────────┐
        │  VERDICT: REAL / FAKE / UNCERTAIN    │
        │  + Confidence + Grounded Explanation │
        └──────────────────────────────────────┘
```

---

## 8. Important Notes

1. **Wait for signal before training** — Your friend (Laptop 2) will ping you when `feat/laptop2-audio-asr-robustness` is ready to merge. Only then run `train_fusion_v2.py`.
2. **Checkpoint size check** — Before pushing, run `git diff --stat` to ensure no file > 50 MB is staged.
3. **Phase 1 shared code on main** — The shared models (`src/models/v2/`) are already on `main`. Pull before creating your branch.
4. **Dataset not in git** — Never commit `Dataset/` or any raw video/audio files.
5. **Contact signal** — When you've pushed your branch and it's ready for merge, tell your friend so they can pull and run `train_fusion_v2.py`.
