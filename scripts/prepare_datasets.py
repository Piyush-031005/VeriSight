#!/usr/bin/env python3
"""
Prepare compact multimodal training splits from VeriSight/Dataset/.

Reads:
  - Dataset/celebdf_v2/{real,fake}/*.mp4       (Video deepfakes)
  - Dataset/deepfake_audio/{real,fake}/*.flac  (Audio deepfakes)
  - Dataset/hc3_english/*.jsonl                (English Human vs ChatGPT)
  - Dataset/hindi_hinglish/*                   (Hindi / Hinglish text)
  - Dataset/all_validate.tsv                   (Fakeddit text+image metadata)

Outputs into data/:
  - video_train.csv, video_val.csv, video_test.csv
  - audio_train.csv, audio_val.csv, audio_test.csv
  - text_train.csv, text_val.csv
  - train.csv, val.csv, test.csv (combined multimodal manifest)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "Dataset"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def extract_video_identity(filename: str) -> str:
    m = re.match(r"^(id\d+)", filename, re.IGNORECASE)
    return m.group(1).lower() if m else Path(filename).stem


def extract_audio_identity(filename: str) -> str:
    # e.g. el_0001_c_part_002.flac -> el_0001, yt_0000_part_001.flac -> yt_0000
    m = re.match(r"^([a-z]+_\d+)", filename, re.IGNORECASE)
    return m.group(1).lower() if m else Path(filename).stem


def split_by_identity(df: pd.DataFrame, seed: int = 42):
    identities = sorted(df["identity_id"].unique())
    rng = pd.Series(identities).sample(frac=1.0, random_state=seed).tolist()
    n = len(rng)
    if n < 3:
        # Fallback row-level split if too few unique identities
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_tr = max(1, int(len(shuffled) * 0.70))
        n_va = max(1, int(len(shuffled) * 0.15))
        return shuffled.iloc[:n_tr], shuffled.iloc[n_tr:n_tr + n_va], shuffled.iloc[n_tr + n_va:]

    n_tr = max(1, int(round(n * 0.70)))
    n_va = max(1, int(round(n * 0.15)))
    if n_tr + n_va >= n:
        n_va = max(1, n - n_tr - 1)
    tr_ids = set(rng[:n_tr])
    va_ids = set(rng[n_tr:n_tr + n_va])
    te_ids = set(rng[n_tr + n_va:])

    return (
        df[df["identity_id"].isin(tr_ids)].reset_index(drop=True),
        df[df["identity_id"].isin(va_ids)].reset_index(drop=True),
        df[df["identity_id"].isin(te_ids)].reset_index(drop=True),
    )


def prepare_video():
    rows = []
    for cls_name, label in [("real", "REAL"), ("fake", "FAKE")]:
        folder = DATASET_DIR / "celebdf_v2" / cls_name
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.mp4")):
            rows.append({
                "video_path": str(p.resolve()),
                "category": "RealVideo-RealAudio" if label == "REAL" else "FakeVideo-RealAudio",
                "identity_id": extract_video_identity(p.name),
                "ethnicity": "unknown",
                "gender": "unknown",
                "video_label": label,
                "audio_label": "REAL",
                "overall_label": label,
                "filename": p.name,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        print("No Celeb-DF v2 videos found.")
        return None
    tr, va, te = split_by_identity(df)
    tr.to_csv(DATA_DIR / "video_train.csv", index=False)
    va.to_csv(DATA_DIR / "video_val.csv", index=False)
    te.to_csv(DATA_DIR / "video_test.csv", index=False)
    print(f"[Video] Celeb-DF v2: {len(df)} clips -> train={len(tr)}, val={len(va)}, test={len(te)}")
    return tr, va, te


def prepare_audio():
    rows = []
    for cls_name, label in [("real", "REAL"), ("fake", "FAKE")]:
        folder = DATASET_DIR / "deepfake_audio" / cls_name
        if not folder.exists():
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() in {".flac", ".wav", ".mp3"}:
                rows.append({
                    "video_path": str(p.resolve()),  # AudioDataset reads 'video_path'
                    "category": "RealVideo-RealAudio" if label == "REAL" else "RealVideo-FakeAudio",
                    "identity_id": extract_audio_identity(p.name),
                    "ethnicity": "unknown",
                    "gender": "unknown",
                    "video_label": "REAL",
                    "audio_label": label,
                    "overall_label": label,
                    "filename": p.name,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        print("No audio clips found.")
        return None
    tr, va, te = split_by_identity(df)
    tr.to_csv(DATA_DIR / "audio_train.csv", index=False)
    va.to_csv(DATA_DIR / "audio_val.csv", index=False)
    te.to_csv(DATA_DIR / "audio_test.csv", index=False)
    print(f"[Audio] Deepfake-Audio: {len(df)} clips -> train={len(tr)}, val={len(va)}, test={len(te)}")
    return tr, va, te


def prepare_text():
    rows = []
    # 1. HC3 English
    hc3_dir = DATASET_DIR / "hc3_english"
    if hc3_dir.exists():
        for fp in sorted(hc3_dir.glob("*.jsonl")):
            with open(fp, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 500:  # cap at 500 Q&A pairs per file for fast training
                        break
                    obj = json.loads(line)
                    for ans in obj.get("human_answers", [])[:1]:
                        if len(ans.strip()) >= 50:
                            rows.append({"text": ans.strip(), "label": 0, "language": "en", "source": f"hc3_{fp.stem}"})
                    for ans in obj.get("chatgpt_answers", [])[:1]:
                        if len(ans.strip()) >= 50:
                            rows.append({"text": ans.strip(), "label": 1, "language": "en", "source": f"hc3_{fp.stem}"})

    # 2. Fakeddit validate sample (user-provided TSV)
    fakeddit_tsv = DATASET_DIR / "all_validate.tsv"
    if fakeddit_tsv.exists():
        fdf = pd.read_csv(fakeddit_tsv, sep="\t", nrows=1000)
        if "clean_title" in fdf.columns and "2_way_label" in fdf.columns:
            for _, r in fdf.dropna(subset=["clean_title", "2_way_label"]).iterrows():
                t = str(r["clean_title"]).strip()
                if len(t) >= 20:
                    rows.append({"text": t, "label": int(r["2_way_label"]), "language": "en", "source": "fakeddit"})

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        n_tr = int(len(df) * 0.85)
        df.iloc[:n_tr].to_csv(DATA_DIR / "text_train.csv", index=False)
        df.iloc[n_tr:].to_csv(DATA_DIR / "text_val.csv", index=False)
        print(f"[Text] HC3 + Fakeddit: {len(df)} samples -> train={n_tr}, val={len(df) - n_tr}")


def main():
    v_splits = prepare_video()
    a_splits = prepare_audio()
    prepare_text()

    # Also write default train/val/test CSVs (video splits by default for compatibility)
    if v_splits is not None:
        tr, va, te = v_splits
        tr.to_csv(DATA_DIR / "train.csv", index=False)
        va.to_csv(DATA_DIR / "val.csv", index=False)
        te.to_csv(DATA_DIR / "test.csv", index=False)


if __name__ == "__main__":
    main()
