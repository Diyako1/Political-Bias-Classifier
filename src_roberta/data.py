import os
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from src_roberta.config import TCFG

# Minimal cleaning patterns - only URLs and whitespace
_url = re.compile(r"https?://\S+|www\.\S+")
_whitespace = re.compile(r"\s+")

# Label mapping
LABEL2ID = {"left": 0, "center": 1, "right": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def clean_text_minimal(text: str) -> str:
    """Minimal text cleaning - only URLs and whitespace (no lowercasing or punctuation removal)"""
    if not text or pd.isna(text):
        return ""
    
    text = str(text)
    
    # Only remove URLs
    text = _url.sub(" ", text)
    
    # Only normalize whitespace - preserve punctuation, case, and all other features
    text = _whitespace.sub(" ", text).strip()
    
    return text

def combine_text_fields_enhanced(df: pd.DataFrame) -> pd.Series:
    """Text field combination with guardrails against label leakage"""
    combined_texts = []
    
    # Assert no label or source columns are used in text assembly
    forbidden_cols = ["bias", "bias_rating", "source", "outlet", "publisher"]
    for col in forbidden_cols:
        if col in df.columns:
            print(f"WARNING: Found column '{col}' - will NOT use in text assembly")
    
    for _, row in df.iterrows():
        text_parts = []
        
        # Use title and heading first
        for col in ["title", "heading"]:
            if col in df.columns and pd.notna(row[col]) and str(row[col]).strip():
                cleaned = clean_text_minimal(row[col])
                if cleaned:
                    text_parts.append(cleaned)
        
        combined_so_far = " ".join(text_parts)
        
        # If very short, fallback to full text
        if len(combined_so_far) < 10 and "text" in df.columns:
            if pd.notna(row["text"]) and str(row["text"]).strip():
                full_text = clean_text_minimal(row["text"])
                if full_text:
                    if combined_so_far:
                        combined_text = combined_so_far + " " + full_text
                    else:
                        combined_text = full_text
                else:
                    combined_text = combined_so_far
            else:
                combined_text = combined_so_far
        else:
            combined_text = combined_so_far
        
        combined_texts.append(combined_text)
    
    return pd.Series(combined_texts)

def load_and_split():
    """Load data using persistent train/validation split to prevent data leakage"""
    print("Loading data using persistent split for RoBERTa...")
    
    # Load persistent split indices
    splits_path = "data/splits.json"
    if not Path(splits_path).exists():
        raise FileNotFoundError(
            f"{splits_path} not found. Run 'make split' first to create persistent split."
        )
    
    with open(splits_path, "r") as f:
        splits_data = json.load(f)
    
    train_idx = splits_data["train_idx"]
    val_idx = splits_data["val_idx"]
    csv_file = splits_data.get("csv_file", TCFG.CSV_PATH)
    
    print(f"Using persistent split: {len(train_idx)} train, {len(val_idx)} val samples")
    
    # Load and preprocess full dataset (matching create_split.py logic)
    df = pd.read_csv(csv_file)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Filter to valid labels
    initial_size = len(df)
    df = df[df[TCFG.LABEL_COL].isin(LABEL2ID)].copy()
    print(f"Filtered to valid labels: {len(df)} rows (removed {initial_size - len(df)})")
    
    # Combine text fields with guardrails
    print(f"Combining text fields with guardrails...")
    df["input_text"] = combine_text_fields_enhanced(df)
    
    # Filter very short texts
    initial_combined = len(df)
    df = df[df["input_text"].str.len() >= 5].copy()
    print(f"Filtered short texts: {len(df)} rows (removed {initial_combined - len(df)})")
    
    # Global deduplication (should match create_split.py)
    initial_unique = len(df)
    df = df.drop_duplicates(subset="input_text", keep="first").reset_index(drop=True)
    print(f"After deduplication: {len(df)} rows (removed {initial_unique - len(df)})")
    
    # Verify split integrity
    assert len(df) == splits_data["total_samples"], f"Dataset size mismatch: {len(df)} vs {splits_data['total_samples']}"
    assert len(set(train_idx) & set(val_idx)) == 0, "Train/val index overlap detected!"
    assert max(train_idx + val_idx) < len(df), "Index out of bounds!"
    
    print("✅ Split integrity verified")
    
    # Map labels to integers
    df["label"] = df[TCFG.LABEL_COL].map(LABEL2ID)
    
    # Split using persistent indices
    train_df = df.iloc[train_idx][["input_text", "label"]].copy()
    val_df = df.iloc[val_idx][["input_text", "label"]].copy()
    
    print(f"\nTrain/Validation Split:")
    print(f"Training set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    
    # Print class distribution
    print("\nTraining set distribution:")
    for label_name, label_id in LABEL2ID.items():
        count = (train_df["label"] == label_id).sum()
        pct = count / len(train_df) * 100
        print(f"  {label_name}: {count} ({pct:.1f}%)")
    
    print("\nValidation set distribution:")
    for label_name, label_id in LABEL2ID.items():
        count = (val_df["label"] == label_id).sum()
        pct = count / len(val_df) * 100
        print(f"  {label_name}: {count} ({pct:.1f}%)")
    
    # Print sample text lengths
    train_lengths = [len(text.split()) for text in train_df["input_text"].head(100)]
    print(f"Sample text lengths (words): min={min(train_lengths)}, max={max(train_lengths)}, avg={np.mean(train_lengths):.1f}")
    
    return train_df, val_df

def main():
    """Test enhanced data loading"""
    train_df, val_df = load_and_split()
    
    print(f"\nSample training texts (enhanced preprocessing):")
    for i, row in train_df.head(3).iterrows():
        label_name = ID2LABEL[row["label"]]
        text_preview = row["input_text"][:150] + "..." if len(row["input_text"]) > 150 else row["input_text"]
        print(f"  {label_name}: {text_preview}")

if __name__ == "__main__":
    main()