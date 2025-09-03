import os
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

from src.config import CFG, LABEL2ID, ID2LABEL

# Regex patterns for text cleaning
_url = re.compile(r"https?://\S+|www\.\S+")
_email = re.compile(r"\S+@\S+")
_nonword = re.compile(r"[^A-Za-z0-9\s]+")
_whitespace = re.compile(r"\s+")

def _clean_text(text: str) -> str:
    """Minimal text cleaning to match create_split.py exactly"""
    if not text or pd.isna(text):
        return ""
    
    text = str(text)
    
    # Only remove URLs (matching create_split.py)
    text = _url.sub(" ", text)
    
    # Only normalize whitespace - preserve punctuation, case, and all other features
    text = _whitespace.sub(" ", text).strip()
    
    return text

def _combine_text_fields(df: pd.DataFrame) -> pd.Series:
    """Combine text fields with guardrails against label leakage (matching create_split.py logic)"""
    combined_texts = []
    
    # Assert no label or source columns are used in text assembly
    forbidden_cols = ["bias", "bias_rating", "source", "outlet", "publisher"]
    for col in forbidden_cols:
        if col in df.columns:
            print(f"WARNING: Found column '{col}' - will NOT use in text assembly")
    
    for _, row in df.iterrows():
        text_parts = []
        
        # Use only title and heading first (matching create_split.py)
        for col in ["title", "heading"]:
            if col in df.columns and pd.notna(row[col]) and str(row[col]).strip():
                cleaned = _clean_text(row[col])
                if cleaned:
                    text_parts.append(cleaned)
        
        combined_so_far = " ".join(text_parts)
        
        # If very short, fallback to full text (matching create_split.py)
        if len(combined_so_far) < 10 and "text" in df.columns:
            if pd.notna(row["text"]) and str(row["text"]).strip():
                full_text = _clean_text(row["text"])
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

def print_class_distribution(y, title="Class Distribution"):
    """Print detailed class distribution"""
    counter = Counter(y)
    total = len(y)
    
    print(f"\n{title}:")
    print("-" * 40)
    for class_id in sorted(counter.keys()):
        class_name = ID2LABEL[class_id]
        count = counter[class_id]
        percentage = (count / total) * 100
        print(f"{class_name:>6}: {count:>6} ({percentage:>5.1f}%)")
    print(f"{'Total':>6}: {total:>6} (100.0%)")

def apply_smote_if_enabled(X_train, y_train):
    """Apply SMOTE oversampling if enabled"""
    if not CFG.USE_SMOTE:
        return X_train, y_train
    
    print("\n🔄 Applying SMOTE oversampling...")
    print_class_distribution(y_train, "Before SMOTE")
    
    # SMOTE requires numerical features, so we need to vectorize first
    # This is a limitation - SMOTE will be applied after vectorization in train.py
    print("⚠️  SMOTE will be applied after vectorization in training phase")
    
    return X_train, y_train

def load_and_split():
    """Load data using persistent train/validation split to prevent data leakage"""
    print("Loading data using persistent split...")
    
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
    csv_file = splits_data.get("csv_file", CFG.CSV_PATH)
    
    print(f"Using persistent split: {len(train_idx)} train, {len(val_idx)} val samples")
    
    # Load and preprocess full dataset (matching create_split.py logic)
    df = pd.read_csv(csv_file)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Filter to valid labels
    initial_size = len(df)
    df = df[df[CFG.LABEL_COL].isin(LABEL2ID)].copy()
    print(f"Filtered to valid labels: {len(df)} rows (removed {initial_size - len(df)})")
    
    # Combine text fields with guardrails
    print(f"Combining text fields: {CFG.TEXT_COLS}")
    df["combined_text"] = _combine_text_fields(df)
    
    # Filter very short texts
    initial_combined = len(df)
    df = df[df["combined_text"].str.len() >= 5].copy()
    print(f"Filtered short texts: {len(df)} rows (removed {initial_combined - len(df)})")
    
    # Global deduplication (should match create_split.py exactly)
    initial_unique = len(df)
    # Use input_text column name to match create_split.py
    df["input_text"] = df["combined_text"]
    df = df.drop_duplicates(subset="input_text", keep="first").reset_index(drop=True)
    print(f"After deduplication: {len(df)} rows (removed {initial_unique - len(df)})")
    
    # Verify split integrity
    assert len(df) == splits_data["total_samples"], f"Dataset size mismatch: {len(df)} vs {splits_data['total_samples']}"
    assert len(set(train_idx) & set(val_idx)) == 0, "Train/val index overlap detected!"
    assert max(train_idx + val_idx) < len(df), "Index out of bounds!"
    
    print("✅ Split integrity verified")
    
    # Split using persistent indices
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    
    # Prepare features and labels
    X_train = train_df["input_text"].tolist()
    X_test = val_df["input_text"].tolist()
    y_train = train_df[CFG.LABEL_COL].map(LABEL2ID).values
    y_test = val_df[CFG.LABEL_COL].map(LABEL2ID).values
    
    print(f"\nTrain/Validation Split:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_test)} samples")
    
    print_class_distribution(y_train, "Training Set Distribution")
    print_class_distribution(y_test, "Validation Set Distribution")
    
    # Apply SMOTE if enabled
    X_train, y_train = apply_smote_if_enabled(X_train, y_train)
    
    return X_train, X_test, y_train, y_test, LABEL2ID, ID2LABEL

def main():
    """Test preprocessing pipeline"""
    X_train, X_test, y_train, y_test, label2id, id2label = load_and_split()
    
    print(f"\n✅ Preprocessing complete!")
    print(f"Final training size: {len(X_train)}")
    print(f"Final test size: {len(X_test)}")
    
    # Sample text lengths
    train_lengths = [len(text.split()) for text in X_train[:1000]]
    print(f"Sample text lengths (words): min={min(train_lengths)}, max={max(train_lengths)}, avg={np.mean(train_lengths):.1f}")

if __name__ == "__main__":
    main()