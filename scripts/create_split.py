#!/usr/bin/env python3
"""
Create persistent train/validation split to eliminate data leakage.
Generates data/splits.json with stratified 80/20 split (seed=42).
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import re

# Text cleaning patterns
_url = re.compile(r"https?://\S+|www\.\S+")
_whitespace = re.compile(r"\s+")

def clean_text_minimal(text: str) -> str:
    """Minimal text cleaning - only URLs and whitespace"""
    if not text or pd.isna(text):
        return ""
    
    text = str(text)
    # Remove URLs
    text = _url.sub(" ", text)
    # Normalize whitespace only
    text = _whitespace.sub(" ", text).strip()
    return text

def combine_text_fields(df: pd.DataFrame) -> pd.Series:
    """Combine text fields with guardrails against label leakage"""
    combined_texts = []
    
    # Define allowed text columns (NO label or source columns)
    allowed_cols = ["title", "heading", "text"]
    
    # Assert no label columns are used
    label_cols = ["bias", "bias_rating", "source", "outlet", "publisher"]
    for col in label_cols:
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
        
        # If very short, fallback to full text
        combined_so_far = " ".join(text_parts)
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

def main():
    """Create persistent train/validation split"""
    print("Creating persistent train/validation split...")
    
    # Try both possible CSV files
    csv_files = ["data/News-Data.csv", "data/allsides_news_complete.csv"]
    df = None
    
    for csv_file in csv_files:
        if Path(csv_file).exists():
            print(f"Loading {csv_file}...")
            df = pd.read_csv(csv_file)
            break
    
    if df is None:
        raise FileNotFoundError(f"No CSV file found. Tried: {csv_files}")
    
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Filter to valid labels
    label_col = "bias_rating" if "bias_rating" in df.columns else "bias"
    valid_labels = {"left", "center", "right"}
    
    initial_size = len(df)
    df = df[df[label_col].isin(valid_labels)].copy()
    print(f"Filtered to valid labels: {len(df)} rows (removed {initial_size - len(df)})")
    
    # Combine text fields with guardrails
    print("Assembling text with guardrails...")
    df["input_text"] = combine_text_fields(df)
    
    # Filter very short texts
    initial_combined = len(df)
    df = df[df["input_text"].str.len() >= 5].copy()
    print(f"Filtered short texts: {len(df)} rows (removed {initial_combined - len(df)})")
    
    # CRITICAL: Global deduplication BEFORE splitting
    print("Performing global deduplication...")
    initial_unique = len(df)
    df = df.drop_duplicates(subset="input_text", keep="first").reset_index(drop=True)
    print(f"After deduplication: {len(df)} rows (removed {initial_unique - len(df)} duplicates)")
    
    # Map labels to integers
    label_map = {"left": 0, "center": 1, "right": 2}
    df["label"] = df[label_col].map(label_map)
    
    # Print class distribution
    print("\nClass distribution in full dataset:")
    for label_name, label_id in label_map.items():
        count = (df["label"] == label_id).sum()
        pct = count / len(df) * 100
        print(f"  {label_name}: {count} ({pct:.1f}%)")
    
    # Stratified split with fixed seed
    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
        shuffle=True
    )
    
    # Convert to regular Python lists for JSON serialization
    train_idx = train_idx.tolist()
    val_idx = val_idx.tolist()
    
    # Verification
    assert len(set(train_idx) & set(val_idx)) == 0, "Train/val index overlap detected!"
    assert len(train_idx) + len(val_idx) == len(df), "Index count mismatch!"
    
    print(f"\nSplit created:")
    print(f"  Training indices: {len(train_idx)} samples")
    print(f"  Validation indices: {len(val_idx)} samples")
    print(f"  Index overlap: {len(set(train_idx) & set(val_idx))} (should be 0)")
    
    # Print class distribution per split
    train_labels = df.iloc[train_idx]["label"]
    val_labels = df.iloc[val_idx]["label"]
    
    print(f"\nTraining split distribution:")
    for label_name, label_id in label_map.items():
        count = (train_labels == label_id).sum()
        pct = count / len(train_labels) * 100
        print(f"  {label_name}: {count} ({pct:.1f}%)")
    
    print(f"\nValidation split distribution:")
    for label_name, label_id in label_map.items():
        count = (val_labels == label_id).sum()
        pct = count / len(val_labels) * 100
        print(f"  {label_name}: {count} ({pct:.1f}%)")
    
    # Save split indices
    splits_data = {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "total_samples": len(df),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "seed": 42,
        "label_column": label_col,
        "csv_file": csv_file
    }
    
    splits_path = "data/splits.json"
    Path("data").mkdir(exist_ok=True)
    
    with open(splits_path, "w") as f:
        json.dump(splits_data, f, indent=2)
    
    print(f"\nSplit indices saved to {splits_path}")
    print("✅ Persistent split created successfully!")

if __name__ == "__main__":
    main()
