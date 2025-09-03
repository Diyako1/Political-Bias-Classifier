import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from src_roberta.config import TCFG
from src_roberta.data import load_and_split, ID2LABEL, LABEL2ID

def setup_device():
    """Auto-detect the best available device"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon MPS acceleration")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    return device

def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    if not os.path.exists(TCFG.OUTPUT_DIR):
        raise FileNotFoundError(f"Model directory not found: {TCFG.OUTPUT_DIR}")
    
    print(f"Loading model from {TCFG.OUTPUT_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(TCFG.OUTPUT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(TCFG.OUTPUT_DIR)
    
    return model, tokenizer

def create_eval_dataset(val_df, tokenizer):
    """Create tokenized dataset for evaluation"""
    def tokenize_function(examples):
        return tokenizer(
            examples["input_text"],
            truncation=True,
            padding=True,
            max_length=TCFG.MAX_LEN
        )
    
    # Convert to Hugging Face dataset
    val_dataset = Dataset.from_pandas(val_df)
    
    # Tokenize
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    return val_dataset

def predict_batch(model, dataset, device, batch_size=32):
    """Make predictions on the dataset"""
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    # Create DataLoader with proper collate function
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        # Extract input_ids, attention_mask, and labels
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['label'] for item in batch]
        
        # Convert tensors to lists for padding
        input_ids_list = [ids.tolist() if hasattr(ids, 'tolist') else ids for ids in input_ids]
        attention_mask_list = [mask.tolist() if hasattr(mask, 'tolist') else mask for mask in attention_mask]
        
        # Pad sequences
        max_len = max(len(ids) for ids in input_ids_list)
        
        padded_input_ids = []
        padded_attention_mask = []
        
        for ids, mask in zip(input_ids_list, attention_mask_list):
            padding_length = max_len - len(ids)
            padded_input_ids.append(ids + [0] * padding_length)
            padded_attention_mask.append(mask + [0] * padding_length)
        
        return {
            'input_ids': torch.tensor(padded_input_ids),
            'attention_mask': torch.tensor(padded_attention_mask),
            'label': torch.tensor(labels)
        }
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    print(f"Making predictions on {len(dataset)} samples...")
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get predictions and probabilities
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities)
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)

def compute_metrics(y_true, y_pred, y_proba):
    """Compute comprehensive evaluation metrics with sanity checks"""
    
    # Sanity checks
    print(f"Evaluating on {len(y_true)} validation samples")
    print(f"Unique labels in validation: {np.unique(y_true, return_counts=True)}")
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    
    # Sanity check for suspicious performance
    if accuracy >= 0.99 or f1_macro >= 0.99:
        print("WARNING: metrics suspiciously high; check for leakage.")
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-macro: {f1_macro:.4f}")
    print(f"F1-weighted: {f1_weighted:.4f}")
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "labels": list(ID2LABEL.values()),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

def save_confusion_matrix(y_true, y_pred, save_path):
    """Save confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=list(ID2LABEL.values()),
                yticklabels=list(ID2LABEL.values()))
    plt.title('Confusion Matrix - RoBERTa Political Bias Classification')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def save_sample_predictions(val_df, y_true, y_pred, save_path):
    """Save sample predictions for error analysis"""
    import random
    
    # Create DataFrame with predictions
    results_df = val_df.copy()
    results_df["gold_label"] = [ID2LABEL[label] for label in y_true]
    results_df["pred_label"] = [ID2LABEL[label] for label in y_pred]
    
    # Select 25 random samples
    sample_size = min(25, len(results_df))
    sample_indices = random.sample(range(len(results_df)), sample_size)
    sample_df = results_df.iloc[sample_indices][["input_text", "gold_label", "pred_label"]]
    
    # Save to CSV
    sample_df.to_csv(save_path, index=False)
    print(f"Sample predictions saved to {save_path}")
    
    # Print some examples
    print("\nSample predictions:")
    for i, row in sample_df.head(5).iterrows():
        text_preview = row["input_text"][:100] + "..." if len(row["input_text"]) > 100 else row["input_text"]
        print(f"  Gold: {row['gold_label']}, Pred: {row['pred_label']}")
        print(f"  Text: {text_preview}")
        print()

def main():
    """Main evaluation pipeline"""
    print("RoBERTa Political Bias Classifier Evaluation")
    print("=" * 60)
    
    # Setup device
    device = setup_device()
    
    # Create reports directory
    Path(TCFG.REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load data using persistent split
    print("Loading data using persistent split...")
    train_df, val_df = load_and_split()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Create evaluation dataset
    print("Creating evaluation dataset...")
    val_dataset = create_eval_dataset(val_df, tokenizer)
    
    # Make predictions
    y_pred, y_true, y_proba = predict_batch(model, val_dataset, device, batch_size=16)
    
    # Compute metrics
    print("\nComputing evaluation metrics...")
    metrics = compute_metrics(y_true, y_pred, y_proba)
    
    # Save metrics
    metrics_path = f"{TCFG.REPORTS_DIR}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Save classification report
    report = classification_report(y_true, y_pred, target_names=list(ID2LABEL.values()))
    report_path = f"{TCFG.REPORTS_DIR}/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")
    
    # Save confusion matrix
    cm_path = f"{TCFG.REPORTS_DIR}/confusion_matrix.png"
    save_confusion_matrix(y_true, y_pred, cm_path)
    
    # Save sample predictions
    sample_path = f"{TCFG.REPORTS_DIR}/sample_val_preds.csv"
    save_sample_predictions(val_df, y_true, y_pred, sample_path)
    
    print(f"\nEvaluation complete!")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1-macro: {metrics['f1_macro']:.4f}")
    print(f"F1-weighted: {metrics['f1_weighted']:.4f}")

if __name__ == "__main__":
    main()