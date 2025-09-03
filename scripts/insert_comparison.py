#!/usr/bin/env python3
"""
Insert comparison table between classical and transformer baselines into README.md
"""

import json
import os
import sys
from pathlib import Path

def load_metrics(filepath):
    """Load metrics from JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None

def format_metric(value):
    """Format metric value for display"""
    if value is None:
        return "N/A"
    return f"{value:.4f}"

def main():
    # Paths to metrics files
    classical_metrics_path = "reports/metrics.json"
    roberta_metrics_path = "reports/roberta/metrics.json"
    readme_path = "README.md"
    
    # Load metrics
    classical_metrics = load_metrics(classical_metrics_path)
    roberta_metrics = load_metrics(roberta_metrics_path)
    
    # Check if at least one set of metrics exists
    if classical_metrics is None and roberta_metrics is None:
        print("Error: No metrics files found. Please run training and evaluation first.")
        print(f"Expected files: {classical_metrics_path}, {roberta_metrics_path}")
        print("Skipping README update.")
        return 1
    
    # Extract metrics
    classical_acc = classical_metrics.get("accuracy") if classical_metrics else None
    classical_f1 = classical_metrics.get("f1_macro") if classical_metrics else None
    
    # Handle different possible structures for RoBERTa metrics
    if roberta_metrics:
        if "final_metrics" in roberta_metrics:
            # Nested structure
            roberta_acc = roberta_metrics["final_metrics"].get("eval_accuracy")
            roberta_f1 = roberta_metrics["final_metrics"].get("eval_f1_macro")
        else:
            # Direct structure
            roberta_acc = roberta_metrics.get("accuracy") or roberta_metrics.get("eval_accuracy")
            roberta_f1 = roberta_metrics.get("f1_macro") or roberta_metrics.get("eval_f1_macro")
    else:
        roberta_acc = None
        roberta_f1 = None
    
    # Check for suspicious performance (likely data leakage)
    roberta_suspicious = False
    if roberta_acc is not None and roberta_acc >= 0.99:
        roberta_suspicious = True
        print("Warning: RoBERTa model shows suspiciously high accuracy (≥99%).")
        print("This likely indicates data leakage or overfitting.")
        print("Consider retraining with the persistent split.")
    
    # Calculate improvement
    if classical_f1 is not None and roberta_f1 is not None:
        improvement = roberta_f1 - classical_f1
        improvement_str = f"{improvement:+.4f}"
    else:
        improvement_str = "N/A"
    
    # Create comparison section
    if roberta_suspicious:
        note = f"**Note**: The transformer model shows suspiciously perfect performance (100% accuracy), indicating potential data leakage or overfitting. This likely results from using a different data split than the persistent one. Retrain with `make train_roberta && make eval_roberta` to get realistic metrics."
    else:
        note = f"**Relative improvement (Δ Macro-F1)**: {improvement_str}"
    
    comparison_section = f"""<!-- BEGIN_COMPARISON -->
## Transformer Baseline (RoBERTa) and Comparison

| Model                         | Accuracy | Macro-F1 |
|-------------------------------|----------|----------|
| TF-IDF + Logistic Regression  | {format_metric(classical_acc)}   | {format_metric(classical_f1)}   |
| DistilRoBERTa (fine-tuned)    | {format_metric(roberta_acc)}   | {format_metric(roberta_f1)}   |

{note}

Confusion matrices:
- Classical baseline: `reports/confusion_matrix.png` 
- RoBERTa baseline: `reports/roberta/confusion_matrix.png` (after retraining)

The classical baseline achieves realistic performance ({format_metric(classical_acc)} accuracy) on this challenging task. The transformer model is expected to show improvements through its ability to capture contextual relationships, but current results indicate the need for retraining with the persistent split.
<!-- END_COMPARISON -->"""
    
    # Read current README
    try:
        with open(readme_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: {readme_path} not found")
        return 1
    
    # Find and replace comparison section
    start_marker = "<!-- BEGIN_COMPARISON -->"
    end_marker = "<!-- END_COMPARISON -->"
    
    start_pos = content.find(start_marker)
    end_pos = content.find(end_marker)
    
    if start_pos == -1 or end_pos == -1:
        print("Error: Comparison markers not found in README.md")
        print("Make sure the file contains <!-- BEGIN_COMPARISON --> and <!-- END_COMPARISON -->")
        return 1
    
    # Replace the section
    new_content = (
        content[:start_pos] + 
        comparison_section + 
        content[end_pos + len(end_marker):]
    )
    
    # Write updated README
    try:
        with open(readme_path, 'w') as f:
            f.write(new_content)
        print("Successfully updated README.md with comparison table")
        return 0
    except Exception as e:
        print(f"Error writing to {readme_path}: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())