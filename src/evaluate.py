import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional

import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize

from src.preprocess import load_and_split
from src.config import CFG, ID2LABEL, LABEL2ID

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = Path("models")

def load_model_and_metadata():
    """Load trained model and metadata"""
    pipeline_path = MODELS_DIR / "pipeline.pkl"
    metadata_path = MODELS_DIR / "model_metadata.json"
    
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Model not found: {pipeline_path}")
    
    pipeline = joblib.load(pipeline_path)
    
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    
    return pipeline, metadata

def evaluate_model_performance(pipeline, X_test, y_test) -> Dict[str, Any]:
    """Comprehensive model evaluation with sanity checks"""
    print("Evaluating model performance...")
    
    # Sanity checks
    print(f"Evaluating on {len(X_test)} validation samples")
    unique_labels, label_counts = np.unique(y_test, return_counts=True)
    print("Validation label distribution:")
    for label_id, count in zip(unique_labels, label_counts):
        label_name = ID2LABEL[label_id]
        print(f"  {label_name}: {count} samples")
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    
    # Try to get prediction probabilities
    try:
        y_pred_proba = pipeline.predict_proba(X_test)
        has_probabilities = True
    except AttributeError:
        y_pred_proba = None
        has_probabilities = False
        print("⚠️  Model doesn't support probability predictions")
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    
    # Sanity check for suspiciously high performance
    if accuracy >= 0.99 or f1_macro >= 0.99:
        print("⚠️  WARNING: Suspiciously high performance detected!")
        print(f"   Accuracy: {accuracy:.4f}, F1-macro: {f1_macro:.4f}")
        print("   This likely indicates data leakage or overfitting.")
    
    # Per-class metrics
    precision, recall, f1_scores, support = precision_recall_fscore_support(
        y_test, y_pred, average=None
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=sorted(LABEL2ID.values()))
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-macro: {f1_macro:.4f}")
    print(f"F1-weighted: {f1_weighted:.4f}")
    
    # Save sample validation predictions for inspection
    sample_size = min(25, len(X_test))
    sample_indices = np.random.RandomState(42).choice(len(X_test), sample_size, replace=False)
    
    sample_data = []
    for idx in sample_indices:
        sample_data.append({
            "input_text": X_test[idx][:200] + "..." if len(X_test[idx]) > 200 else X_test[idx],
            "gold_label": ID2LABEL[y_test[idx]],
            "pred_label": ID2LABEL[y_pred[idx]]
        })
    
    import pandas as pd
    sample_df = pd.DataFrame(sample_data)
    sample_path = REPORTS_DIR / "sample_val_preds.csv"
    sample_df.to_csv(sample_path, index=False)
    print(f"Sample predictions saved to {sample_path}")
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1_scores": f1_scores.tolist(),
        "support": support.tolist(),
        "confusion_matrix": cm.tolist(),
        "predictions": y_pred.tolist(),
        "probabilities": y_pred_proba.tolist() if has_probabilities else None,
        "has_probabilities": has_probabilities
    }

def save_classification_report(y_test, y_pred):
    """Save detailed classification report"""
    target_names = [ID2LABEL[i] for i in sorted(ID2LABEL.keys())]
    
    report = classification_report(
        y_test, y_pred, 
        target_names=target_names, 
        digits=4,
        zero_division=0
    )
    
    report_path = REPORTS_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("Enhanced Political Bias Classifier - Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    
    print(f"Saved classification report: {report_path}")
    return report

def plot_confusion_matrix(cm, save_path: Optional[str] = None):
    """Plot and save confusion matrix"""
    target_names = [ID2LABEL[i] for i in sorted(ID2LABEL.keys())]
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=target_names,
        yticklabels=target_names,
        cbar_kws={'label': 'Number of Samples'}
    )
    
    plt.title('Confusion Matrix - Enhanced Political Bias Classifier', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Add accuracy text
    accuracy = np.trace(cm) / np.sum(cm)
    plt.text(1.5, -0.1, f'Overall Accuracy: {accuracy:.1%}', 
             transform=plt.gca().transAxes, ha='center', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix: {save_path}")
    
    plt.close()

def plot_class_support_chart(support, save_path: Optional[str] = None):
    """Plot class support distribution"""
    target_names = [ID2LABEL[i] for i in sorted(ID2LABEL.keys())]
    
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(target_names, support, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    plt.title('Test Set Class Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Political Bias', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    
    # Add value labels on bars
    for bar, value in zip(bars, support):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved class support chart: {save_path}")
    
    plt.close()

def plot_roc_curves(y_test, y_pred_proba, save_path: Optional[str] = None):
    """Plot ROC curves for each class"""
    if y_pred_proba is None:
        print("⚠️  Cannot plot ROC curves - no probability predictions available")
        return
    
    # Binarize labels for multiclass ROC
    y_test_bin = label_binarize(y_test, classes=sorted(LABEL2ID.values()))
    n_classes = len(LABEL2ID)
    
    plt.figure(figsize=(12, 8))
    
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    
    for i, (class_id, class_name) in enumerate(sorted(ID2LABEL.items())):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Political Bias Classification', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved ROC curves: {save_path}")
    
    plt.close()

def plot_precision_recall_curves(y_test, y_pred_proba, save_path: Optional[str] = None):
    """Plot Precision-Recall curves for each class"""
    if y_pred_proba is None:
        print("⚠️  Cannot plot PR curves - no probability predictions available")
        return
    
    # Binarize labels for multiclass PR curves
    y_test_bin = label_binarize(y_test, classes=sorted(LABEL2ID.values()))
    
    plt.figure(figsize=(12, 8))
    
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    
    for i, (class_id, class_name) in enumerate(sorted(ID2LABEL.items())):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
        pr_auc = auc(recall, precision)
        
        plt.plot(recall, precision, color=colors[i], lw=2,
                label=f'{class_name} (AUC = {pr_auc:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves - Political Bias Classification', 
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved PR curves: {save_path}")
    
    plt.close()

def extract_top_tokens_for_linear_model(pipeline, model_name: str):
    """Extract and save top tokens for linear models"""
    print(f"\n🔍 Extracting top tokens for {model_name}...")
    
    # Check if we can extract coefficients
    classifier = pipeline.named_steps["classifier"]
    
    # Handle different classifier types
    if hasattr(classifier, "coef_"):
        coef = classifier.coef_
    elif hasattr(classifier, "estimator") and hasattr(classifier.estimator, "coef_"):
        coef = classifier.estimator.coef_
    else:
        print("⚠️  Cannot extract coefficients from this model type")
        return
    
    # Get feature names from vectorizer
    vectorizer = pipeline.named_steps["vectorizer"]
    
    # Handle FeatureUnion (word + char n-grams)
    if hasattr(vectorizer, "transformer_list"):
        word_features = vectorizer.named_transformers["word_tfidf"].get_feature_names_out()
        char_features = vectorizer.named_transformers["char_tfidf"].get_feature_names_out()
        
        # Combine feature names
        feature_names = np.concatenate([
            [f"word:{feat}" for feat in word_features],
            [f"char:{feat}" for feat in char_features]
        ])
    else:
        feature_names = vectorizer.get_feature_names_out()
    
    # Extract top tokens for each class
    n_top = 20
    
    for class_id, class_name in ID2LABEL.items():
        class_coef = coef[class_id]
        
        # Get top positive and negative coefficients
        top_positive_idx = np.argsort(class_coef)[-n_top:][::-1]
        top_negative_idx = np.argsort(class_coef)[:n_top]
        
        # Save to file
        tokens_file = REPORTS_DIR / f"top_tokens_{class_name.upper()}.txt"
        
        with open(tokens_file, "w") as f:
            f.write(f"Top Tokens for {class_name.upper()} Bias Classification\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("TOP POSITIVE INDICATORS (most predictive of this class):\n")
            f.write("-" * 50 + "\n")
            for i, idx in enumerate(top_positive_idx, 1):
                token = feature_names[idx]
                coeff = class_coef[idx]
                f.write(f"{i:2d}. {token:30s} ({coeff:+.4f})\n")
            
            f.write("\nTOP NEGATIVE INDICATORS (most predictive against this class):\n")
            f.write("-" * 50 + "\n")
            for i, idx in enumerate(top_negative_idx, 1):
                token = feature_names[idx]
                coeff = class_coef[idx]
                f.write(f"{i:2d}. {token:30s} ({coeff:+.4f})\n")
        
        print(f"Saved top tokens for {class_name}: {tokens_file}")

def save_comprehensive_metrics(metrics: Dict[str, Any], metadata: Dict[str, Any]):
    """Save comprehensive metrics to JSON"""
    target_names = [ID2LABEL[i] for i in sorted(ID2LABEL.keys())]
    
    comprehensive_metrics = {
        "model_info": {
            "best_model": metadata.get("best_model", "Unknown"),
            "cv_f1_macro_mean": metadata.get("cv_f1_macro_mean", 0.0),
            "cv_f1_macro_std": metadata.get("cv_f1_macro_std", 0.0),
            "use_smote": metadata.get("use_smote", False),
            "config": metadata.get("config", {})
        },
        "test_metrics": {
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "has_probabilities": metrics["has_probabilities"]
        },
        "per_class_metrics": {
            target_names[i]: {
                "precision": metrics["precision"][i],
                "recall": metrics["recall"][i],
                "f1_score": metrics["f1_scores"][i],
                "support": metrics["support"][i]
            }
            for i in range(len(target_names))
        },
        "confusion_matrix": metrics["confusion_matrix"],
        "class_labels": target_names
    }
    
    metrics_path = REPORTS_DIR / "comprehensive_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(comprehensive_metrics, f, indent=2)
    
    print(f"Saved comprehensive metrics: {metrics_path}")

def main():
    """Main evaluation pipeline"""
    print("📈 Enhanced Political Bias Classifier Evaluation")
    print("=" * 50)
    
    # Load data and model
    X_train, X_test, y_train, y_test, _, _ = load_and_split()
    pipeline, metadata = load_model_and_metadata()
    
    print(f"Loaded model: {metadata.get('best_model', 'Unknown')}")
    print(f"Test set size: {len(X_test)}")
    
    # Evaluate model
    metrics = evaluate_model_performance(pipeline, X_test, y_test)
    
    # Generate and save reports
    print("\n📄 Generating detailed reports...")
    
    # Classification report
    report = save_classification_report(y_test, metrics["predictions"])
    print(report)
    
    # Confusion matrix
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]), 
        REPORTS_DIR / "confusion_matrix.png"
    )
    
    # Class support chart
    plot_class_support_chart(
        metrics["support"], 
        REPORTS_DIR / "class_support.png"
    )
    
    # ROC and PR curves (if probabilities available)
    if metrics["has_probabilities"]:
        plot_roc_curves(
            y_test, 
            np.array(metrics["probabilities"]), 
            REPORTS_DIR / "roc_curves.png"
        )
        
        plot_precision_recall_curves(
            y_test, 
            np.array(metrics["probabilities"]), 
            REPORTS_DIR / "precision_recall_curves.png"
        )
    
    # Extract top tokens for linear models
    model_name = metadata.get("best_model", "")
    if "Logistic" in model_name or "SVC" in model_name:
        extract_top_tokens_for_linear_model(pipeline, model_name)
    
    # Save comprehensive metrics
    save_comprehensive_metrics(metrics, metadata)
    
    # Final summary
    print(f"\n✅ Evaluation complete!")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1-macro: {metrics['f1_macro']:.4f}")
    print(f"CV F1-macro: {metadata.get('cv_f1_macro_mean', 0):.4f} ± {metadata.get('cv_f1_macro_std', 0):.4f}")
    print(f"Reports saved in: {REPORTS_DIR}")

if __name__ == "__main__":
    main()