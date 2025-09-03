import os
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

from src.preprocess import load_and_split, print_class_distribution
from src.config import CFG, LABEL2ID, ID2LABEL

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def build_tfidf_pipeline():
    """Build TF-IDF pipeline with word + character n-grams and Logistic Regression"""
    print("🔧 Building TF-IDF + Logistic Regression pipeline...")
    
    # Word-level TF-IDF (1-2 n-grams)
    word_tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=200_000,
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        stop_words='english',
        strip_accents='unicode',
        lowercase=True,
        token_pattern=r'\b\w+\b'
    )
    
    # Character-level TF-IDF (3-5 n-grams)
    char_tfidf = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        sublinear_tf=True,
        lowercase=True,
        max_features=None  # No limit on char features
    )
    
    # Combine both vectorizers
    vectorizer = FeatureUnion([
        ("word_tfidf", word_tfidf),
        ("char_tfidf", char_tfidf)
    ])
    
    # Logistic Regression classifier
    classifier = LogisticRegression(
        solver="saga",
        max_iter=5000,
        class_weight="balanced",
        multi_class="multinomial",
        random_state=CFG.SEED,
        n_jobs=1  # Set to 1 to avoid conflicts with GridSearchCV n_jobs
    )
    
    # Complete pipeline
    pipeline = Pipeline([
        ("vectorizer", vectorizer),
        ("classifier", classifier)
    ])
    
    print("✅ Pipeline components:")
    print(f"  • Word TF-IDF: (1-2) n-grams, max_features=200k")
    print(f"  • Char TF-IDF: (3-5) n-grams, unlimited features")
    print(f"  • Classifier: LogisticRegression (multinomial, balanced)")
    
    return pipeline

def train_with_grid_search(pipeline, X_train, y_train):
    """Train pipeline with grid search over regularization parameter"""
    print("\n🏋️ Training with grid search...")
    
    # Parameter grid for C values
    param_grid = {
        "classifier__C": [0.5, 1.0, 2.0]
    }
    
    # Stratified K-Fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=CFG.SEED)
    
    # Grid search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    
    print(f"🔍 Grid search over C = {param_grid['classifier__C']}")
    print(f"📊 Using 5-fold stratified cross-validation")
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Results
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_
    
    print(f"\n✅ Grid search complete!")
    print(f"🏆 Best parameters: {best_params}")
    print(f"📈 Best CV F1-macro: {best_cv_score:.4f}")
    
    return grid_search.best_estimator_, best_params, best_cv_score

def evaluate_on_holdout(pipeline, X_test, y_test):
    """Evaluate trained pipeline on holdout test set"""
    print("\n📊 Evaluating on holdout test set...")
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    
    print(f"🎯 Holdout Results:")
    print(f"  • Accuracy: {accuracy:.4f}")
    print(f"  • F1-macro: {f1_macro:.4f}")
    
    return accuracy, f1_macro

def save_artifacts(pipeline, best_params, best_cv_score, holdout_accuracy, holdout_f1):
    """Save trained pipeline and metadata"""
    print("\n💾 Saving model artifacts...")
    
    # Save complete pipeline
    pipeline_path = MODELS_DIR / "pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)
    print(f"✅ Saved pipeline: {pipeline_path}")
    
    # Save vectorizer separately for compatibility
    vectorizer = pipeline.named_steps["vectorizer"]
    vectorizer_path = MODELS_DIR / "vectorizer.pkl"
    joblib.dump(vectorizer, vectorizer_path)
    print(f"✅ Saved vectorizer: {vectorizer_path}")
    
    # Save classifier separately for compatibility
    classifier = pipeline.named_steps["classifier"]
    classifier_path = MODELS_DIR / "classifier.pkl"
    joblib.dump(classifier, classifier_path)
    print(f"✅ Saved classifier: {classifier_path}")
    
    # Save metadata
    metadata = {
        "model_type": "LogisticRegression",
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "best_params": best_params,
        "cv_f1_macro": float(best_cv_score),
        "holdout_accuracy": float(holdout_accuracy),
        "holdout_f1_macro": float(holdout_f1),
        "feature_config": {
            "word_ngrams": "(1, 2)",
            "char_ngrams": "(3, 5)",
            "word_max_features": 200_000,
            "min_df": 2,
            "max_df": 0.9,
            "sublinear_tf": True,
            "stop_words": "english",
            "strip_accents": "unicode"
        },
        "classifier_config": {
            "solver": "saga",
            "max_iter": 5000,
            "class_weight": "balanced",
            "multi_class": "multinomial"
        }
    }
    
    metadata_path = MODELS_DIR / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata: {metadata_path}")

def main():
    """Main training pipeline"""
    print("🚀 Political Bias Classifier - Logistic Regression Training")
    print("=" * 60)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, _, _ = load_and_split()
    
    # Build pipeline
    pipeline = build_tfidf_pipeline()
    
    # Train with grid search
    best_pipeline, best_params, best_cv_score = train_with_grid_search(pipeline, X_train, y_train)
    
    # Evaluate on holdout
    holdout_accuracy, holdout_f1 = evaluate_on_holdout(best_pipeline, X_test, y_test)
    
    # Save artifacts
    save_artifacts(best_pipeline, best_params, best_cv_score, holdout_accuracy, holdout_f1)
    
    # Final summary
    print(f"\n🎉 Training Complete!")
    print("=" * 40)
    print(f"🏆 Best CV F1-macro: {best_cv_score:.4f}")
    print(f"🎯 Holdout Accuracy: {holdout_accuracy:.4f}")
    print(f"📈 Holdout F1-macro: {holdout_f1:.4f}")
    print(f"⚙️  Best C parameter: {best_params['classifier__C']}")
    print(f"💾 Models saved in: {MODELS_DIR}")
    
    # Performance assessment
    if holdout_f1 > 0.35:
        print("🌟 Excellent performance for this challenging task!")
    elif holdout_f1 > 0.30:
        print("✅ Good performance - solid classical baseline!")
    elif holdout_f1 > 0.25:
        print("👍 Reasonable performance - room for improvement")
    else:
        print("⚠️  Performance below expectations - check data quality")

if __name__ == "__main__":
    main()