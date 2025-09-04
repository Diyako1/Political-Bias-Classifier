# Classifying News Headlines as Left, Center, or Right

## Abstract

Political bias in news media is a growing concern, with different outlets often framing the same events in strikingly different ways. This project explores how machine learning can detect these biases by classifying news headlines into left, center, or right categories.

I built two different approaches: a simple baseline using word frequencies and logistic regression, and a more sophisticated model using DistilRoBERTa. The goal was to see how much modern transformer models improve over traditional methods on this specific task. Both models use the same dataset and evaluation setup for a fair comparison.

## Acknowledgements

This work was inspired by the Stanford CS224N project on bias detection. Unlike their broad survey of multiple NLP models, my project focuses on a direct comparison between a classical TF-IDF + logistic regression baseline and a transformer model (DistilRoBERTa) on labeled news headlines.

Link: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1224/reports/custom_116661041.pdf

## 1. Introduction

Detecting political bias in news text is tricky. Headlines are short, so they rely heavily on word choice to convey stance. The same vocabulary appears across political perspectives, making simple keyword matching unreliable. Plus, what counts as "biased" is subjective - different people might label the same headline differently.

I started with a simple baseline to understand what patterns the models are actually learning. This helps interpret the results and see where each approach succeeds or fails.

## 2. Related Work

Most previous work on bias detection uses word frequency counts combined with simple classifiers like logistic regression. These methods are interpretable but miss the context that makes words meaningful. Recent transformer models like BERT and RoBERTa can capture these contextual relationships.

Comparing results across different studies is difficult because datasets, labeling schemes, and evaluation methods vary so much. I focused on comparing methods within the same dataset using consistent evaluation.

## 3. Data

The dataset comes from `data/News-Data.csv` and contains news headlines and articles. I used the title and heading fields primarily, with the full text as backup when available. After cleaning (removing URLs, normalizing whitespace) and removing duplicates, the dataset has 21,739 samples.

The data is split 80/20 for training and validation, with the split saved to ensure consistent evaluation across experiments.

## 4. Methods

### 4.1 Classical Baseline

The baseline uses TF-IDF features:
- Word n-grams (1-2 words) with sublinear scaling
- Character n-grams (3-5 characters) for stylistic patterns
- Logistic regression with balanced class weights
- Grid search over regularization strength using 5-fold cross-validation

### 4.2 Transformer Baseline

The transformer approach fine-tunes DistilRoBERTa:
- RoBERTa tokenizer with 256 token limit
- DistilRoBERTa-base with classification head
- Class-weighted loss to handle imbalanced data
- Gradient accumulation for effective batch size of 16

### 4.3 Pipeline

```
Raw CSV → Clean → [TF-IDF | Tokenize] → [LogReg | RoBERTa] → {Left, Center, Right}
```

## 5. Experimental Setup

- Fixed 80/20 split with seed 42
- Metrics: accuracy and macro-F1
- All random seeds fixed for reproducibility
- Models saved to `models/` and `models/roberta/`

## 6. Results — Classical Baseline

**Performance**: 
- Accuracy: 30.1%
- Macro-F1: 28.0%

The baseline achieves modest but realistic performance. Detailed results are in `reports/metrics.json` and `reports/classification_report.txt`.

Confusion matrix: `reports/confusion_matrix.png`

**Top discriminative features**:
- Left: `reports/top_tokens_LEFT.txt`
- Center: `reports/top_tokens_CENTER.txt` 
- Right: `reports/top_tokens_RIGHT.txt`

## 7. Results — Transformer Baseline

**Performance**:
- Accuracy: 52.5%
- Macro-F1: 50.6%

The transformer model shows clear improvements over the baseline, roughly doubling the accuracy. This suggests that contextual understanding helps significantly with this task.

Detailed results: `reports/roberta/metrics.json` and `reports/roberta/classification_report.txt`.

## 8. Error Analysis

Sample predictions from both models are available in:
- Classical: `reports/sample_val_preds.csv`
- RoBERTa: `reports/roberta/sample_val_preds.csv`

These files show specific examples of where each model succeeds or fails.

## 9. Limitations

Several factors limit the current approach:
- Headlines are short, missing the full context of articles
- TF-IDF can't understand word meanings or relationships
- Bias labeling is subjective and varies between annotators
- The dataset could be larger for better generalization

## 10. Reproducibility

**Setup**:
```bash
make install
make split
```

**Train and evaluate**:
```bash
make train && make eval                    # Classical baseline
make install_roberta && make train_roberta && make eval_roberta  # Transformer
make compare                               # Generate comparison
```

All results are reproducible with the fixed random seed. Python 3.8+ required.

## 11. Project Structure

```
.
├── data/                    # Dataset and split indices
├── src/                     # Classical baseline code
├── src_roberta/            # Transformer pipeline
├── scripts/                 # Utility scripts
├── models/                  # Saved models
├── reports/                 # Results and plots
└── requirements*.txt        # Dependencies
```

<!-- BEGIN_COMPARISON -->
## Model Comparison

| Model                         | Accuracy | Macro-F1 |
|-------------------------------|----------|----------|
| TF-IDF + Logistic Regression  | 0.3007   | 0.2799   |
| DistilRoBERTa (fine-tuned)    | 0.5253   | 0.5058   |

**Improvement**: +22.5% accuracy, +22.6% F1-macro

The transformer model shows substantial improvements over the baseline. This suggests that understanding word context and relationships is crucial for detecting political bias in news headlines.

Confusion matrices: `reports/confusion_matrix.png` (baseline), `reports/roberta/confusion_matrix.png` (transformer)
<!-- END_COMPARISON -->
