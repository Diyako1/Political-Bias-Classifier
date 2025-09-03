import os
import json
import torch
import numpy as np
from pathlib import Path
from collections import Counter
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)
from datasets import Dataset
import evaluate
from src_roberta.config import TCFG
from src_roberta.data import load_and_split, ID2LABEL, LABEL2ID

class WeightedTrainer(Trainer):
    """Custom Trainer with class-weighted loss"""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute weighted cross-entropy loss"""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Apply class weights
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

def setup_device():
    """Auto-detect and setup the best available device"""
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

def compute_class_weights(train_df, device):
    """Compute class weights based on inverse frequency"""
    label_counts = Counter(train_df["label"])
    total_samples = len(train_df)
    
    # Compute inverse frequency weights
    weights = []
    for class_id in sorted(LABEL2ID.values()):
        count = label_counts.get(class_id, 1)  # Avoid division by zero
        weight = total_samples / (len(LABEL2ID) * count)  # Inverse frequency, normalized
        weights.append(weight)
    
    class_weights = torch.tensor(weights, dtype=torch.float, device=device)
    
    print(f"\nClass weights (inverse frequency):")
    for class_id in sorted(LABEL2ID.values()):
        class_name = ID2LABEL[class_id]
        count = label_counts.get(class_id, 0)
        weight = weights[class_id]
        print(f"  {class_name}: {count} samples, weight={weight:.4f}")
    
    return class_weights

def create_datasets(train_df, val_df, tokenizer):
    """Create tokenized datasets for training"""
    def tokenize_function(examples):
        return tokenizer(
            examples["input_text"],
            truncation=True,
            padding=True,
            max_length=TCFG.MAX_LEN
        )
    
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    # Tokenize
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Set format for PyTorch
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    return train_dataset, val_dataset

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Load metrics
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    
    # Compute metrics
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1_macro = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    f1_weighted = f1_metric.compute(predictions=predictions, references=labels, average="weighted")
    
    return {
        "accuracy": accuracy["accuracy"],
        "f1_macro": f1_macro["f1"],
        "f1_weighted": f1_weighted["f1"]
    }

def main():
    """Main training pipeline with improvements"""
    print("RoBERTa Political Bias Classifier Training")
    print("=" * 60)
    
    # Set random seed
    set_seed(TCFG.SEED)
    
    # Setup device
    device = setup_device()
    
    # Create output directories
    Path(TCFG.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(TCFG.REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(TCFG.LOG_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load and split data
    train_df, val_df = load_and_split()
    
    # Compute class weights
    class_weights = compute_class_weights(train_df, device)
    
    # Initialize tokenizer and model
    print(f"\nLoading {TCFG.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(TCFG.MODEL_NAME)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        TCFG.MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    
    # Move model to device
    model.to(device)
    
    # Create datasets
    print("Creating tokenized datasets...")
    train_dataset, val_dataset = create_datasets(train_df, val_df, tokenizer)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Max sequence length: {TCFG.MAX_LEN}")
    print(f"Effective batch size: {TCFG.BATCH_SIZE * 2} (with gradient accumulation)")
    
    # Training arguments with gradient accumulation and early stopping
    training_args = TrainingArguments(
        output_dir=TCFG.OUTPUT_DIR,
        num_train_epochs=TCFG.EPOCHS,
        per_device_train_batch_size=TCFG.BATCH_SIZE,
        per_device_eval_batch_size=16,  # Larger for evaluation
        gradient_accumulation_steps=2,  # Effective batch size = 8 * 2 = 16
        learning_rate=TCFG.LR,
        weight_decay=TCFG.WEIGHT_DECAY,
        warmup_ratio=TCFG.WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to="none",  # Disable wandb/tensorboard
        seed=TCFG.SEED,
        logging_dir=TCFG.LOG_DIR,
        logging_steps=50,
        save_total_limit=3,  # Keep best 3 checkpoints
        dataloader_pin_memory=False if device.type == "mps" else True,  # MPS compatibility
    )
    
    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=2,
        early_stopping_threshold=0.001  # Minimum improvement threshold
    )
    
    # Initialize weighted trainer
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,  # Correct parameter name
        compute_metrics=compute_metrics,
        callbacks=[early_stopping]
    )
    
    # Train the model
    print(f"\nStarting training:")
    print(f"  Epochs: {TCFG.EPOCHS} (with early stopping)")
    print(f"  Learning rate: {TCFG.LR}")
    print(f"  Warmup ratio: {TCFG.WARMUP_RATIO}")
    print(f"  Batch size: {TCFG.BATCH_SIZE} (effective: {TCFG.BATCH_SIZE * 2})")
    print(f"  Max length: {TCFG.MAX_LEN}")
    print(f"  Device: {device}")
    print(f"  Class weights: enabled")
    
    train_result = trainer.train()
    
    # Save the final model and tokenizer
    print(f"\nSaving model to {TCFG.OUTPUT_DIR}")
    trainer.save_model(TCFG.OUTPUT_DIR)
    tokenizer.save_pretrained(TCFG.OUTPUT_DIR)
    
    # Final evaluation
    print("\nFinal evaluation on validation set:")
    eval_result = trainer.evaluate()
    
    for key, value in eval_result.items():
        if key.startswith("eval_"):
            metric_name = key.replace("eval_", "")
            print(f"  {metric_name}: {value:.4f}")
    
    # Save training metrics
    training_metrics = {
        "model_name": TCFG.MODEL_NAME,
        "hyperparameters": {
            "epochs": TCFG.EPOCHS,
            "batch_size": TCFG.BATCH_SIZE,
            "effective_batch_size": TCFG.BATCH_SIZE * 2,
            "learning_rate": TCFG.LR,
            "max_length": TCFG.MAX_LEN,
            "dropout": 0.3,
            "label_smoothing": 0.1
        },
        "dataset_info": {
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "class_weights": class_weights.cpu().tolist()
        },
        "final_metrics": eval_result,
        "device": str(device),
        "training_completed": train_result.metrics.get("train_runtime", 0) > 0
    }
    
    with open(f"{TCFG.REPORTS_DIR}/training_metrics.json", "w") as f:
        json.dump(training_metrics, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Model saved to: {TCFG.OUTPUT_DIR}")
    print(f"Training metrics saved to: {TCFG.REPORTS_DIR}/training_metrics.json")
    print(f"Best validation F1-macro: {eval_result.get('eval_f1_macro', 'N/A'):.4f}")

if __name__ == "__main__":
    main()