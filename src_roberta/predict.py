import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src_roberta.config import TCFG
from src_roberta.data import ID2LABEL, clean_text

def setup_device():
    """Auto-detect the best available device"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    return device

def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(TCFG.OUTPUT_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(TCFG.OUTPUT_DIR)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please ensure the model is trained and saved in: {TCFG.OUTPUT_DIR}")
        return None, None

def predict_bias(text: str, model, tokenizer, device):
    """Predict political bias for a given text"""
    # Clean the input text
    cleaned_text = clean_text(text)
    
    if not cleaned_text:
        print("Warning: Text became empty after cleaning")
        return None, None
    
    # Tokenize
    inputs = tokenizer(
        cleaned_text,
        truncation=True,
        padding=True,
        max_length=TCFG.MAX_LEN,
        return_tensors="pt"
    )
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get probabilities
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        
        # Get predicted class
        predicted_class_id = torch.argmax(logits, dim=-1).cpu().numpy()[0]
        predicted_label = ID2LABEL[predicted_class_id]
    
    return predicted_label, probabilities

def main():
    """CLI interface for making predictions"""
    parser = argparse.ArgumentParser(description="Predict political bias using RoBERTa")
    parser.add_argument("--text", type=str, required=True, help="Headline to classify")
    parser.add_argument("--verbose", action="store_true", help="Show probabilities for all classes")
    
    args = parser.parse_args()
    
    # Setup device
    device = setup_device()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    if model is None or tokenizer is None:
        return
    
    # Move model to device
    model.to(device)
    
    # Make prediction
    predicted_label, probabilities = predict_bias(args.text, model, tokenizer, device)
    
    if predicted_label is None:
        print("Failed to make prediction")
        return
    
    # Print results
    if args.verbose:
        print(f"Input: {args.text}")
        print(f"Predicted bias: {predicted_label}")
        print("\nClass probabilities:")
        for i, (class_id, class_name) in enumerate(ID2LABEL.items()):
            prob = probabilities[i]
            print(f"  {class_name:>6}: {prob:.4f}")
    else:
        print(predicted_label)

if __name__ == "__main__":
    main()

