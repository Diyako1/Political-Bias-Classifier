import argparse, joblib
from src.config import ID2LABEL

def load_pipeline():
    return joblib.load("models/pipeline.pkl")

def predict_bias(headline: str) -> str:
    pipe = load_pipeline()
    pred_id = pipe.predict([headline])[0]
    return ID2LABEL[int(pred_id)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Headline to classify")
    args = parser.parse_args()
    label = predict_bias(args.text)
    print(label)

if __name__ == "__main__":
    main()