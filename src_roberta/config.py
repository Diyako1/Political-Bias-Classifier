from dataclasses import dataclass

@dataclass
class TCFG:
    CSV_PATH: str = "data/News-Data.csv"
    TEXT_COLS: tuple = ("title", "heading", "text")   # Order: title, heading, fallback to text
    LABEL_COL: str = "bias_rating"           # left/center/right
    MODEL_NAME: str = "distilroberta-base"   # faster than roberta-base; good for MacBook
    MAX_LEN: int = 256                       # Increased for more context
    BATCH_SIZE: int = 8                      # Reduced for memory efficiency
    EPOCHS: int = 4                          # Reasonable epochs with early stopping
    LR: float = 2e-5                         # Lower learning rate
    WEIGHT_DECAY: float = 0.01
    WARMUP_RATIO: float = 0.1                # Increased warmup
    SEED: int = 42
    OUTPUT_DIR: str = "models/roberta"
    REPORTS_DIR: str = "reports/roberta"
    LOG_DIR: str = "reports/roberta_logs"