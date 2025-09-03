from dataclasses import dataclass
from typing import Tuple

LABEL2ID = {"left": 0, "center": 1, "right": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

@dataclass
class CFG:
    # Data paths
    CSV_PATH: str = "data/News-Data.csv"
    
    # Text columns (will combine all available)
    TEXT_COLS: Tuple[str, ...] = ("title", "heading", "text")
    LABEL_COL: str = "bias_rating"
    
    # Train/test split
    TEST_SIZE: float = 0.2
    SEED: int = 42
    N_JOBS: int = -1
    
    # Enhanced TF-IDF config
    WORD_NGRAMS: Tuple[int, int] = (1, 3)  # Expanded to trigrams
    CHAR_NGRAMS: Tuple[int, int] = (3, 5)
    MAX_FEATURES: int = 300_000  # Increased capacity
    MIN_DF: int = 2
    MAX_DF: float = 0.9  # More aggressive filtering
    STRIP_ACCENTS: str = "unicode"
    STOP_WORDS: str = "english"
    SUBLINEAR_TF: bool = True
    
    # Cross-validation
    CV_FOLDS: int = 5
    
    # Class imbalance handling
    USE_SMOTE: bool = False  # Toggle for SMOTE vs class_weight
    SMOTE_K_NEIGHBORS: int = 5
    
    # Model hyperparameters
    LR_C_GRID = [0.1, 0.5, 1.0, 2.0, 5.0]
    SVC_C_GRID = [0.1, 0.5, 1.0, 2.0, 5.0]
    RF_N_ESTIMATORS_GRID = [100, 200, 300]
    RF_MAX_DEPTH_GRID = [10, 20, None]