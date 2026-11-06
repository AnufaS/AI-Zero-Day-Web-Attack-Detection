"""
config.py - ONE place for paths and settings, so every script agrees.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

RAW_CSV = os.path.join(ROOT, "dataset", "raw", "web_requests.csv")
PROCESSED_DIR = os.path.join(ROOT, "dataset", "processed")
TRAIN_CSV = os.path.join(PROCESSED_DIR, "train.csv")
VAL_CSV = os.path.join(PROCESSED_DIR, "val.csv")
TEST_CSV = os.path.join(PROCESSED_DIR, "test.csv")

MODEL_DIR = os.path.join(ROOT, "saved_models")
REPORT_DIR = os.path.join(ROOT, "reports")
FIG_DIR = os.path.join(REPORT_DIR, "figures")
METRICS_JSON = os.path.join(REPORT_DIR, "metrics.json")
DECISION_CONFIG_JSON = os.path.join(MODEL_DIR, "decision_config.json")

CLASSES = ["NORMAL", "SQL_INJECTION", "XSS", "PATH_TRAVERSAL", "COMMAND_INJECTION"]
RANDOM_STATE = 42


def ensure_dirs():
    for folder in (os.path.dirname(RAW_CSV), PROCESSED_DIR, MODEL_DIR, FIG_DIR):
        os.makedirs(folder, exist_ok=True)
