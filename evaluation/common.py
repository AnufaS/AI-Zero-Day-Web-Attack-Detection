"""Shared helpers for evaluation scripts: load models + data, score a DataFrame."""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import matplotlib
matplotlib.use("Agg")           # draw to files, no window needed
import numpy as np
import pandas as pd
from config import MODEL_DIR, METRICS_JSON, REPORT_DIR, FIG_DIR, ensure_dirs
from preprocessing.feature_engineering import FEATURE_NAMES
from models.train_anomaly_detector import anomaly_scores


def load_artifacts():
    try:
        return {"clf": joblib.load(os.path.join(MODEL_DIR, "attack_classifier.pkl")),
                "scaler": joblib.load(os.path.join(MODEL_DIR, "scaler.pkl")),
                "anomaly": joblib.load(os.path.join(MODEL_DIR, "anomaly_detector.pkl"))}
    except FileNotFoundError:
        print("ERROR: models not found. Train them first (see README, steps 6-7).")
        sys.exit(1)


def score_dataframe(df, art, clf=None, scaler=None, anomaly=None):
    """Return (probabilities DataFrame, anomaly score array) for a processed DataFrame."""
    clf = clf or art["clf"]
    X = (scaler or art["scaler"]).transform(df[FEATURE_NAMES])
    proba = pd.DataFrame(clf.predict_proba(X), columns=clf.classes_, index=df.index)
    return proba, anomaly_scores(anomaly or art["anomaly"], X)


def save_metrics(section, values):
    ensure_dirs()
    data = {}
    if os.path.exists(METRICS_JSON):
        with open(METRICS_JSON) as f:
            data = json.load(f)
    data[section] = values
    with open(METRICS_JSON, "w") as f:
        json.dump(data, f, indent=2)


def rates(y_true_attack, y_pred_attack):
    """Precision/recall/F1/FPR/FNR for a binary 'is it an attack?' question."""
    t, p = np.asarray(y_true_attack, bool), np.asarray(y_pred_attack, bool)
    tp, fp, fn, tn = (t & p).sum(), (~t & p).sum(), (t & ~p).sum(), (~t & ~p).sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(float(prec), 4), "recall": round(float(rec), 4), "f1": round(float(f1), 4),
            "false_positive_rate": round(float(fp / (fp + tn)) if fp + tn else 0.0, 4),
            "false_negative_rate": round(float(fn / (fn + tp)) if fn + tp else 0.0, 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}
