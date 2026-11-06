"""
STEP 7 - ISOLATION FOREST anomaly detector (trained on NORMAL traffic only).

Run:  python models/train_anomaly_detector.py     (after train_classifier.py)

How Isolation Forest works (simple): it builds many random trees that split the data
with random questions ("is url_length > 37?"). A weird point is alone in its own corner,
so it gets separated after FEW splits. A typical point sits in a crowd and needs MANY
splits. Short path = more anomalous.

The raw sklearn output is NOT a probability. We convert it to 0..1 like this:
    raw = -score_samples(x)               (bigger = more unusual)
    score = (raw - typical_normal) / (rare_normal - typical_normal), clipped to 0..1
  typical_normal = median raw score of NORMAL validation requests   -> maps to 0.0
  rare_normal    = 99th percentile of NORMAL validation requests    -> maps to 1.0
So 1.0 means "more unusual than 99% of normal traffic we saw". It is a relative
ranking score, not "probability of attack".
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from config import TRAIN_CSV, VAL_CSV, MODEL_DIR, RANDOM_STATE, ensure_dirs
from preprocessing.feature_engineering import FEATURE_NAMES


def build_anomaly_detector(X_train_normal, X_val_normal):
    """Fit on normal training data; use normal validation data to set the 0..1 scale."""
    model = IsolationForest(n_estimators=300, contamination="auto", random_state=RANDOM_STATE)
    model.fit(X_train_normal)
    raw_val = -model.score_samples(X_val_normal)
    return {"model": model, "lo": float(np.median(raw_val)), "hi": float(np.percentile(raw_val, 99))}


def anomaly_scores(bundle, X_scaled):
    raw = -bundle["model"].score_samples(X_scaled)
    return np.clip((raw - bundle["lo"]) / (bundle["hi"] - bundle["lo"]), 0, 1)


def main():
    ensure_dirs()
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if not (os.path.exists(TRAIN_CSV) and os.path.exists(scaler_path)):
        print("ERROR: run prepare_data.py and train_classifier.py first.")
        sys.exit(1)
    scaler = joblib.load(scaler_path)
    train = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    Xtr = scaler.transform(train[train.attack_type == "NORMAL"][FEATURE_NAMES])
    Xva = scaler.transform(val[val.attack_type == "NORMAL"][FEATURE_NAMES])
    bundle = build_anomaly_detector(Xtr, Xva)
    joblib.dump(bundle, os.path.join(MODEL_DIR, "anomaly_detector.pkl"))
    print(f"Trained Isolation Forest on {len(Xtr)} NORMAL requests only.")
    print(f"Score scale: raw {bundle['lo']:.3f} -> 0.0 , raw {bundle['hi']:.3f} -> 1.0")

    # quick look at validation scores
    Xv_all = scaler.transform(val[FEATURE_NAMES])
    s = anomaly_scores(bundle, Xv_all)
    for cls in ["NORMAL", "SQL_INJECTION", "XSS", "PATH_TRAVERSAL", "COMMAND_INJECTION"]:
        mask = (val.attack_type == cls).values
        if mask.any():
            print(f"  validation {cls:18s} mean anomaly score = {s[mask].mean():.2f}")
    print("Saved saved_models/anomaly_detector.pkl")


if __name__ == "__main__":
    main()
