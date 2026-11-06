"""
STEP 6 - RANDOM FOREST attack classifier.

Run:  python models/train_classifier.py

Random Forest (simple): one decision tree asks a chain of yes/no questions
("many special characters? SQL keywords?") and ends in a class. One tree can be
fooled by quirks of the training data. A forest trains MANY trees on random
slices of the data and lets them VOTE. The share of votes for each class is the
predicted probability.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler
from config import TRAIN_CSV, VAL_CSV, MODEL_DIR, RANDOM_STATE, ensure_dirs
from preprocessing.feature_engineering import FEATURE_NAMES


def new_forest():
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                  class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)


def main():
    ensure_dirs()
    if not os.path.exists(TRAIN_CSV):
        print("ERROR: processed data not found. Run: python preprocessing/prepare_data.py")
        sys.exit(1)
    train, val = pd.read_csv(TRAIN_CSV), pd.read_csv(VAL_CSV)

    scaler = StandardScaler().fit(train[FEATURE_NAMES])     # learn mean/std from TRAIN only
    Xtr, Xva = scaler.transform(train[FEATURE_NAMES]), scaler.transform(val[FEATURE_NAMES])
    ytr, yva = train["attack_type"], val["attack_type"]

    forest = new_forest().fit(Xtr, ytr)
    pred = forest.predict(Xva)
    print(f"Validation accuracy: {accuracy_score(yva, pred):.4f}")
    print(f"Validation macro F1: {f1_score(yva, pred, average='macro'):.4f}")

    # Optional: calibration. Raw forest probabilities are not guaranteed to be honest.
    calibrated = CalibratedClassifierCV(new_forest(), method="sigmoid", cv=3).fit(Xtr, ytr)
    for name, m in (("raw forest", forest), ("calibrated", calibrated)):
        proba = m.predict_proba(Xva)
        print(f"  {name:11s} log-loss = {log_loss(yva, proba, labels=m.classes_):.4f} (lower = more honest probabilities)")

    joblib.dump(forest, os.path.join(MODEL_DIR, "attack_classifier.pkl"))
    joblib.dump(calibrated, os.path.join(MODEL_DIR, "attack_classifier_calibrated.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(FEATURE_NAMES, os.path.join(MODEL_DIR, "feature_names.pkl"))
    print("Saved attack_classifier.pkl, attack_classifier_calibrated.pkl, scaler.pkl, feature_names.pkl")


if __name__ == "__main__":
    main()
