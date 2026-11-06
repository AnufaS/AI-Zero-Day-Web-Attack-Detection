"""
STEP 10 - Compare different threat-score weights on VALIDATION data (not test!).

Run:  python evaluation/tune_decision_engine.py

We keep behaviour weight = 0.15 (the dataset has no client history, so behaviour = 0 here)
and try splitting the remaining 0.85 between classifier and anomaly detector.
We score each choice with F2 (like F1 but recall counts MORE, since missed attacks hurt).
The winner is saved to saved_models/decision_config.json. It is only 'best on this
synthetic validation set' - do not call it scientifically optimal.
"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from config import VAL_CSV, DECISION_CONFIG_JSON, ensure_dirs
from evaluation.common import load_artifacts, score_dataframe
from models.decision_engine import DEFAULT_CONFIG


def f_beta(p, r, beta=2):
    return (1 + beta**2) * p * r / (beta**2 * p + r) if p + r else 0.0


def main():
    ensure_dirs()
    art = load_artifacts()
    val = pd.read_csv(VAL_CSV)
    proba, anom = score_dataframe(val, art)
    attack_score = 1 - proba["NORMAL"].values
    truth = (val["attack_type"] != "NORMAL").values
    thr = DEFAULT_CONFIG["suspicious_threshold"]
    print(f"{'w_attack':>8} {'w_anomaly':>9} {'precision':>9} {'recall':>7} {'F2':>6}")
    best = None
    for w_att in np.arange(0.30, 0.86, 0.05):
        w_an = round(float(0.85 - w_att), 2)
        threat = w_att * attack_score + w_an * anom
        flagged = threat >= thr
        tp = (flagged & truth).sum(); fp = (flagged & ~truth).sum(); fn = (~flagged & truth).sum()
        p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
        f2 = f_beta(p, r)
        print(f"{w_att:8.2f} {w_an:9.2f} {p:9.3f} {r:7.3f} {f2:6.3f}")
        if best is None or f2 > best[0] + 1e-9:
            best = (f2, round(float(w_att), 2), w_an)
    cfg = {"weights": {"attack": best[1], "anomaly": best[2], "behaviour": 0.15}}
    with open(DECISION_CONFIG_JSON, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"\nChosen weights (first best F2 = {best[0]:.3f}): {cfg['weights']}")
    print("CAUTION: validation data only contains attack types the classifier was trained on, so tuning")
    print("naturally rewards the classifier. The anomaly detector's real value (unseen attacks) is NOT")
    print("measured here - see evaluation/unseen_attack_experiment.py. Treat these weights as a starting point.")


if __name__ == "__main__":
    main()
