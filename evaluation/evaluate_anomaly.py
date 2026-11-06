"""
STEP 12 + 17 - Evaluate the Isolation Forest on the TEST set.

Run:  python evaluation/evaluate_anomaly.py

Question asked: "if we flag every request with anomaly score >= threshold, how well
do we separate NORMAL from ATTACK?"  (Isolation Forest never saw attacks in training.)
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score
from config import TEST_CSV, FIG_DIR, ensure_dirs
from evaluation.common import load_artifacts, score_dataframe, save_metrics, rates
from models.decision_engine import load_config


def main():
    ensure_dirs()
    art = load_artifacts()
    test = pd.read_csv(TEST_CSV)
    _, anom = score_dataframe(test, art)
    is_attack = (test["attack_type"] != "NORMAL").values
    thr = load_config()["anomaly_high"]
    m = rates(is_attack, anom >= thr)
    auc = roc_auc_score(is_attack, anom)
    print(f"Threshold {thr}: precision {m['precision']} recall {m['recall']} F1 {m['f1']}")
    print(f"False positive rate {m['false_positive_rate']} | false negative rate {m['false_negative_rate']}")
    print(f"ROC-AUC (threshold-free separation quality): {auc:.4f}")
    print("\nWhat to expect: LOW recall on the 'harder' classes means anomaly detection alone misses things;")
    print("a high false-positive rate means normal-but-odd requests are being flagged.")
    print(test.assign(anomaly=anom).groupby("attack_type")["anomaly"].agg(["mean", "min", "max"]).round(2).to_string())
    save_metrics("anomaly_detector", {**{k: v for k, v in m.items() if k in (
        "precision", "recall", "f1", "false_positive_rate", "false_negative_rate")},
        "roc_auc": round(float(auc), 4), "threshold": thr})

    plt.figure(figsize=(7, 4))
    plt.hist(anom[~is_attack], bins=30, alpha=.7, label="NORMAL", color="#4C72B0")
    plt.hist(anom[is_attack], bins=30, alpha=.7, label="ATTACK", color="#C44E52")
    plt.axvline(thr, color="k", ls="--", label=f"threshold {thr}")
    plt.xlabel("anomaly score (0 = normal, 1 = very unusual)"); plt.ylabel("requests")
    plt.title("Anomaly score distribution (test set)"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/anomaly_distribution.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
