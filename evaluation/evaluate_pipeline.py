"""
STEP 17/18 - End-to-end test of the FULL decision engine on the test set + threat-score plot.

Run:  python evaluation/evaluate_pipeline.py
(behaviour score = 0 here because the dataset rows have no client history)
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import pandas as pd
from config import TEST_CSV, FIG_DIR, ensure_dirs
from evaluation.common import load_artifacts, score_dataframe, save_metrics, rates
from models.decision_engine import decide, load_config


def main():
    ensure_dirs()
    art, cfg = load_artifacts(), load_config()
    test = pd.read_csv(TEST_CSV)
    proba, anom = score_dataframe(test, art)
    res = pd.DataFrame([decide(proba.iloc[i].to_dict(), float(anom[i]), 0.0, cfg) for i in range(len(test))])
    truth_attack = (test["attack_type"] != "NORMAL").values
    print("Weights used:", cfg["weights"])
    print(pd.crosstab(test["attack_type"], res["decision"]).to_string())
    m = rates(truth_attack, (res["decision"] != "NORMAL").values)
    print(f"\nTreating anything other than NORMAL as an alert: precision {m['precision']}, recall {m['recall']}, "
          f"F1 {m['f1']}, FPR {m['false_positive_rate']}, FNR {m['false_negative_rate']}")
    save_metrics("full_pipeline", {k: v for k, v in m.items() if k in (
        "precision", "recall", "f1", "false_positive_rate", "false_negative_rate")})
    plt.figure(figsize=(7, 4))
    plt.hist(res.threat_score[~truth_attack], bins=30, alpha=.7, label="NORMAL", color="#4C72B0")
    plt.hist(res.threat_score[truth_attack], bins=30, alpha=.7, label="ATTACK", color="#C44E52")
    for t in (cfg["suspicious_threshold"], cfg["high_threshold"]):
        plt.axvline(t, color="k", ls="--")
    plt.xlabel("threat score"); plt.ylabel("requests"); plt.title("Threat score distribution (test set)")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{FIG_DIR}/threat_distribution.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
