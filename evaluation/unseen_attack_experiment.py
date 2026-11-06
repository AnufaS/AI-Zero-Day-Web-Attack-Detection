"""
STEP 11 - UNSEEN ATTACK DETECTION EXPERIMENT

Run:  python evaluation/unseen_attack_experiment.py

Idea: pretend COMMAND_INJECTION is a brand-new attack family nobody has seen.
  * Scaler + Random Forest are trained WITHOUT any COMMAND_INJECTION row.
  * Isolation Forest only ever sees NORMAL traffic (as always).
  * We then test every COMMAND_INJECTION request and see what the full system says.
We also check normal test traffic, to count false alarms.

LIMITATIONS (read these!): this is a simulation. COMMAND_INJECTION is only 'unseen' as a
label - its payloads are still synthetic, and its look (special characters, odd words)
overlaps with the attacks we did train on. A real zero-day may look very different, and
it may look perfectly normal. This does NOT prove real-world zero-day detection.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import TRAIN_CSV, VAL_CSV, TEST_CSV, FIG_DIR, ensure_dirs
from evaluation.common import score_dataframe, save_metrics
from models.decision_engine import decide, load_config
from models.train_anomaly_detector import build_anomaly_detector
from models.train_classifier import new_forest
from preprocessing.feature_engineering import FEATURE_NAMES

UNSEEN = "COMMAND_INJECTION"


def main():
    ensure_dirs()
    train, val, test = pd.read_csv(TRAIN_CSV), pd.read_csv(VAL_CSV), pd.read_csv(TEST_CSV)

    # 1) Remove the unseen family from EVERYTHING used for fitting/tuning.
    train_known = train[train.attack_type != UNSEEN]
    val_known = val[val.attack_type != UNSEEN]
    assert UNSEEN not in set(train_known.attack_type), "leak: unseen class in training!"
    print(f"Training classes: {sorted(train_known.attack_type.unique())}")

    scaler = StandardScaler().fit(train_known[FEATURE_NAMES])
    clf = new_forest().fit(scaler.transform(train_known[FEATURE_NAMES]), train_known.attack_type)
    anomaly = build_anomaly_detector(
        scaler.transform(train_known[train_known.attack_type == "NORMAL"][FEATURE_NAMES]),
        scaler.transform(val_known[val_known.attack_type == "NORMAL"][FEATURE_NAMES]))

    # 2) Test sets: ALL command-injection rows (never used for anything) + held-out normal rows
    unseen = pd.concat([train, val, test])[lambda d: d.attack_type == UNSEEN].reset_index(drop=True)
    normal_test = test[test.attack_type == "NORMAL"].reset_index(drop=True)
    cfg = load_config()

    def run(df):
        proba, anom = score_dataframe(df, None, clf=clf, scaler=scaler, anomaly=anomaly)
        return [decide(proba.iloc[i].to_dict(), float(anom[i]), 0.0, cfg) for i in range(len(df))]

    res_unseen, res_normal = run(unseen), run(normal_test)
    u = pd.DataFrame(res_unseen)
    n = pd.DataFrame(res_normal)

    print(f"\n=== {len(unseen)} unseen {UNSEEN} requests ===")
    print(u["decision"].value_counts().to_string())
    print(f"\nMean anomaly score: unseen attack {u.anomaly_score.mean():.2f} | normal traffic {n.anomaly_score.mean():.2f}")
    print(f"Mean classifier confidence on unseen attack: {u.max_probability.mean():.2f}")
    print("What the classifier (wrongly, it has never seen this class) guessed:")
    print(u["prediction"].value_counts().to_string())

    detected = (u.decision != "NORMAL").mean()
    flagged_unknown = (u.decision == "POSSIBLE UNKNOWN/ZERO-DAY ATTACK").mean()
    fp = (n.decision != "NORMAL").mean()
    print(f"\nClassifier ALONE would call {(u.prediction == 'NORMAL').mean():.1%} of them NORMAL (missed attacks!)")
    print(f"Unseen attacks flagged as anything other than NORMAL : {detected:.1%}")
    print(f"   ...of which labelled POSSIBLE UNKNOWN/ZERO-DAY        : {flagged_unknown:.1%}")
    print(f"   ...labelled 'KNOWN ATTACK' (right alarm, wrong type) : {(u.decision == 'KNOWN ATTACK').mean():.1%}")
    print(f"False alarms on normal test traffic (not NORMAL)         : {fp:.1%}")

    print("\nSample of individual requests:")
    show = pd.DataFrame({"url": unseen["url"].str[:46], "classifier": u["prediction"],
                         "conf": u["max_probability"].round(2), "anomaly": u["anomaly_score"].round(2),
                         "final": u["decision"].str[:30]})
    print(show.sample(8, random_state=1).to_string(index=False))

    print("\nHOW TO READ THIS: the anomaly detector gives a score high enough to trip alarms even though the")
    print("classifier has no idea what command injection is. But if many are labelled 'KNOWN ATTACK', the")
    print("classifier confidently MIS-labelled them as a related family - the alarm is right, the name is wrong.")
    save_metrics("unseen_attack_experiment", {
        "unseen_class": UNSEEN, "n_unseen": int(len(unseen)),
        "flagged_not_normal": round(float(detected), 4),
        "flagged_possible_unknown": round(float(flagged_unknown), 4),
        "false_alarm_rate_normal": round(float(fp), 4),
        "mean_anomaly_unseen": round(float(u.anomaly_score.mean()), 3),
        "mean_anomaly_normal": round(float(n.anomaly_score.mean()), 3)})

    plt.figure(figsize=(7, 4))
    plt.hist(n.anomaly_score, bins=25, alpha=.7, label="normal (test)", color="#4C72B0")
    plt.hist(u.anomaly_score, bins=25, alpha=.7, label=f"unseen {UNSEEN}", color="#C44E52")
    plt.xlabel("anomaly score"); plt.ylabel("requests"); plt.legend()
    plt.title("Unseen attack experiment: anomaly scores"); plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/unseen_attack_experiment.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
