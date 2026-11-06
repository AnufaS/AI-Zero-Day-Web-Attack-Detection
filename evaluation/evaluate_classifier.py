"""
STEP 12 + 17 + 18 - Evaluate the Random Forest on the TEST set (data it never saw).

Run:  python evaluation/evaluate_classifier.py

Why security people care:
  False Negative = a real attack called NORMAL   -> attacker gets in. Very costly.
  False Positive = a normal user called ATTACK   -> annoyed customer. Costly but milder.
  Recall = of all real attacks, how many did we catch?  (so: watch recall!)
  Precision = of all alarms we raised, how many were real?
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from config import TEST_CSV, RAW_CSV, FIG_DIR, CLASSES, ensure_dirs
from evaluation.common import load_artifacts, save_metrics
from preprocessing.feature_engineering import FEATURE_NAMES


def main():
    ensure_dirs()
    if not os.path.exists(TEST_CSV):
        print("ERROR: run python preprocessing/prepare_data.py first.")
        sys.exit(1)
    art = load_artifacts()
    test = pd.read_csv(TEST_CSV)
    y = test["attack_type"]
    pred = art["clf"].predict(art["scaler"].transform(test[FEATURE_NAMES]))

    acc = accuracy_score(y, pred)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    print(f"Accuracy {acc:.4f} | macro precision {p:.4f} | macro recall {r:.4f} | macro F1 {f:.4f}\n")
    print(classification_report(y, pred, labels=CLASSES, zero_division=0))
    cm = confusion_matrix(y, pred, labels=CLASSES)
    print("Confusion matrix (rows = truth, columns = prediction):")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())
    attack_missed = int(((y != "NORMAL") & (pred == "NORMAL")).sum())
    normal_flagged = int(((y == "NORMAL") & (pred != "NORMAL")).sum())
    print(f"\nFalse negatives (attack called NORMAL): {attack_missed}")
    print(f"False positives (normal called attack): {normal_flagged}")
    save_metrics("classifier", {"accuracy": round(acc, 4), "precision_macro": round(p, 4),
                                "recall_macro": round(r, 4), "f1_macro": round(f, 4),
                                "false_negatives": attack_missed, "false_positives": normal_flagged})

    # --- plots ---
    raw = pd.read_csv(RAW_CSV)
    ax = raw["attack_type"].value_counts().reindex(CLASSES).plot(kind="bar", color="#4C72B0", figsize=(7, 4))
    ax.set_title("Attack class distribution (whole dataset)"); ax.set_ylabel("requests")
    plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig(f"{FIG_DIR}/class_distribution.png", dpi=120); plt.close()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5)); ax.set_xticklabels(CLASSES, rotation=40, ha="right")
    ax.set_yticks(range(5)); ax.set_yticklabels(CLASSES)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion matrix (test set)")
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/confusion_matrix.png", dpi=120); plt.close()

    imp = pd.Series(art["clf"].feature_importances_, index=FEATURE_NAMES).sort_values()
    imp.tail(12).plot(kind="barh", figsize=(7, 5), color="#55A868")
    plt.title("Random Forest feature importance (top 12)"); plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/feature_importance.png", dpi=120); plt.close()
    print("\nTop features:\n" + imp.sort_values(ascending=False).head(8).round(3).to_string())

    rep = classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0)
    pd.DataFrame({c: [rep[c]["precision"], rep[c]["recall"], rep[c]["f1-score"]] for c in CLASSES},
                 index=["precision", "recall", "f1"]).T.plot(kind="bar", figsize=(8, 4), ylim=(0, 1.05))
    plt.title("Per-class performance (test set)"); plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/model_performance.png", dpi=120); plt.close()
    print(f"\nFigures saved in {FIG_DIR}")


if __name__ == "__main__":
    main()
