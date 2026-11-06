"""
STEP 3 + 15 - Clean the raw data, extract features, split into train / validation / test.

Run:  python preprocessing/prepare_data.py

DATA LEAKAGE = test information accidentally leaking into training, which makes
results look better than they really are. Example: the same payload (or the same
template with a different number) in both train and test -> the model just memorises.
We prevent this by splitting by `group_id` (the template), not by single rows:
all rows of one template go to ONE split only. Split: 70% / 15% / 15% of groups.
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from config import (RAW_CSV, PROCESSED_DIR, TRAIN_CSV, VAL_CSV, TEST_CSV, RANDOM_STATE,
                    ensure_dirs)
from preprocessing.feature_engineering import features_from_dataframe


def group_split(df, seed=RANDOM_STATE):
    rng = random.Random(seed)
    assign = {}
    for attack_type, part in df.groupby("attack_type"):
        groups = sorted(part["group_id"].unique())
        rng.shuffle(groups)
        n = len(groups)
        n_val = max(1, round(n * 0.15))
        n_test = max(1, round(n * 0.15))
        for g in groups[:n_val]:
            assign[g] = "val"
        for g in groups[n_val:n_val + n_test]:
            assign[g] = "test"
        for g in groups[n_val + n_test:]:
            assign[g] = "train"
    return df["group_id"].map(assign)


def main():
    ensure_dirs()
    if not os.path.exists(RAW_CSV):
        print("ERROR: dataset not found. Create it first with:")
        print("       python data_generation/generate_dataset.py")
        sys.exit(1)

    df = pd.read_csv(RAW_CSV)
    print(f"Loaded {len(df)} rows.")
    # --- cleaning ---
    df = df.dropna(subset=["method", "url", "attack_type"])
    df["user_agent"] = df["user_agent"].fillna("")
    df["method"] = df["method"].str.upper().str.strip()
    is_normal = df["attack_type"] == "NORMAL"
    df = pd.concat([df[is_normal].drop_duplicates(subset=["url", "user_agent", "request_size"]),
                    df[~is_normal].drop_duplicates(subset=["url"])]).reset_index(drop=True)
    print(f"After cleaning/removing duplicates: {len(df)} rows.")

    feats = features_from_dataframe(df)
    out = pd.concat([df[["method", "url", "label", "attack_type", "group_id"]], feats], axis=1)
    out["split"] = group_split(out)

    for name, path in (("train", TRAIN_CSV), ("val", VAL_CSV), ("test", TEST_CSV)):
        part = out[out["split"] == name].drop(columns=["split"])
        part.to_csv(path, index=False)
        print(f"\n{name.upper():5s} -> {len(part)} rows  ({path})")
        print(part["attack_type"].value_counts().to_string())

    # leakage check: no group may appear in two splits
    overlap = out.groupby("group_id")["split"].nunique().max()
    assert overlap == 1, "Leakage! A template appears in more than one split."
    print("\nLeakage check passed: every template lives in exactly one split.")


if __name__ == "__main__":
    main()
