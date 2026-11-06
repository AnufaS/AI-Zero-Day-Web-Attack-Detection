"""
Runs the WHOLE pipeline in order (Steps 2-12). Run from the project root:
    python run_all.py
"""
import subprocess
import sys

STEPS = [
    ("STEP 2   Create dataset", "data_generation/generate_dataset.py"),
    ("STEP 3-4 Preprocess + features + split", "preprocessing/prepare_data.py"),
    ("STEP 6   Train Random Forest", "models/train_classifier.py"),
    ("STEP 7   Train Isolation Forest", "models/train_anomaly_detector.py"),
    ("STEP 10  Compare decision weights", "evaluation/tune_decision_engine.py"),
    ("STEP 12  Evaluate classifier", "evaluation/evaluate_classifier.py"),
    ("STEP 12  Evaluate anomaly detector", "evaluation/evaluate_anomaly.py"),
    ("STEP 11  Unseen attack experiment", "evaluation/unseen_attack_experiment.py"),
    ("STEP 17  Threat score plot + end-to-end test", "evaluation/evaluate_pipeline.py"),
]
for title, script in STEPS:
    print("\n" + "=" * 70 + f"\n{title}  ->  python {script}\n" + "=" * 70)
    if subprocess.call([sys.executable, script]) != 0:
        print(f"\nStopped: {script} failed. Fix the error above, then run again.")
        sys.exit(1)
print("\nAll done. Next: python main.py   or   python prediction/predict.py")
