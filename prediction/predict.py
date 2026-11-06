"""
STEP 13 + 14 + 20 - PREDICTION PIPELINE (load saved models, analyse one request).

Run the interactive tool:   python prediction/predict.py
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import joblib
import pandas as pd
from config import MODEL_DIR
from models.behaviour import behaviour_score
from models.decision_engine import decide, load_config
from models.train_anomaly_detector import anomaly_scores
from preprocessing.feature_engineering import FEATURE_NAMES, extract_features, validate_request


def load_models():
    """Load everything saved by the training scripts. Gives a friendly error if missing."""
    try:
        return {"clf": joblib.load(os.path.join(MODEL_DIR, "attack_classifier.pkl")),
                "scaler": joblib.load(os.path.join(MODEL_DIR, "scaler.pkl")),
                "anomaly": joblib.load(os.path.join(MODEL_DIR, "anomaly_detector.pkl")),
                "config": load_config()}
    except FileNotFoundError as err:
        raise FileNotFoundError(
            f"Model file missing ({os.path.basename(err.filename or '')}). Train first:\n"
            "  python models/train_classifier.py\n  python models/train_anomaly_detector.py") from err


def analyze_request(models, method, url, user_agent="", request_size=None, behaviour=0.0):
    """Full pipeline for ONE request. Raises ValueError for bad input."""
    method, url, user_agent, request_size = validate_request(method, url, user_agent, request_size)
    feats = extract_features(method, url, user_agent, request_size)
    X = models["scaler"].transform(pd.DataFrame([feats], columns=FEATURE_NAMES))
    probs = dict(zip(models["clf"].classes_, models["clf"].predict_proba(X)[0]))
    anomaly = float(anomaly_scores(models["anomaly"], X)[0])
    result = decide(probs, anomaly, behaviour, models["config"])
    result.update({"method": method, "url": url})
    return result


def print_result(r):
    print("\n" + "-" * 50)
    print(f"Prediction     : {r['prediction']}")
    for cls, p in sorted(r["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"   {cls:18s} {p:.2f}")
    print(f"Confidence     : {r['max_probability']:.1%} ({r['confidence_level']})")
    print(f"Anomaly score  : {r['anomaly_score']:.2f}   (0 = normal, 1 = very unusual)")
    print(f"Behaviour score: {r['behaviour_score']:.2f}")
    print(f"Threat score   : {r['threat_score']:.2f}   (risk {r['risk_level']})")
    print(f"FINAL          : {r['decision_detail']}")
    print(f"Why            : {r['reason']}")
    print("-" * 50)


def main():
    try:
        models = load_models()
    except FileNotFoundError as err:
        print("ERROR:", err)
        sys.exit(1)
    print("AI Zero-Day Web Attack Detector  (type 'quit' as the URL to stop)")
    while True:
        url = input("\nEnter URL (e.g. /login?username=admin' OR '1'='1): ").strip()
        if url.lower() in ("quit", "exit", "q"):
            break
        method = input("HTTP method [GET]: ").strip() or "GET"
        ua = input("User-Agent [Mozilla/5.0]: ").strip() or "Mozilla/5.0"
        size = input("Request size in bytes [auto]: ").strip()
        rpm = input("Requests per minute from this client [skip]: ").strip()
        beh = 0.0
        try:
            if rpm:
                beh = behaviour_score({"requests_per_minute": float(rpm), "unique_urls": 1,
                                       "repeated_requests": 0, "failed_requests": 0})
            print_result(analyze_request(models, method, url, ua, size, beh))
        except ValueError as err:
            print("Input problem:", err)


if __name__ == "__main__":
    main()
