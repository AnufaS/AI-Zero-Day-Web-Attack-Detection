"""
STEP 9 + 10 - UNCERTAINTY + CUSTOM DECISION ENGINE.

Inputs : classifier probabilities, anomaly score (0..1), behaviour score (0..1)
Output : NORMAL / SUSPICIOUS / KNOWN ATTACK / POSSIBLE UNKNOWN/ZERO-DAY ATTACK

threat_score = w_attack*attack_score + w_anomaly*anomaly_score + w_behaviour*behaviour_score
   attack_score = 1 - P(NORMAL)  ("how sure is the classifier that it is NOT normal")

The weights are NOT magic numbers. Defaults are a starting guess; run
evaluation/tune_decision_engine.py to compare weight choices on VALIDATION data.
Even then they are only 'good for this synthetic dataset', not proven optimal.

Confidence = highest class probability (HIGH / MEDIUM / LOW). Forest probabilities
are vote shares and may not be perfectly calibrated (see CalibratedClassifierCV).
"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DECISION_CONFIG_JSON

DEFAULT_CONFIG = {
    "weights": {"attack": 0.50, "anomaly": 0.35, "behaviour": 0.15},
    "suspicious_threshold": 0.30,   # threat >= this  -> at least SUSPICIOUS
    "high_threshold": 0.70,         # threat >= this  -> risk level HIGH
    "anomaly_high": 0.80,           # anomaly >= this -> 'highly unusual'
    "behaviour_high": 0.50,         # behaviour >= this -> client acts like a bot / scanner
    "confidence_high": 0.70,        # max probability >= this -> HIGH confidence
    "confidence_low": 0.50,         # max probability <  this -> LOW confidence
}


def load_config():
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if os.path.exists(DECISION_CONFIG_JSON):
        with open(DECISION_CONFIG_JSON) as f:
            cfg.update(json.load(f))
    return cfg


def confidence_level(max_prob, cfg):
    if max_prob >= cfg["confidence_high"]:
        return "HIGH"
    if max_prob < cfg["confidence_low"]:
        return "LOW"
    return "MEDIUM"


def threat_score(attack_score, anomaly_score, behaviour_score, weights):
    return (weights["attack"] * attack_score + weights["anomaly"] * anomaly_score
            + weights["behaviour"] * behaviour_score)


def decide(probabilities, anomaly, behaviour, cfg=None):
    """probabilities: dict {class_name: probability}. Returns a result dict."""
    cfg = cfg or load_config()
    predicted = max(probabilities, key=probabilities.get)
    max_prob = probabilities[predicted]
    conf = confidence_level(max_prob, cfg)
    attack_score = 1.0 - probabilities.get("NORMAL", 0.0)
    threat = threat_score(attack_score, anomaly, behaviour, cfg["weights"])
    risk = "HIGH" if threat >= cfg["high_threshold"] else \
           "MEDIUM" if threat >= cfg["suspicious_threshold"] else "LOW"

    anomalous = anomaly >= cfg["anomaly_high"]
    if predicted != "NORMAL" and conf == "HIGH" and threat >= cfg["suspicious_threshold"]:
        decision = "KNOWN ATTACK"
        detail = f"KNOWN ATTACK - {predicted.replace('_', ' ')}"
        why = "Classifier is confident this matches an attack type it was trained on."
    elif anomalous and conf != "HIGH":
        decision = "POSSIBLE UNKNOWN/ZERO-DAY ATTACK"
        detail = "POSSIBLE UNKNOWN / ZERO-DAY ATTACK (previously unseen pattern)"
        why = "Request is highly unusual AND the classifier is not confident about any class."
    elif threat >= cfg["suspicious_threshold"] or anomalous or behaviour >= cfg["behaviour_high"]:
        decision = "SUSPICIOUS"
        detail = "SUSPICIOUS - needs review"
        why = ("Client behaviour is abnormal." if behaviour >= cfg["behaviour_high"] else
               "Request is unusual or partly matches an attack, but evidence is not strong enough.")
    else:
        decision = "NORMAL"
        detail = "NORMAL"
        why = "Looks like ordinary traffic."

    return {"prediction": predicted, "max_probability": max_prob, "confidence_level": conf,
            "probabilities": probabilities, "attack_score": attack_score,
            "anomaly_score": float(anomaly), "behaviour_score": float(behaviour),
            "threat_score": float(threat), "risk_level": risk,
            "decision": decision, "decision_detail": detail, "reason": why}


if __name__ == "__main__":
    cases = {
        "confident SQLi": ({"NORMAL": .03, "SQL_INJECTION": .91, "XSS": .02, "PATH_TRAVERSAL": .03, "COMMAND_INJECTION": .01}, .82, 0.0),
        "uncertain + very odd": ({"NORMAL": .35, "SQL_INJECTION": .18, "XSS": .27, "PATH_TRAVERSAL": .10, "COMMAND_INJECTION": .10}, .96, 0.0),
        "plain normal": ({"NORMAL": .97, "SQL_INJECTION": .01, "XSS": .01, "PATH_TRAVERSAL": .005, "COMMAND_INJECTION": .005}, .05, 0.0),
    }
    for name, (p, a, b) in cases.items():
        r = decide(p, a, b, DEFAULT_CONFIG)
        print(f"{name:22s} -> threat {r['threat_score']:.2f} | {r['decision_detail']}")
