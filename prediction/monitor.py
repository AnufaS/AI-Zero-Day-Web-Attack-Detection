"""
monitor.py - console monitor: runs the full detection pipeline and prints a text summary.

Monitor.process(...) runs the full pipeline for one request:
    behaviour tracking -> features -> classifier + anomaly detector -> decision engine
and stores the result in an in-memory log + reports/detection_log.csv.
Monitor.summary() prints the detection counts and recent requests as plain text.
"""
import csv
import os
import sys
from collections import Counter, deque
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REPORT_DIR
from models.behaviour import BehaviourTracker, behaviour_score
from prediction.predict import analyze_request, load_models

LOG_FILE = os.path.join(REPORT_DIR, "detection_log.csv")
DEFAULT_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0 Safari/537.36"
DECISIONS = ["NORMAL", "KNOWN ATTACK", "SUSPICIOUS", "POSSIBLE UNKNOWN/ZERO-DAY ATTACK"]


class Monitor:
    def __init__(self, write_csv=True):
        self.models = load_models()
        self.tracker = BehaviourTracker()
        self.logs = deque(maxlen=500)          # newest first
        self.write_csv = write_csv

    def process(self, source, client, method, url, ua=DEFAULT_UA, size=None, status=200):
        """Analyse one request. Returns the full result dict (raises ValueError on bad input)."""
        self.tracker.record(client, url, status)
        beh = behaviour_score(self.tracker.features(client))
        r = analyze_request(self.models, method, url, ua, size, beh)
        row = {"time": datetime.now().strftime("%H:%M:%S"), "source": source, "client": client,
               "request": f"{r['method']} {r['url']}", "prediction": r["prediction"],
               "confidence": f"{r['max_probability']:.0%}", "anomaly": round(r["anomaly_score"], 2),
               "behaviour": round(r["behaviour_score"], 2), "threat": round(r["threat_score"], 2),
               "decision": r["decision"]}
        self.logs.appendleft(row)
        if self.write_csv:
            os.makedirs(REPORT_DIR, exist_ok=True)
            new_file = not os.path.exists(LOG_FILE)
            with open(LOG_FILE, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new_file:
                    w.writeheader()
                w.writerow(row)
        return r

    def counts(self):
        c = Counter(row["decision"] for row in self.logs)
        return {d: c.get(d, 0) for d in DECISIONS}

    def summary(self, last=8):
        c = self.counts()
        print("\n" + "=" * 78)
        print(f"DETECTION SUMMARY   total requests analysed: {len(self.logs)}")
        for d in DECISIONS:
            print(f"   {d:34s} {c[d]}")
        print("-" * 78)
        print(f"Last {last} requests (newest first):")
        for row in list(self.logs)[:last]:
            print(f"  {row['time']} {row['client']:11s} {row['decision']:32s} "
                  f"threat={row['threat']:.2f} | {row['request'][:60]}")
        print("=" * 78)
