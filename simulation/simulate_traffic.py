"""
Generate normal + safe attack-like traffic and analyse it with the full pipeline.
No web server needed - everything runs inside this script.

Run from the project root:   python simulation/simulate_traffic.py

SAFETY: the 'attacks' are just text strings that are fed to the ML models.
Nothing is sent over the network and nothing is executed.
Results are also saved to reports/detection_log.csv
"""
import os
import random
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prediction.monitor import Monitor

NORMAL = ["/", "/products", "/products?id=10", "/products?id=42", "/search?q=phone", "/search?q=laptop+bag",
          "/profile", "/cart", "/cart?item=7&qty=2", "/products?category=shoes&page=2"]
KNOWN = ["/login?username=admin' OR '1'='1", "/products?id=5 UNION SELECT username,password FROM users--",
         "/search?q=<script>alert(1)</script>", "/search?q=<img src=x onerror=alert(2)>",
         "/products?file=../../../etc/passwd", "/search?q=phone; ls -la"]
UNSEEN_STYLE = ["/search?q={{7*7}}", "/search?q=${7*7}", "/products?id=1)(uid=*))(|(uid=*",
                "/search?q=%00%00%00%00%00%00AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
                "/products?x=<!ENTITY xxe SYSTEM \"file:///etc/hostname\">"]


def show(title, rows):
    print(f"\n{title}")
    for client, method, url, r in rows:
        print(f"  [{r['decision']:32s}] threat={r['threat_score']:.2f} anomaly={r['anomaly_score']:.2f} "
              f"{method} {url[:55]}")


def main():
    mon = Monitor()
    out = []
    print("1) Normal users (varied traffic)...")
    for i in range(30):
        u = random.choice(NORMAL)
        out.append((f"user-{i % 3}", "GET", u, mon.process("sim", f"user-{i % 3}", "GET", u)))
    n_normal = sum(1 for *_, r in out if r["decision"] == "NORMAL")
    print(f"   {n_normal}/30 correctly NORMAL")

    known = [("attacker-1", "GET", p, mon.process("sim", "attacker-1", "GET", p)) for p in KNOWN]
    show("2) Known attack-like requests...", known)

    unseen = [("attacker-2", "GET", p, mon.process("sim", "attacker-2", "GET", p)) for p in UNSEEN_STYLE]
    show("3) Unusual patterns the classifier was never trained on...", unseen)

    print("\n4) A bot hammering /login with wrong passwords (behaviour analysis)...")
    last = None
    for i in range(80):
        last = mon.process("sim", "bot-1", "POST", "/login?username=alice&password=wrong", status=401)
        if i in (0, 19, 79):
            print(f"   request #{i + 1:2d}: behaviour={last['behaviour_score']:.2f} "
                  f"threat={last['threat_score']:.2f} -> {last['decision']}")
    mon.summary()
    print("Full log saved to reports/detection_log.csv")


if __name__ == "__main__":
    main()
