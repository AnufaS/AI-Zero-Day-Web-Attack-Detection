"""
Analyse a text file of requests (one URL per line, optional 'METHOD URL').
Usage:   python analyze_log.py my_requests.txt
Without an argument it analyses sample_requests.txt.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prediction.monitor import Monitor


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_requests.txt")
    if not os.path.exists(path):
        print("File not found:", path)
        sys.exit(1)
    mon = Monitor()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            method, url = ("GET", line)
            first = line.split(" ", 1)
            if len(first) == 2 and first[0].upper() in ("GET", "POST"):
                method, url = first[0].upper(), first[1]
            try:
                r = mon.process("file", "file-client", method, url)
                print(f"[{r['decision']:32s}] threat={r['threat_score']:.2f}  {method} {url[:60]}")
            except ValueError as err:
                print(f"[skipped] {url[:50]}  ({err})")
    mon.summary()


if __name__ == "__main__":
    main()
