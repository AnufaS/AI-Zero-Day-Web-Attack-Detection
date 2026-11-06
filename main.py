"""
Main menu - run this in VS Code (press the Run button, or:  python main.py)
"""
import subprocess
import sys

MENU = [
    ("Train everything (dataset + models + evaluation)", "run_all.py"),
    ("Analyse a request interactively",                  "prediction/predict.py"),
    ("Run traffic simulation + detection summary",       "simulation/simulate_traffic.py"),
    ("Analyse requests from sample_requests.txt",        "analyze_log.py"),
    ("Show behaviour analysis demo",                     "models/behaviour.py"),
]

while True:
    print("\n=== AI Zero-Day Web Attack Detection ===")
    for i, (name, _) in enumerate(MENU, 1):
        print(f" {i}. {name}")
    print(" q. Quit")
    choice = input("Choose: ").strip().lower()
    if choice in ("q", "quit", "exit"):
        break
    if choice.isdigit() and 1 <= int(choice) <= len(MENU):
        subprocess.call([sys.executable, MENU[int(choice) - 1][1]])
    else:
        print("Invalid choice.")
