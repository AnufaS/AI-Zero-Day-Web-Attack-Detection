# AI-Powered Zero-Day Web Attack Detection System

A beginner-friendly Machine Learning + Cybersecurity project (Python). It analyses HTTP requests and decides:
**NORMAL**, **KNOWN ATTACK**, **SUSPICIOUS**, or **POSSIBLE UNKNOWN / ZERO-DAY ATTACK** (a previously unseen pattern).

> Honest claim: *the system detects anomalous and previously unseen web-request behaviour that may indicate an unknown attack.*
> It does **not** detect all zero-day attacks.

## Problem statement
Signature rules and classifiers only recognise attacks they already know. Attackers invent new payloads. We combine a classifier (known attacks) with an anomaly detector (unusual traffic) and client-behaviour analysis, and let a decision engine weigh all evidence.

## Motivation / Objectives
Learn the full ML workflow (data -> features -> models -> evaluation -> app) on a security problem, and show why *one* model is not enough.
Objectives: build a feature engine; train Random Forest and Isolation Forest; add behaviour + uncertainty; build a decision engine; run an unseen-attack experiment; present results as a console detection summary; run fully offline in VS Code.

## System architecture
```
HTTP REQUEST -> FEATURE ENGINE -+-> ATTACK CLASSIFIER (Random Forest) --+
                                +-> ANOMALY DETECTOR (Isolation Forest)-+-> BEHAVIOUR + UNCERTAINTY -> DECISION ENGINE
                                                                                   -> NORMAL | SUSPICIOUS | KNOWN ATTACK | POSSIBLE UNKNOWN/ZERO-DAY
```

## Technologies
Python, pandas, numpy, scikit-learn, matplotlib, joblib (nothing else). No web server or frontend.

## Installation (Windows)
```bat
cd AI-Zero-Day-Web-Attack-Detection
python -m venv venv
venv\Scripts\activate            :: PowerShell: venv\Scripts\Activate.ps1  (if blocked: Set-ExecutionPolicy -Scope Process Bypass)
pip install -r requirements.txt  :: or: pip install pandas numpy scikit-learn matplotlib joblib
```
Linux/macOS: `python3 -m venv venv && source venv/bin/activate`.

## Usage
Run everything from the **project root folder**.
```bat
python run_all.py                       :: runs steps 2-12 in order (about a minute)
python main.py                          :: menu with all options (easiest in VS Code)
python prediction/predict.py            :: type your own requests
python simulation/simulate_traffic.py   :: simulated traffic + detection summary (no server needed)
python analyze_log.py [file.txt]        :: analyse a file of requests (default: sample_requests.txt)
```
Or run each step yourself (this is the learning order):

| Step | What | Command |
|---|---|---|
| 1 | Setup | install section above |
| 2 | Create dataset | `python data_generation/generate_dataset.py` |
| 3-4 | Preprocess, features, split | `python preprocessing/prepare_data.py` (test features: `python preprocessing/feature_engineering.py`) |
| 5 | EDA / plots | `python evaluation/evaluate_classifier.py` (class distribution figure) |
| 6 | Random Forest | `python models/train_classifier.py` |
| 7 | Isolation Forest | `python models/train_anomaly_detector.py` |
| 8 | Behaviour analysis | `python models/behaviour.py` |
| 9-10 | Uncertainty + decision engine | `python models/decision_engine.py`, `python evaluation/tune_decision_engine.py` |
| 11 | Unseen attack experiment | `python evaluation/unseen_attack_experiment.py` |
| 12 | Evaluation | `evaluate_classifier.py`, `evaluate_anomaly.py`, `evaluate_pipeline.py` in `evaluation/` |
| 13-14, 20 | Save/load + prediction | `python prediction/predict.py` |
| 15-17 | Monitor, simulated traffic, console summary | `python simulation/simulate_traffic.py`, `python analyze_log.py` |

Example (`python prediction/predict.py`, URL `/login?username=admin' OR '1'='1`):
```
Prediction     : SQL_INJECTION   (confidence ~94%, HIGH)
Anomaly score  : 1.00     Threat score: ~0.9 (risk HIGH)
FINAL          : KNOWN ATTACK - SQL INJECTION
```
Exact numbers can differ slightly if you regenerate data or change a seed.

## Project structure
```
config.py                 paths + settings (one place)
run_all.py                runs the whole pipeline
data_generation/          generate_dataset.py          (synthetic CSV)
preprocessing/            feature_engineering.py, prepare_data.py (features + leakage-safe split)
models/                   train_classifier.py, train_anomaly_detector.py, behaviour.py, decision_engine.py
prediction/               predict.py                   (load models, analyse a request)
evaluation/               evaluate_*.py, tune_decision_engine.py, unseen_attack_experiment.py, common.py
prediction/monitor.py     Monitor class: behaviour tracking + logging + console summary
simulation/               simulate_traffic.py          (in-process, no network)
main.py, analyze_log.py   menu launcher / batch analysis of a request file
.vscode/launch.json       Run/Debug entries for VS Code
saved_models/             attack_classifier.pkl, anomaly_detector.pkl, scaler.pkl, ...
reports/                  figures/, metrics.json, detection_log.csv
docs/VIVA.md             viva questions and answers
```
Differences from your suggested layout: `config.py`, `run_all.py`, `behaviour.py`, `prepare_data.py`, `evaluate_pipeline.py`, `simulation/` were added to keep each file small and single-purpose.

## Dataset
Synthetic, generated by `generate_dataset.py` (~5.6k rows: NORMAL, SQL_INJECTION, XSS, PATH_TRAVERSAL, COMMAND_INJECTION). Attack rows are harmless text strings built from 10 templates per class, with random values, optional extra parameters and URL-encoding.

| Column | Meaning / why it exists |
|---|---|
| method | GET/POST - POST is common for logins |
| url | path + query string - the main thing we analyse |
| user_agent | client software; scanners often identify themselves |
| request_size | bytes sent (the user can enter it; used as a feature) |
| response_size | bytes returned - **not** a feature (unknown before the server answers); kept for reference |
| label | NORMAL / ATTACK (binary view) |
| attack_type | the 5-class target |
| group_id | which template made the row - used to split data without leakage |

**Leakage prevention.** *Data leakage* = information from the test data seeping into training, which makes scores look better than reality. We split 70/15/15 **by template (`group_id`)**, so one payload family never sits in both train and test; `prepare_data.py` asserts this. Identical repeat visits to *normal static pages* (`/`, `/cart`) are legitimate, so they are spread across splits.

## Feature engineering (19 numbers per request)
Examples - `special_character_count`: payloads contain symbols (`' " < > ; |`) rare in normal URLs. `encoded_character_count`: `%27`-style encoding hides payloads. `sql_keyword_count`, `script_keyword_count`, `path_traversal_count`, `command_keyword_count`: counts of suspicious words per attack family. `entropy`: randomness of characters. `*_ratio`: size-independent versions. `quote_count`: SQL injection needs quotes. `user_agent_is_tool`: scanner/CLI client. Every feature is commented in `preprocessing/feature_engineering.py`. Keywords are counted after URL-decoding twice so encoded payloads are not hidden.

## Random Forest
Many decision trees, each trained on a random slice of data, **vote**; vote shares become class probabilities. Good for tabular numeric features, needs little tuning, gives feature importance. Output: predicted class, 5 probabilities, confidence.

## Isolation Forest
Anomaly detection = finding things unlike the normal data. The classifier only knows classes it was shown; a new attack gets forced into the nearest known class (or into NORMAL). Isolation Forest is trained on **normal traffic only**; unusual points are isolated by few random splits. The raw output is *not a probability*. We map it to 0..1 using normal validation data: median normal score -> 0.0, 99th percentile of normal -> 1.0 (see `train_anomaly_detector.py`). 1.0 means "more unusual than 99% of normal traffic".

## Behaviour analysis
Per client, over the last 60 s: `requests_per_minute`, `unique_urls`, `repeated_requests`, `failed_requests` -> `behaviour_score` (0..1). A bot doing 200 failed logins/min scores ~1.0 though each single request looks harmless. In the console version, clients are told apart by the client name passed to `Monitor.process()`.

## Uncertainty
Confidence = highest class probability: HIGH >= 0.70, MEDIUM 0.50-0.70, LOW < 0.50. Forest probabilities are vote shares and not guaranteed calibrated. `train_classifier.py` also fits `CalibratedClassifierCV` and prints log-loss for both; on our data calibration did **not** help (0.089 vs 0.083), so the plain forest is used. With real data, check again.

## Decision engine
`threat = w_attack*(1 - P(NORMAL)) + w_anomaly*anomaly + w_behaviour*behaviour`. Defaults 0.50/0.35/0.15; `tune_decision_engine.py` compares weight pairs on **validation** data (F2 score, recall counts more) and saved 0.55/0.30/0.15. These are *not* scientifically optimal - validation contains only known attacks, so tuning favours the classifier. Rules (in order):
1. predicted attack class + HIGH confidence + threat >= 0.30 -> **KNOWN ATTACK**
2. anomaly >= 0.80 + confidence not HIGH -> **POSSIBLE UNKNOWN/ZERO-DAY ATTACK**
3. threat >= 0.30, or anomaly >= 0.80, or behaviour >= 0.50 -> **SUSPICIOUS**
4. otherwise **NORMAL**. Risk level: LOW <0.30, MEDIUM 0.30-0.70, HIGH >=0.70.

## Unseen attack experiment
COMMAND_INJECTION is removed from the scaler and Random Forest training (asserted in code); the Isolation Forest only sees normal traffic. Then all 621 command-injection requests are tested.

| Result | Value |
|---|---|
| Classifier alone calls them NORMAL | 40.6% (missed!) |
| Full system: flagged as not NORMAL | 99.7% |
| ...of which POSSIBLE UNKNOWN/ZERO-DAY | 90.8% |
| Mean anomaly score: unseen attack vs normal | 0.98 vs 0.12 |
| False alarms on normal test traffic | 1.7% |

Reading: the classifier is confused (low confidence or wrongly "NORMAL"), but the anomaly detector sees the requests as very unusual, so the engine raises "possible unknown". **Limitations:** the "unseen" class is still synthetic and shares symbols (`; | &&`) with SQL injection, so it is easy to find anomalous; a real zero-day may look completely normal. This does not prove real-world zero-day detection.

## Results (test set, synthetic data, template-held-out split - run `python run_all.py` to reproduce)
| Model | Result |
|---|---|
| Random Forest (5 classes) | accuracy 0.884, macro precision 0.956, macro recall 0.845, macro F1 0.880; 107 attacks called NORMAL, 2 normal flagged |
| Isolation Forest (threshold 0.8) | precision 0.982, recall 0.752, F1 0.852, FPR 0.017, FNR 0.248, ROC-AUC 0.967 |
| Full pipeline (alert = not NORMAL) | precision 0.981, recall 0.909, F1 0.944, FPR 0.021, FNR 0.091 |

Why the classifier scores "only" 0.88: whole held-out templates were never seen in training. Two (`/download?file=/etc/passwd` and the single-quote comment `admin'--`) have weak signals and were called NORMAL - **this is a realistic weakness** and exactly why the anomaly detector and SUSPICIOUS category exist. Top features: `sql_keyword_count`, `script_keyword_count`, `path_traversal_count`, `command_keyword_count`, `special_character_count` - each family's own keyword counter matters most. Figures are in `reports/figures/` (class distribution, confusion matrix, feature importance, per-class performance, anomaly score distribution, threat score distribution, unseen experiment).

**Why recall / false negatives matter:** a false negative (attack called normal) lets an attacker in; a false positive (normal flagged) only annoys a user and costs analyst time. Security teams therefore prioritise recall, then work to reduce false positives.

## Safety
Everything runs locally and offline. Attack-like strings are only analysed as text; nothing is sent over a network or executed. Do not use these ideas against systems you do not own.

## Limitations
- Synthetic data may not represent real traffic; results are optimistic.
- Attackers can modify payloads (obfuscation, rare encodings) to evade feature-based detection.
- Anomaly detection creates false positives (rare-but-legitimate requests).
- Zero-day detection cannot be guaranteed; unknown attacks can look normal.
- Quality depends heavily on dataset quality; model quality on a feature design that only sees URL/query text (not POST bodies/headers fully).
- Real deployment needs privacy controls (URLs contain personal data) and access control for logs.
- Model drift: traffic changes, so models must be monitored and retrained.
- Weights/thresholds are tuned on synthetic data, not proven optimal.
- Lesson learned during development: a detector trained on a *too small / too deduplicated* normal set flags ordinary pages; normal traffic must represent real usage.

## Future enhancements
Real-time log ingestion; Zeek / Suricata integration; Nginx/Apache log analysis; deep learning; transformer request embeddings; online learning; concept-drift detection; explainable AI (SHAP); SIEM integration; advanced uncertainty (conformal prediction, ensembles). Not implemented here.

## Conclusion
Classification handles known attacks; anomaly detection adds a safety net for unfamiliar patterns; behaviour and confidence provide context; the decision engine turns it into an explainable verdict. Treat "possible unknown" as a lead for a human analyst, not proof of a zero-day.
