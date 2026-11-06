# Viva preparation (simple answers)

## Basic
**What is a zero-day attack?** An attack using a weakness (or technique) the defenders have not seen or patched yet - they have had "zero days" to prepare.
**What is anomaly detection?** Finding data that does not look like what is usual, without needing examples of every attack.
**What is supervised learning?** Learning from labelled examples (request -> NORMAL/SQL_INJECTION...). It can only recognise classes it was taught.
**Why Random Forest?** Works well on numeric tabular features, robust, little tuning, gives probabilities and feature importance, easy to explain.
**Why Isolation Forest?** Needs only normal data, fast, no distance calculations, and "easy to isolate = unusual" is simple to explain.
**What is feature engineering?** Turning raw text (a URL) into meaningful numbers (length, special characters, keyword counts) that a model can learn from.

## Intermediate
**Why can't Random Forest alone detect unknown attacks?** It must pick one of the classes it was trained on. A new attack gets mapped to the closest known class or to NORMAL, often with wrong confidence. In our experiment the classifier alone called 40.6% of unseen command injections NORMAL.
**How does Isolation Forest work?** It builds random trees with random splits. Odd points get isolated in few splits (short path); typical points need many splits. Short average path = more anomalous.
**What is an anomaly score?** A number saying how unusual a request is. Ours is 0..1, scaled using normal validation traffic (median -> 0, 99th percentile -> 1). It is not a probability of attack.
**What is confidence?** Here: the highest class probability the classifier outputs (HIGH/MEDIUM/LOW).
**Probability vs confidence?** Probability is the model's output per class. Confidence is how much we trust the top one; forest probabilities may be mis-calibrated (e.g. "90%" might be right only 75% of the time), so we calibrate/check.
**Why is recall important?** Recall = share of real attacks caught. Missing an attack (false negative) is usually costlier than a false alarm.
**False positive?** A normal request flagged as an attack. **False negative?** A real attack called normal.

## Advanced
**How did you simulate zero-day detection?** Removed COMMAND_INJECTION from scaler + classifier training, kept the Isolation Forest normal-only, then tested all command-injection requests and compared classifier output, confidence, anomaly score and final decision. It is a simulation, not proof.
**How did you prevent data leakage?** Split by template (`group_id`) so a payload family is in exactly one of train/val/test; scalers/thresholds fit on training/validation only; an assertion checks it; the unseen class is excluded from every fitted component of the experiment.
**How would concept drift affect it?** If normal traffic changes (new site features, new users), more normal requests look anomalous (false positives) and the classifier's accuracy falls. Monitor score distributions, retrain on recent data, add drift detection.
**How could an attacker evade it?** Blend payloads to look normal (low special-character count, split across parameters, unusual encodings), or slow-rate attacks that avoid the behaviour score. Feature-based detection can be bypassed - so use it with a WAF, patching and input validation.
**How would you reduce false positives?** More/more realistic normal data, tune thresholds on real validation data, allow-list known tools, require two signals before alerting, analyst feedback loop.
**How would you calibrate the classifier?** `CalibratedClassifierCV` (sigmoid or isotonic) on held-out data, then compare log-loss/reliability curves (we did; it did not help here).
**How would it work on real HTTP traffic?** Parse Nginx/Apache/Zeek logs into method/url/user-agent/size, run the same feature engine, keep per-client behaviour windows, send decisions to a dashboard/SIEM. Needs real labelled data and privacy controls.
**What if normal behaviour changes?** Same as drift - retrain, use rolling windows, review thresholds.
**How would you deploy in production?** As a monitor-only service first (no blocking), behind an API, with logging, model versioning, scheduled retraining, alert triage, and human review before any automatic blocking.
