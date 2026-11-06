"""
STEP 8 - BEHAVIOUR ANALYSIS (looks at a CLIENT over time, not one request).

One request to /login is normal. 200 failed requests to /login in a minute is not.
We remember the last 60 seconds of requests per client and compute:
  requests_per_minute, unique_urls, repeated_requests, failed_requests
"""
import time
from collections import defaultdict, deque


class BehaviourTracker:
    def __init__(self, window_seconds=60):
        self.window = window_seconds
        self.history = defaultdict(deque)   # client_id -> deque of (time, url, status)

    def record(self, client_id, url, status=200, timestamp=None):
        self.history[client_id].append((timestamp if timestamp is not None else time.time(), url, status))

    def features(self, client_id, now=None):
        now = now if now is not None else time.time()
        events = self.history[client_id]
        while events and now - events[0][0] > self.window:   # forget old requests
            events.popleft()
        count = len(events)
        unique = len({url for _, url, _ in events})
        failed = sum(1 for _, _, status in events if status >= 400)
        return {
            "requests_per_minute": count * 60.0 / self.window,
            "unique_urls": unique,
            "repeated_requests": count - unique,
            "failed_requests": failed,
        }


def behaviour_score(f):
    """Turn behaviour features into one number 0..1. Weights are a simple, explainable
    choice (rate matters most), not a scientifically proven optimum."""
    count = f["unique_urls"] + f["repeated_requests"]
    rate_part = min(f["requests_per_minute"] / 120.0, 1.0)            # 120/min -> maxed out
    repeat_part = f["repeated_requests"] / count if count >= 10 else 0.0
    failed_part = min(f["failed_requests"] / 15.0, 1.0)                # 15 failures -> maxed
    return round(0.5 * rate_part + 0.2 * repeat_part + 0.3 * failed_part, 4)


if __name__ == "__main__":
    t = BehaviourTracker()
    for i in range(5):
        t.record("normal-user", f"/products?id={i}", 200)
    for i in range(200):
        t.record("bot", "/login?u=admin", 401)
    for who in ("normal-user", "bot"):
        f = t.features(who)
        print(who, f, "-> behaviour score", behaviour_score(f))
