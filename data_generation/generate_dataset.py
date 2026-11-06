"""
STEP 2 - Create a SYNTHETIC dataset of web requests.

Run from the project root:   python data_generation/generate_dataset.py

Columns written to dataset/raw/web_requests.csv
  method         GET / POST
  url            path + query string, e.g. /products?id=10
  user_agent     the browser/tool name sent by the client
  request_size   bytes in the request
  response_size  bytes the server sent back (NOT used as a feature: we won't
                 know it before the server answers - kept for reference only)
  label          NORMAL or ATTACK
  attack_type    NORMAL, SQL_INJECTION, XSS, PATH_TRAVERSAL, COMMAND_INJECTION
  group_id       which "template" made the row. Used later so near-identical
                 payloads never end up in both train and test (data leakage).

All payloads are harmless TEXT strings. Nothing is sent anywhere.
"""
import os
import random
import sys
from urllib.parse import quote

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from config import RAW_CSV, RANDOM_STATE, ensure_dirs

rng = random.Random(RANDOM_STATE)

WORDS = ["phone", "laptop", "shoes", "camera", "book", "watch", "table", "lamp",
         "headphones", "backpack", "tv", "keyboard", "mouse", "chair", "bottle"]
# Normal searches that LOOK a little like attacks - real users type these too
TRICKY_WORDS = ["union jack flag", "drop shipping", "update password", "select shoes",
                "delete account help", "cat food", "dir hair dye", "ping pong table",
                "echo dot", "rm 500 spare part", "windows laptop", "tom and jerry"]
USERS = ["alice", "bob", "admin", "carol", "dave", "eve", "root", "guest", "john"]
TABLES = ["users", "orders", "accounts", "products", "customers"]

BROWSER_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) Firefox/121.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0) Mobile Safari/604.1",
]
TOOL_UAS = ["curl/8.4.0", "python-requests/2.31", "sqlmap/1.7", "Nikto/2.5", "Wget/1.21"]


def n():
    return rng.randint(1, 9999)


def add_extra_param(url, chance):
    """Real requests often carry extra harmless parameters (session id, language...).
    Adding them makes rows from the same template differ from each other."""
    if rng.random() < chance:
        name = rng.choice(["ref", "t", "sid", "lang", "utm"])
        value = rng.choice(["en", "fr", "home", "ad"]) + str(rng.randint(1, 999))
        url += ("&" if "?" in url else "?") + f"{name}={value}"
    return url


def encode_some(url):
    """Attackers often URL-encode payloads. Randomly leave raw / partly / fully encoded."""
    path, _, query = url.partition("?")
    if not query:
        return url
    r = rng.random()
    if r < 0.5:
        query = query.replace(" ", "%20")
    elif r < 0.8:
        query = quote(query, safe="=&'<>()/\\")
    else:
        query = quote(query, safe="=&")
    return path + "?" + query


# ---------- NORMAL traffic templates (each lambda makes one random URL) ----------
NORMAL_TEMPLATES = [
    lambda: "/",
    lambda: "/home",
    lambda: "/products",
    lambda: f"/products?id={n()}",
    lambda: f"/products?category={rng.choice(WORDS)}&page={rng.randint(1, 30)}",
    lambda: f"/search?q={rng.choice(WORDS)}",
    lambda: f"/search?q={rng.choice(WORDS)}+{rng.choice(WORDS)}",
    lambda: f"/search?q={rng.choice(TRICKY_WORDS).replace(' ', '+')}",
    lambda: "/profile",
    lambda: f"/profile?user={rng.choice(USERS)}{rng.randint(1, 500)}",
    lambda: "/cart",
    lambda: f"/cart?item={n()}&qty={rng.randint(1, 5)}",
    lambda: f"/login?username={rng.choice(USERS)}{n()}&password=pass{rng.randint(1000, 99999)}",
    lambda: "/about",
    lambda: f"/products/{n()}",
]

# ---------- ATTACK templates: (class, template) ----------
SQLI = [
    lambda: f"/login?username={rng.choice(USERS)}' OR '1'='1" + rng.choice(["", "&password=x"]),
    lambda: f"/login?username={rng.choice(USERS)}'--" + rng.choice(["", "&password="]),
    lambda: f"/products?id={n()} UNION SELECT username,password FROM {rng.choice(TABLES)}--",
    lambda: f"/search?q={rng.choice(WORDS)}'; DROP TABLE {rng.choice(TABLES)};--",
    lambda: f"/products?id={n()} OR 1=1",
    lambda: f"/products?id={n()} AND SLEEP({rng.randint(2, 9)})--",
    lambda: f"/search?q=' UNION SELECT table_name FROM information_schema.tables--",
    lambda: f"/products?id=-1 UNION SELECT NULL,NULL,{n()}--",
    lambda: f"/profile?user={rng.choice(USERS)}'; INSERT INTO {rng.choice(TABLES)} VALUES('{rng.choice(USERS)}','pw');--",
    lambda: f"/login?username=admin\" OR \"{n()}\"=\"{n()}" + rng.choice(["", "&password=x"]),
]
XSS = [
    lambda: f"/search?q=<script>alert({n()})</script>",
    lambda: f"/search?q=<img src=x onerror=alert({n()})>",
    lambda: f"/profile?name=<svg onload=alert({n()})>",
    lambda: f"/search?q=<body onload=alert('{rng.choice(WORDS)}')>",
    lambda: "/products?id=\"><script>alert(document.cookie)</script>",
    lambda: f"/search?q=<iframe src=javascript:alert({n()})>",
    lambda: f"/search?q=javascript:alert({n()})",
    lambda: f"/profile?bio=<a href=\"javascript:alert({n()})\">click</a>",
    lambda: "/search?q=<script>document.location='http://localhost/c?='+document.cookie</script>",
    lambda: f"/search?q=<input onfocus=alert({n()}) autofocus>",
]
TRAVERSAL = [
    lambda: f"/products?file={'../' * rng.randint(2, 7)}etc/passwd",
    lambda: f"/download?file={'..\\' * rng.randint(2, 6)}windows\\win.ini",
    lambda: f"/products?page={'....//' * rng.randint(2, 5)}etc/passwd",
    lambda: f"/view?path={'../' * rng.randint(3, 8)}etc/shadow",
    lambda: f"/products?file={'%2e%2e%2f' * rng.randint(2, 6)}etc%2fpasswd",
    lambda: f"/files?name={'../' * rng.randint(2, 5)}boot.ini",
    lambda: "/download?file=/etc/passwd",
    lambda: f"/view?path={'..%2f' * rng.randint(3, 7)}etc%2fpasswd",
    lambda: f"/download?file={'../' * rng.randint(3, 7)}windows/system32/drivers/etc/hosts",
    lambda: f"/files?doc={'../' * rng.randint(1, 4)}config/secrets.txt",
]
COMMAND = [
    lambda: f"/search?q={rng.choice(WORDS)}; ls -la",
    lambda: "/ping?host=127.0.0.1 && cat /etc/passwd",
    lambda: "/ping?host=127.0.0.1 | whoami",
    lambda: "/search?q=$(whoami)",
    lambda: "/ping?host=localhost`id`",
    lambda: "/lookup?domain=example.com; wget http://localhost/x.sh",
    lambda: "/ping?host=127.0.0.1; uname -a",
    lambda: "/ping?host=127.0.0.1 && ipconfig /all",
    lambda: f"/convert?file=a{n()}.png; rm -rf /tmp/demo",
    lambda: "/ping?host=127.0.0.1 | dir C:\\",
]
ATTACKS = {"SQL_INJECTION": SQLI, "XSS": XSS, "PATH_TRAVERSAL": TRAVERSAL,
           "COMMAND_INJECTION": COMMAND}


def make_row(url, attack_type, group_id):
    is_attack = attack_type != "NORMAL"
    path = url.split("?")[0]
    method = "POST" if (path == "/login" and rng.random() < 0.7) else "GET"
    # attackers use scanning tools more often than normal users - but not always
    tool_chance = 0.3 if is_attack else 0.05
    ua = rng.choice(TOOL_UAS) if rng.random() < tool_chance else rng.choice(BROWSER_UAS)
    request_size = len(url) + rng.randint(200, 500) + (rng.randint(20, 80) if method == "POST" else 0)
    response_size = rng.randint(100, 1500) if is_attack else rng.randint(1500, 60000)
    return {"method": method, "url": url, "user_agent": ua, "request_size": request_size,
            "response_size": response_size, "label": "ATTACK" if is_attack else "NORMAL",
            "attack_type": attack_type, "group_id": group_id}


def main():
    ensure_dirs()
    rows = []
    # NORMAL: 3000 attempts; duplicates removed below.
    for _ in range(3000):
        url = add_extra_param(NORMAL_TEMPLATES[rng.randrange(len(NORMAL_TEMPLATES))](), 0.25)
        # normal rows are grouped by their exact URL (same URL never in two splits)
        # Pages WITH parameters are grouped by exact URL. Static pages (/, /cart ...) are visited
        # by everyone all the time, so their rows are spread over 20 buckets (legit repeat visits).
        group = "N:" + url if "?" in url else f"N:{url}:{rng.randint(0, 19)}"
        rows.append(make_row(url, "NORMAL", group))
    # ATTACKS: ~500 attempts per class, spread over the templates
    for attack_type, templates in ATTACKS.items():
        for _ in range(700):
            idx = rng.randrange(len(templates))
            url = encode_some(add_extra_param(templates[idx](), 0.8))
            rows.append(make_row(url, attack_type, f"{attack_type}:{idx}"))

    df = pd.DataFrame(rows)
    is_normal = df["attack_type"] == "NORMAL"
    # Repeat visits to the same normal page are real traffic -> keep them (only drop exact copies).
    # Attack rows with the same URL are redundant -> drop them.
    df = pd.concat([df[is_normal].drop_duplicates(subset=["url", "user_agent", "request_size"]),
                    df[~is_normal].drop_duplicates(subset=["url"])])
    df = df.sample(frac=1, random_state=RANDOM_STATE)
    df.to_csv(RAW_CSV, index=False)
    print(f"Saved {len(df)} rows to {RAW_CSV}\n")
    print(df["attack_type"].value_counts().to_string())
    print("\nSample rows:")
    print(df[["method", "url", "attack_type"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
