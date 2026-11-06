"""
STEP 4 - FEATURE ENGINE: turn one HTTP request into a row of numbers.

Machine-learning models only understand numbers, not text like "/login?id=1 OR 1=1".
So we measure properties of the request. Each feature below says WHY it helps.

Quick test:   python preprocessing/feature_engineering.py
"""
import math
import re
import sys
import os
from collections import Counter
from urllib.parse import parse_qsl, unquote, urlsplit

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

VALID_METHODS = {"GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"}

FEATURE_NAMES = [
    "url_length", "query_length", "parameter_count", "parameter_length",
    "digit_count", "special_character_count", "encoded_character_count", "quote_count",
    "sql_keyword_count", "script_keyword_count", "path_traversal_count",
    "command_keyword_count", "entropy", "digit_ratio", "special_character_ratio",
    "alphabetic_ratio", "method_is_post", "request_size", "user_agent_length",
    "user_agent_is_tool",
]

# Patterns (re.I = ignore upper/lower case). Each one is a "suspicious word list".
SQL_RE = re.compile(
    r"\b(select|union|insert|update|delete|drop|sleep|benchmark|information_schema|having)\b"
    r"|\b(or|and)\b\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+"   # OR 1=1 / AND 'a'='a'
    r"|--|/\*", re.I)
SCRIPT_RE = re.compile(
    r"<\s*/?\s*(script|img|svg|iframe|body|input|a)\b|javascript:"
    r"|\bon(error|load|focus|click|mouseover)\s*=|alert\s*\(|document\.(cookie|location)", re.I)
TRAVERSAL_RE = re.compile(
    r"\.\./|\.\.\\|\.\.//|/etc/|\\windows\\|system32|boot\.ini|win\.ini|passwd|shadow", re.I)
CMD_OPERATOR_RE = re.compile(r"(;|\|\||\||&&|`|\$\()\s*[a-z/]", re.I)
CMD_WORD_RE = re.compile(
    r"\b(ls|cat|whoami|uname|ping|wget|curl|rm|dir|ipconfig|ifconfig|netstat|"
    r"chmod|bash|sh|nc|powershell|cmd|echo)\b", re.I)
TOOL_UA_RE = re.compile(r"sqlmap|nikto|curl|wget|python|scanner|nmap|havij", re.I)


def validate_request(method, url, user_agent="", request_size=None):
    """Check user input and return cleaned values. Raises ValueError with a friendly message."""
    if method is None or str(method).strip() == "":
        raise ValueError("HTTP method is empty. Use GET or POST.")
    method = str(method).strip().upper()
    if method not in VALID_METHODS:
        raise ValueError(f"'{method}' is not a valid HTTP method. Try one of: {sorted(VALID_METHODS)}")
    if url is None or str(url).strip() == "":
        raise ValueError("URL is empty. Example: /products?id=10")
    url = str(url).strip()
    if url.startswith(("http://", "https://")):          # allow full URLs, keep path + query
        parts = urlsplit(url)
        url = parts.path or "/"
        if parts.query:
            url += "?" + parts.query
    if not url.startswith("/"):
        raise ValueError("URL must start with '/'. Example: /search?q=phone")
    if len(url) > 8000:
        raise ValueError("URL is too long (over 8000 characters).")
    if request_size in (None, ""):
        request_size = len(url) + 300                    # simple estimate if user skips it
    try:
        request_size = int(request_size)
    except (TypeError, ValueError):
        raise ValueError("Request size must be a whole number (bytes), e.g. 450.")
    if request_size < 0:
        raise ValueError("Request size cannot be negative.")
    return method, url, str(user_agent or "").strip(), request_size


def shannon_entropy(text):
    """Randomness of the characters. 0 = all same char; higher = more mixed."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def decode_twice(text):
    """Attackers sometimes encode twice (%252e). Decode twice to reveal the real text."""
    return unquote(unquote(text.replace("+", " ")))


def extract_features(method, url, user_agent="", request_size=0):
    """Return a dict {feature_name: number} for ONE request."""
    path, _, query = url.partition("?")
    decoded = decode_twice(url)            # what the server would 'really' see
    decoded_query = decode_twice(query)
    params = parse_qsl(query, keep_blank_values=True)
    n_chars = max(len(decoded), 1)

    digits = sum(ch.isdigit() for ch in decoded)
    letters = sum(ch.isalpha() for ch in decoded)
    # "special" = anything that is not a letter, digit or normal URL punctuation
    specials = sum((not ch.isalnum()) and ch not in "/?=&._-" for ch in decoded)

    return {
        # Long URLs can hide long payloads.
        "url_length": len(url),
        "query_length": len(query),
        # Attacks often stuff data into parameters.
        "parameter_count": len(params),
        "parameter_length": sum(len(v) for _, v in params),
        # Injection payloads often contain numbers like 1=1 or SLEEP(5).
        "digit_count": digits,
        # Symbols such as ' " < > ; ( ) | are rare in normal requests.
        "special_character_count": specials,
        # %27 %3C ... are used to hide payloads from simple filters.
        "encoded_character_count": len(re.findall(r"%[0-9a-fA-F]{2}", url)),
        # Quotes ' and " are how SQL injection "breaks out" of a text field.
        "quote_count": decoded_query.count("'") + decoded_query.count('"'),
        "sql_keyword_count": len(SQL_RE.findall(decoded_query)),
        "script_keyword_count": len(SCRIPT_RE.findall(decoded_query)),
        "path_traversal_count": len(TRAVERSAL_RE.findall(decoded)),
        "command_keyword_count": len(CMD_OPERATOR_RE.findall(decoded_query))
                                 + len(CMD_WORD_RE.findall(decoded_query)),
        # Random-looking text (encoded junk, shellcode) has high entropy.
        "entropy": shannon_entropy(decoded),
        # Ratios let us compare short and long requests fairly.
        "digit_ratio": digits / n_chars,
        "special_character_ratio": specials / n_chars,
        "alphabetic_ratio": letters / n_chars,
        "method_is_post": 1 if method == "POST" else 0,
        "request_size": request_size,
        "user_agent_length": len(user_agent),
        "user_agent_is_tool": 1 if TOOL_UA_RE.search(user_agent or "") else 0,
    }


def features_from_dataframe(df):
    """Apply extract_features to every row of a DataFrame -> DataFrame of FEATURE_NAMES."""
    import pandas as pd
    rows = [extract_features(r.method, r.url, r.user_agent, r.request_size)
            for r in df.itertuples(index=False)]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


if __name__ == "__main__":
    tests = [("GET", "/products?id=10"), ("GET", "/login?username=admin' OR '1'='1"),
             ("GET", "/search?q=<script>alert(1)</script>"),
             ("GET", "/view?path=../../etc/passwd"), ("GET", "/ping?host=1.1.1.1; whoami")]
    for m, u in tests:
        f = extract_features(m, u, "Mozilla/5.0", 400)
        print(u)
        print("   ", {k: round(v, 2) for k, v in f.items() if k in (
            "special_character_count", "sql_keyword_count", "script_keyword_count",
            "path_traversal_count", "command_keyword_count", "entropy")})
