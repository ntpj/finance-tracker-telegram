import math
import pandas as pd
import yfinance as yf
import requests

import os, json

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_MAX = int(os.getenv("TELEGRAM_MAX", "3900"))

# ====== TICKERS ======
tickers = json.loads(os.environ["TICKERS_JSON"])

# ====== YOUR BAND RULES (% extension above MA) ======
BANDS = [
    (-10_000, 0,      "V.cheap"),
    (0,       50,     "Cheap"),
    (50,      100,    "Fair"),
    (100,     150,    "Expensive"),
    (150,     10_000, "V.expensive"),
]

# ====== HELPERS ======
def label_band(ext_pct: float) -> str:
    for lo, hi, name in BANDS:
        if lo <= ext_pct < hi:
            return name
    return "Unknown"

def pick_ma_length(num_weeks: int) -> int | None:
    for L in (200, 100, 50, 36, 12):
        if num_weeks >= L:
            return L
    return None

def fmt_num(x, decimals=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    return f"{x:,.{decimals}f}"

def fmt_int(x):
    return "-" if x is None else str(int(x))

def telegram_table(df: pd.DataFrame, title: str) -> str:
    """
    Telegram-friendly monospace table. Paste-able + aligned on mobile/PC.
    """
    d = df.copy()
    d["Last"] = d["Last(Weekly Close)"]
    d["MAw"]  = d["MA_Weeks_Used"]
    d["MA"]   = d["MA_Value"]
    d["Ext%"] = d["Extension_%"]
    d["Band"] = d["Band"]
    d = d[["Asset", "Last", "MAw", "MA", "Ext%", "Band"]]

    d_fmt = pd.DataFrame({
        "Asset": d["Asset"].astype(str),
        "Last": d["Last"].apply(lambda x: fmt_num(x, 2)),
        "MAw":  d["MAw"].apply(fmt_int),
        "MA":   d["MA"].apply(lambda x: fmt_num(x, 2)),
        "Ext%": d["Ext%"].apply(lambda x: "-" if x is None or (isinstance(x, float) and math.isnan(x))
                                else f"{x:,.1f}%"),
        "Band": d["Band"].astype(str),
    })

    # Column caps for mobile friendliness
    caps = {"Asset": 6, "Last": 12, "MAw": 3, "MA": 12, "Ext%": 8, "Band": 13}
    widths = {}
    for col in d_fmt.columns:
        w = max(len(col), *(len(str(v)) for v in d_fmt[col].values))
        widths[col] = min(max(w, len(col)), caps.get(col, w))

    def trunc(s, w):
        s = str(s)
        return s if len(s) <= w else s[: max(0, w - 1)] + "…"

    for col in d_fmt.columns:
        d_fmt[col] = d_fmt[col].apply(lambda s: trunc(s, widths[col]))

    header = " | ".join(col.ljust(widths[col]) for col in d_fmt.columns)
    sep    = "-+-".join("-" * widths[col] for col in d_fmt.columns)

    lines = [title, header, sep]
    for _, r in d_fmt.iterrows():
        lines.append(" | ".join(str(r[col]).ljust(widths[col]) for col in d_fmt.columns))

    return "```\n" + "\n".join(lines) + "\n```"

def split_telegram(text: str, max_len: int = TELEGRAM_MAX) -> list[str]:
    """
    Split while preserving code fences per chunk (Telegram sometimes breaks monospace otherwise).
    """
    # remove outer fences so we can re-wrap per chunk cleanly
    if text.startswith("```") and text.endswith("```"):
        core = text.strip("`").strip()
        # The above is a bit aggressive; better:
    if text.startswith("```") and text.endswith("```"):
        core = text[3:-3].strip("\n")
    else:
        core = text

    lines = core.splitlines()
    parts, cur = [], []

    def wrap(block_lines):
        return "```\n" + "\n".join(block_lines) + "\n```"

    for line in lines:
        candidate = wrap(cur + [line])
        if len(candidate) > max_len and cur:
            parts.append(wrap(cur))
            cur = [line]
        else:
            cur.append(line)

    if cur:
        parts.append(wrap(cur))

    return parts

def tg_send(bot_token: str, chat_id: str, text: str) -> None:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",  # works with ``` monospace blocks
        "disable_web_page_preview": True,
    }
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()

# ====== FETCH + CALC ======
rows = []
for name, yf_ticker in tickers.items():
    px = yf.download(
        yf_ticker,
        period="max",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    if px.empty or "Close" not in px:
        rows.append([name, yf_ticker, None, None, None, None, "No data"])
        continue

    # Weekly closes
    close = px["Close"]
    if isinstance(close, pd.DataFrame):      # sometimes yfinance returns 1-col DataFrame
        close = close.iloc[:, 0]            # or: close = close.squeeze("columns")

    w = close.resample("W-FRI").last().dropna()
    n = len(w)
    last = float(w.iloc[-1])

    L = pick_ma_length(n)
    if L is None:
        rows.append([name, yf_ticker, last, None, None, None, "Insufficient (<50w)"])
        continue

    ma = float(w.rolling(L).mean().iloc[-1])
    ext = (last / ma - 1.0) * 100.0
    band = label_band(ext)

    rows.append([name, yf_ticker, last, L, ma, ext, band])

out = pd.DataFrame(
    rows,
    columns=["Asset", "YahooTicker", "Last(Weekly Close)", "MA_Weeks_Used", "MA_Value", "Extension_%", "Band"]
)

# Sort cheapest first
out = out.sort_values(["Extension_%", "MA_Weeks_Used"], na_position="last").reset_index(drop=True)

# ====== BUILD + SEND ======
msg = telegram_table(out, title="MA Extension (pref 200W→100W→50W→36W→12W)")
parts = split_telegram(msg)

for p in parts:
    tg_send(BOT_TOKEN, CHAT_ID, p)

print(f"Sent {len(parts)} message(s) to Telegram.")
