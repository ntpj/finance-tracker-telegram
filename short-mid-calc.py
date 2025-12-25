import math
import pandas as pd
import yfinance as yf
import requests

# ====== TELEGRAM CONFIG ======
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_MAX = int(os.getenv("TELEGRAM_MAX", "3900"))

# ====== TICKERS ======
tickers = json.loads(os.environ["TICKERS_JSON"])

# ====== INDICATOR PARAMS ======
SMA_LEN = 120
BB_LEN = 50
BB_K = 2.0
RSI_LEN = 21
ATR_LEN = 21
ADX_LEN = 21
MFI_LEN = 21

# ====== HELPERS ======
def is_bad(x):
    return x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))

def fmt_num(x, decimals=2):
    if is_bad(x):
        return "-"
    return f"{x:,.{decimals}f}"

def fmt_pct(x, decimals=2):
    if is_bad(x):
        return "-"
    return f"{x:+.{decimals}f}%"

def pct_away(last, ref):
    if is_bad(last) or is_bad(ref) or ref == 0:
        return None
    return (last / ref - 1.0) * 100.0

def band_pos(last, bb_l, bb_u):
    # 0 = at lower band, 1 = at upper band, <0 below lower, >1 above upper
    if is_bad(last) or is_bad(bb_l) or is_bad(bb_u) or (bb_u - bb_l) == 0:
        return None
    return (last - bb_l) / (bb_u - bb_l)

def _as_series(x):
    if isinstance(x, pd.DataFrame):
        return x.iloc[:, 0]
    return x

# ====== INDICATORS ======
def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    if close is None or len(close) < length + 1:
        return pd.Series(index=getattr(close, "index", None), dtype="float64")

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, float("nan"))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    if min(len(high), len(low), len(close)) < length + 1:
        return pd.Series(index=getattr(close, "index", None), dtype="float64")
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / length, adjust=False).mean()
    return atr

def adx_wilder(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    if min(len(high), len(low), len(close)) < length + 2:
        return pd.Series(index=getattr(close, "index", None), dtype="float64")

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr = atr_wilder(high, low, close, length)
    # Smooth DM with Wilder EMA
    plus_dm_s = plus_dm.ewm(alpha=1.0 / length, adjust=False).mean()
    minus_dm_s = minus_dm.ewm(alpha=1.0 / length, adjust=False).mean()

    plus_di = 100.0 * (plus_dm_s / atr.replace(0.0, float("nan")))
    minus_di = 100.0 * (minus_dm_s / atr.replace(0.0, float("nan")))

    dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, float("nan")))
    adx = dx.ewm(alpha=1.0 / length, adjust=False).mean()
    return adx

def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 14) -> pd.Series:
    if min(len(high), len(low), len(close), len(volume)) < length + 1:
        return pd.Series(index=getattr(close, "index", None), dtype="float64")

    tp = (high + low + close) / 3.0
    raw_mf = tp * volume

    tp_diff = tp.diff()
    pos_mf = raw_mf.where(tp_diff > 0, 0.0)
    neg_mf = raw_mf.where(tp_diff < 0, 0.0)

    pos_sum = pos_mf.rolling(length).sum()
    neg_sum = neg_mf.rolling(length).sum()

    mfr = pos_sum / neg_sum.replace(0.0, float("nan"))
    mfi_val = 100.0 - (100.0 / (1.0 + mfr))
    return mfi_val

# ====== OB/OS LOGIC (WITH ATR + ADX + MFI) ======
def classify_obos(last, bb_l, bb_u, pct_vs_sma, rsi14, atr_pct, adx14, mfi14):
    """
    Uses:
      - ATR%: noise filter + normalized stretch thresholds
      - ADX: trend regime (down-weight mean reversion when strong trend)
      - MFI: volume confirmation (confidence)
    Returns: (state, confidence_tag)
    """
    bp = band_pos(last, bb_l, bb_u)
    if any(is_bad(x) for x in [bp, pct_vs_sma, rsi14, atr_pct, adx14, mfi14]):
        return "N/A", "N/A"

    # --- 1) Noise filter (if move is small vs typical daily movement) ---
    if abs(pct_vs_sma) < 0.8 * atr_pct and 0.25 < bp < 0.75 and 40 < rsi14 < 60:
        return "Neutral", "Low"

    score = 0.0

    # --- 2) RSI contribution ---
    if rsi14 <= 30:
        score -= 2
    elif rsi14 <= 40:
        score -= 1
    elif rsi14 >= 70:
        score += 2
    elif rsi14 >= 60:
        score += 1

    # --- 3) BB position contribution ---
    if bp <= 0.10 or last < bb_l:
        score -= 2
    elif bp <= 0.30:
        score -= 1
    elif bp >= 0.90 or last > bb_u:
        score += 2
    elif bp >= 0.70:
        score += 1

    # --- 4) SMA stretch contribution (normalized by ATR%) ---
    # thresholds scale with volatility: 0.8x ATR% is "meaningful", 1.6x is "strong"
    if pct_vs_sma <= -1.6 * atr_pct:
        score -= 2
    elif pct_vs_sma <= -0.8 * atr_pct:
        score -= 1
    elif pct_vs_sma >= +1.6 * atr_pct:
        score += 2
    elif pct_vs_sma >= +0.8 * atr_pct:
        score += 1

    # --- 5) ADX regime (trend strength) ---
    # Strong trend -> reduce mean-reversion confidence (closer to neutral unless extreme)
    if adx14 >= 30:
        score *= 0.60
    elif adx14 >= 25:
        score *= 0.75
    elif adx14 <= 15:
        score *= 1.15  # choppy/range -> mean reversion slightly more reliable

    # --- 6) Map score to state ---
    if score <= -3.5:
        state = "Oversold"
    elif score <= -1.8:
        state = "Slight OS"
    elif score <= +1.2:
        state = "Neutral"
    elif score <= +3.0:
        state = "Slight OB"
    else:
        state = "Overbought"

    # --- 7) Confidence using MFI confirmation + ADX penalty ---
    # Oversold confirmation: MFI <= 30, Overbought confirmation: MFI >= 70
    conf = "Med"
    if state in ["Oversold", "Slight OS"]:
        if mfi14 <= 30:
            conf = "High"
        elif mfi14 >= 50:
            conf = "Low"
    elif state in ["Overbought", "Slight OB"]:
        if mfi14 >= 70:
            conf = "High"
        elif mfi14 <= 50:
            conf = "Low"
    else:
        conf = "Low"

    # Strong trend reduces reversal confidence
    if adx14 >= 30 and conf == "High":
        conf = "Med"
    if adx14 >= 30 and conf == "Med":
        conf = "Low"

    return state, conf

# ====== TELEGRAM TABLE ======
def telegram_table(df: pd.DataFrame, title: str) -> str:
    d = df.copy()
    d_fmt = pd.DataFrame({
        "Asset": d["Asset"].astype(str),
        "Last": d["Last(Daily Close)"].apply(lambda x: fmt_num(x, 2)),
        f"SMA{SMA_LEN}": d[f"SMA{SMA_LEN}"].apply(lambda x: fmt_num(x, 2)),
        "%SMA": d[f"Pct_vs_SMA{SMA_LEN}"].apply(lambda x: fmt_pct(x, 2)),
        "BB_Lower": d["BB_Lower(20,2)"].apply(lambda x: fmt_num(x, 2)),
        "%Lower": d["Pct_vs_BB_L"].apply(lambda x: fmt_pct(x, 2)),
        "BB_Upper": d["BB_Upper(20,2)"].apply(lambda x: fmt_num(x, 2)),
        "%Upper": d["Pct_vs_BB_U"].apply(lambda x: fmt_pct(x, 2)),
        "B_Pos": d["BandPos(0=L,1=U)"].apply(lambda x: "-" if is_bad(x) else f"{x:.2f}"),
        "RSI": d["RSI14"].apply(lambda x: "-" if is_bad(x) else f"{x:.1f}"),
        "ATR%": d["ATR14_pct"].apply(lambda x: "-" if is_bad(x) else f"{x:.2f}"),
        "ADX": d["ADX14"].apply(lambda x: "-" if is_bad(x) else f"{x:.1f}"),
        "MFInd": d["MFI14"].apply(lambda x: "-" if is_bad(x) else f"{x:.1f}"),
        "State": d["State"].astype(str),
        "Conf": d["Confidence"].astype(str),
        "Note": d["Note"].astype(str),
    })

    caps = {
        "Asset": 6, "Last": 11, f"SMA{SMA_LEN}": 11, "%SMA": 7,
        "BB_Lower": 11, "%Lower": 7, "BB_Upper": 11, "%Upper": 7,
        "B_Pos": 5, "RSI": 5, "ATR%": 5, "ADX": 5, "MFInd": 5,
        "State": 10, "Conf": 4, "Note": 16
    }
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
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
    }
    r = requests.post(url, json=payload, timeout=30)
    r.raise_for_status()

# ====== FETCH + CALC (DAILY) ======
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

    if px.empty:
        rows.append([name, yf_ticker, None, None, None, None, None, None, None, None, None, None, None, None, "N/A", "N/A", "No data"])
        continue

    # yfinance sometimes returns 1-col DataFrames per field; normalize
    close = _as_series(px.get("Close"))
    high  = _as_series(px.get("High"))
    low   = _as_series(px.get("Low"))
    vol   = _as_series(px.get("Volume"))

    if close is None or close.dropna().empty:
        rows.append([name, yf_ticker, None, None, None, None, None, None, None, None, None, None, None, None, "N/A", "N/A", "No close series"])
        continue

    close = close.dropna()
    high = high.reindex(close.index).dropna()
    low = low.reindex(close.index).dropna()
    vol = vol.reindex(close.index).fillna(0.0)

    # Align strictly
    idx = close.index.intersection(high.index).intersection(low.index).intersection(vol.index)
    close, high, low, vol = close.loc[idx], high.loc[idx], low.loc[idx], vol.loc[idx]

    last = float(close.iloc[-1])

    note_parts = []

    # SMA{SMA_LEN}
    sma = None
    if len(close) >= SMA_LEN:
        sma = float(close.rolling(SMA_LEN).mean().iloc[-1])
    else:
        note_parts.append(f"<{SMA_LEN}d")

    # Bollinger 20,2
    bb_lower = bb_upper = None
    if len(close) >= BB_LEN:
        bb_mid = close.rolling(BB_LEN).mean().iloc[-1]
        bb_std = close.rolling(BB_LEN).std(ddof=0).iloc[-1]
        if pd.notna(bb_mid) and pd.notna(bb_std):
            bb_upper = float(bb_mid + BB_K * bb_std)
            bb_lower = float(bb_mid - BB_K * bb_std)
    else:
        note_parts.append(f"<{BB_LEN}d")

    # RSI(14)
    rsi14 = None
    if len(close) >= RSI_LEN + 1:
        rsi_series = rsi_wilder(close, RSI_LEN)
        if not rsi_series.empty and pd.notna(rsi_series.iloc[-1]):
            rsi14 = float(rsi_series.iloc[-1])
    else:
        note_parts.append(f"<{RSI_LEN}d(RSI)")

    # ATR%(14)
    atr_pct = None
    if len(close) >= ATR_LEN + 1:
        atr_series = atr_wilder(high, low, close, ATR_LEN)
        if not atr_series.empty and pd.notna(atr_series.iloc[-1]) and last != 0:
            atr_pct = float(atr_series.iloc[-1] / last * 100.0)
    else:
        note_parts.append(f"<{ATR_LEN}d(ATR)")

    # ADX(14)
    adx14 = None
    if len(close) >= ADX_LEN + 2:
        adx_series = adx_wilder(high, low, close, ADX_LEN)
        if not adx_series.empty and pd.notna(adx_series.iloc[-1]):
            adx14 = float(adx_series.iloc[-1])
    else:
        note_parts.append(f"<{ADX_LEN}d(ADX)")

    # MFI(14)
    mfi14 = None
    if len(close) >= MFI_LEN + 1:
        mfi_series = mfi(high, low, close, vol, MFI_LEN)
        if not mfi_series.empty and pd.notna(mfi_series.iloc[-1]):
            mfi14 = float(mfi_series.iloc[-1])
    else:
        note_parts.append(f"<{MFI_LEN}d(MFI)")

    pct_vs_sma = pct_away(last, sma)
    pct_vs_bbl = pct_away(last, bb_lower)
    pct_vs_bbu = pct_away(last, bb_upper)
    bp = band_pos(last, bb_lower, bb_upper)

    state, conf = classify_obos(last, bb_lower, bb_upper, pct_vs_sma, rsi14, atr_pct, adx14, mfi14)

    # Helpful extra note if trend is strong
    trend_note = ""
    if not is_bad(adx14) and adx14 >= 25:
        trend_note = "TrendStrong"
    note = "OK" if not note_parts else "Insufficient " + ",".join(note_parts)
    if trend_note:
        note = (note + ";" + trend_note) if note != "OK" else trend_note

    rows.append([
        name, yf_ticker, last,
        sma, pct_vs_sma,
        bb_lower, pct_vs_bbl,
        bb_upper, pct_vs_bbu,
        bp, rsi14, atr_pct, adx14, mfi14,
        state, conf,
        note
    ])

out = pd.DataFrame(
    rows,
    columns=[
        "Asset", "YahooTicker", "Last(Daily Close)",
        f"SMA{SMA_LEN}", f"Pct_vs_SMA{SMA_LEN}",
        "BB_Lower(20,2)", "Pct_vs_BB_L",
        "BB_Upper(20,2)", "Pct_vs_BB_U",
        "BandPos(0=L,1=U)", "RSI14", "ATR14_pct", "ADX14", "MFI14",
        "State", "Confidence",
        "Note"
    ]
)

# ====== SORT: most Oversold first, then closest to lower band ======
state_rank = {
    "Oversold": 0,
    "Slight OS": 1,
    "Neutral": 2,
    "Slight OB": 3,
    "Overbought": 4,
    "N/A": 99,
}
out["_rank"] = out["State"].map(state_rank).fillna(99)

def _dist_to_lower(row):
    if pd.isna(row["BB_Lower(20,2)"]) or pd.isna(row["Last(Daily Close)"]):
        return float("inf")
    return (row["Last(Daily Close)"] / row["BB_Lower(20,2)"]) - 1.0

out["_dist"] = out.apply(_dist_to_lower, axis=1)
out = out.sort_values(["_rank", "_dist"], na_position="last").drop(columns=["_rank", "_dist"]).reset_index(drop=True)

# ====== BUILD + SEND ======
title = f"BB(20,{BB_K}σ)+SMA{SMA_LEN}+RSI{RSI_LEN}+ATR%{ATR_LEN}+ADX{ADX_LEN}+MFI{MFI_LEN} | OB/OS"
msg = telegram_table(out, title=title)
parts = split_telegram(msg)

for p in parts:
    tg_send(BOT_TOKEN, CHAT_ID, p)

print(f"Sent {len(parts)} message(s) to Telegram.")
