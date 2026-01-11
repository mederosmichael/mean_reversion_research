import ccxt
import pandas as pd
import time
import numpy as np
import requests
import os

exchange = ccxt.okx({"enableRateLimit": True})

TICK_SIZE = "5m"
CAPS_PATH = "data/caps_by_symbol.csv"

def full_fetch_ohlcv(sym: str, tf: str, since_ms: int, lim: int = 300):
    out = []
    while True:
        batch = exchange.fetch_ohlcv(symbol=sym, timeframe=tf, since=since_ms, limit=lim)
        if not batch:
            break
        out.extend(batch)
        since_ms = batch[-1][0] + 1
        if len(batch) < lim:
            break
    return pd.DataFrame(out, columns=["ts", "open", "high", "low", "close", "volume"])

def load_or_fetch_caps():
    if os.path.exists(CAPS_PATH):
        s = pd.read_csv(CAPS_PATH)
        return dict(zip(s["base"], s["market_cap_usd"]))

    caps = fetch_market_caps_by_base_symbol_usd()
    pd.DataFrame({"base": list(caps.keys()), "market_cap_usd": list(caps.values())}).to_csv(CAPS_PATH, index=False)
    return caps

def define_universe_by_liquidity(
    exchange,
    quote: str = "USDT",
    number_of_tickers: int = 50,
    timeframe: str = "5m",
    lookback_bars: int = 288,  # 1 day at 5m bars
):
    exchange.load_markets()

    symbols = [
        s for s in exchange.symbols
        if s.endswith(f"/{quote}") and exchange.markets[s].get("spot")
    ]

    tf_minutes = int(timeframe[:-1])  # assumes 'Xm'
    since_ms = int((time.time() - lookback_bars * tf_minutes * 60) * 1000)

    rows = []
    for sym in symbols:
        try:
            ohlcv = full_fetch_ohlcv(sym, timeframe, since_ms)
        except Exception:
            continue
        if len(ohlcv) < lookback_bars:
            continue

        dv = ohlcv["close"].astype(float) * ohlcv["volume"].astype(float)
        rows.append((sym, float(np.median(dv.tail(lookback_bars)))))

    out = pd.DataFrame(rows, columns=["symbol", "liquidity"])
    return out.sort_values("liquidity", ascending=False).head(number_of_tickers).reset_index(drop=True)


def fetch_market_caps_by_base_symbol_usd(pages: int = 4, per_page: int = 250) -> dict:
    url = "https://api.coingecko.com/api/v3/coins/markets"
    caps = {}

    for page in range(1, pages + 1):
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": per_page,
            "page": page,
            "sparkline": False,
        }

        r = requests.get(url, params=params, timeout=20)
        data = r.json()

        if not isinstance(data, list):
            print("coingecko returned non-list:", data)
            break

        for row in data:
            base = row["symbol"].upper()
            cap = row["market_cap"]
            if base not in caps and cap is not None:
                caps[base] = float(cap)

    return caps



def restrict_universe_by_max_market_cap(
    universe_df: pd.DataFrame,
    max_market_cap_usd: float,
    caps_by_base_symbol: dict,
) -> pd.DataFrame:
    df = universe_df.copy()
    df["base"] = df["symbol"].str.split("/").str[0].str.upper()
    df["market_cap_usd"] = df["base"].map(caps_by_base_symbol)

    df = df.dropna(subset=["market_cap_usd"])
    df = df[df["market_cap_usd"] <= max_market_cap_usd]

    return df.drop(columns=["base"]).reset_index(drop=True)

"""
u_liq = define_universe_by_liquidity(exchange, number_of_tickers=200, timeframe="5m", lookback_bars=288)
caps = load_or_fetch_caps()
u = restrict_universe_by_max_market_cap(u_liq, max_market_cap_usd=10e9, caps_by_base_symbol=caps)

print(u.head(25))
print("final universe size:", len(u))
"""

start = int((time.time() - 720 * 3600) * 1000)
df = full_fetch_ohlcv(sym="BTC/USDT", tf=TICK_SIZE, since_ms=start, lim=100)

W = 288
mu = df['close'].rolling(W).mean()
sd = df['close'].rolling(W).std(ddof=0)
df['z'] = (df['close'] - mu)/sd
df = df.dropna(subset=['z'])
print(df["z"].describe())
