import ccxt
import pandas as pd
import time
import numpy as np
import requests
import os
from src.ingest.ingest import full_fetch_ohlcv


TICK_SIZE = "5m"
CAPS_PATH = "data/caps_by_symbol.csv"

def cov_lag(r, k):
    return np.cov(r[:-k], r[k:], ddof=0)[0,1]

def corr_lag(r, k):
    return np.corrcoef(r[:-k], r[k:])[0,1]

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
            ohlcv = full_fetch_ohlcv(exchange,sym, timeframe, since_ms)
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

def sweep_universe_for_mr(
    exchange,
    u: pd.DataFrame,
    *,
    tick_size: str = "5m",
    days: int = 30,
    W: int = 288,                 # z-score window in bars
    H: int = 24,                  # horizon in bars (24*5m = 2h)
    event_z: float = 2.0,
    revert_z: float = 1.0,
    min_events: int = 50,
    delta_p_min: float = 0.05,    # must beat baseline by this
    rho_min_threshold: float = -0.01,  # must have at least one rho below this
    max_symbols: int = 60,
    verbose: bool = False,        # like "-v": print stats for flagged symbols
) -> pd.DataFrame:
    """
    Sweep symbols in `u['symbol']` and screen for mean-reversion candidates.

    Uses your existing logic:
      - price z-score: z_t = (close - rolling_mean_W) / rolling_std_W
      - event starts when |z_t| >= event_z and previous bar was not an event
      - 'reverted' if |z_{t+H}| <= revert_z
      - p = P(reverted | event_start), base = P(|z_{t+H}|<=revert_z), delta_p = p - base
      - q = P(|z_{t+H}| < |z_t| | event_start)
      - return autocorr rhos at lags 1,3,6,12,24 (in bars)

    Boolean flag is_mr:
      n_events >= min_events AND delta_p >= delta_p_min AND rho_min <= rho_min_threshold

    Returns:
      DataFrame sorted by (is_mr desc, delta_p desc, rho_min asc).
    """

    since_ms = int((time.time() - days * 24 * 3600) * 1000)

    rows = []
    syms = list(u["symbol"])[:max_symbols]

    for sym in syms:
        try:
            df = full_fetch_ohlcv(exchange,sym=sym, tf=tick_size, since_ms=since_ms, lim=300)
        except Exception:
            continue

        if len(df) < W + H + 50:
            continue

        close = df["close"].astype(float)

        # z-score on price
        mu = close.rolling(W).mean()
        sd = close.rolling(W).std(ddof=0)
        z = ((close - mu) / sd).dropna()

        if len(z) <= H + 10:
            continue

        event = z.abs() >= event_z
        event_start = event & (~event.shift(1, fill_value=False))

        reverted = z.shift(-H).abs() <= revert_z
        mask = event_start & reverted.notna()

        n_events = int(mask.sum())
        if n_events == 0:
            continue

        p = float(reverted[mask].mean())
        base = float((z.shift(-H).abs() <= revert_z).mean())
        delta_p = p - base

        z0 = z[event_start]
        zH = z.shift(-H)[event_start]
        q = float((zH.abs() < z0.abs()).mean())

        # return autocorrs
        r = np.diff(np.log(close.values))
        if len(r) < 200:
            continue

        rhos = {
            1: corr_lag(r, 1),
            3: corr_lag(r, 3),
            6: corr_lag(r, 6),
            12: corr_lag(r, 12),
            24: corr_lag(r, 24),
        }
        rho_min = float(np.nanmin(list(rhos.values())))

        is_mr = (n_events >= min_events) and (delta_p >= delta_p_min) and (rho_min <= rho_min_threshold)

        row = {
            "symbol": sym,
            "n_events": n_events,
            "p_revert": p,
            "base": base,
            "delta_p": delta_p,
            "q_partial": q,
            "rho_5m": rhos[1],
            "rho_15m": rhos[3],
            "rho_30m": rhos[6],
            "rho_60m": rhos[12],
            "rho_120m": rhos[24],
            "rho_min": rho_min,
            "is_mr": is_mr,
        }
        rows.append(row)

        if verbose and is_mr:
            print(f"\n{sym}")
            print("  n_events:", n_events)
            print("  p_revert:", p, "base:", base, "delta_p:", delta_p)
            print("  q_partial:", q)
            print("  rhos:", rhos)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["is_mr", "delta_p", "rho_min"], ascending=[False, False, True]).reset_index(drop=True)
    return out


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

def fetch_closes_matrix(exchange, symbols, tick_size="5m", days=30, limit=300):
    since_ms = int((time.time() - days * 24 * 3600) * 1000)
    series = {}
    for sym in symbols:
        try:
            df = full_fetch_ohlcv(exchange,sym=sym, tf=tick_size, since_ms=since_ms, lim=limit)
        except Exception:
            continue
        if df.empty:
            continue
        s = df[["ts", "close"]].copy()
        s["ts"] = pd.to_datetime(s["ts"], unit="ms", utc=True)
        s = s.set_index("ts")["close"].astype(float)
        series[sym] = s

    if not series:
        return pd.DataFrame()

    closes = pd.concat(series, axis=1).sort_index()
    return closes

def fetch_closes_matrix(exchange, symbols, tick_size="5m", days=30, lim=300):
    since_ms = int((time.time() - days * 24 * 3600) * 1000)
    series = {}
    for sym in symbols:
        try:
            df = full_fetch_ohlcv(exchange,sym=sym, tf=tick_size, since_ms=since_ms, lim=lim)
        except Exception:
            continue
        if df.empty:
            continue
        s = df[["ts", "close"]].copy()
        s["ts"] = pd.to_datetime(s["ts"], unit="ms", utc=True)
        s = s.set_index("ts")["close"].astype(float)
        series[sym] = s

    if not series:
        return pd.DataFrame()

    closes = pd.concat(series, axis=1).sort_index()
    return closes


def run_simple_xsmr_backtest(
    exchange,
    universe_df: pd.DataFrame,
    caps_by_base_symbol: dict,
    *,
    tick_size="5m",
    days=30,
    max_symbols=120,
    K=288,      # signal lookback in bars
    H=24,       # holding horizon in bars
    q=0.2,      # long bottom q, short top q
    weight_mode="equal",  # "equal" or "mcap"
    min_assets=40,
):
    # pick symbols
    syms = list(universe_df["symbol"])[:max_symbols]

    # fetch and align closes
    closes = fetch_closes_matrix(exchange, syms, tick_size=tick_size, days=days)
    if closes.empty:
        raise RuntimeError("no price data fetched")

    # drop assets with lots of missing and then align on common timestamps
    closes = closes.dropna(axis=1, thresh=int(0.9 * len(closes)))
    closes = closes.dropna(axis=0, how="any")

    if closes.shape[1] < min_assets:
        raise RuntimeError(f"too few assets after alignment: {closes.shape[1]}")

    # returns
    rets = np.log(closes).diff().dropna()

    # static weights (keep it simple first)
    if weight_mode == "equal":
        w = pd.Series(1.0, index=rets.columns)
        w = w / w.sum()
    elif weight_mode == "mcap":
        bases = pd.Index(rets.columns).str.split("/").str[0].str.upper()
        mcap = bases.map(caps_by_base_symbol).astype(float)
        w = pd.Series(mcap.values, index=rets.columns).dropna()
        w = w / w.sum()
        rets = rets[w.index]
    else:
        raise ValueError("weight_mode must be 'equal' or 'mcap'")

    # universe return and excess returns
    rU = rets.mul(w, axis=1).sum(axis=1)
    X = rets.sub(rU, axis=0)

    # signal: trailing sum of excess returns, then cross-sectional z-score each bar
    S = X.rolling(K).sum()
    Z = S.sub(S.mean(axis=1), axis=0).div(S.std(axis=1, ddof=0), axis=0)

    # forward excess return over horizon H
    X_fwd = X.shift(-H)

    # cut edges
    Z = Z.iloc[K:-H]
    X_fwd = X_fwd.iloc[K:-H]

    # long-short series
    def long_short_at_time(z_row, x_row):
        n = len(z_row)
        k = max(1, int(q * n))
        order = np.argsort(z_row.values)
        long_names = z_row.index[order[:k]]
        short_names = z_row.index[order[-k:]]
        return float(x_row[long_names].mean() - x_row[short_names].mean())

    ls = pd.Series(
        (long_short_at_time(Z.loc[t], X_fwd.loc[t]) for t in Z.index),
        index=Z.index,
        name="LS_excess"
    )

    # slope diagnostic: b(t) = cov(z, x_fwd)/var(z)
    zc = Z.sub(Z.mean(axis=1), axis=0)
    xc = X_fwd.sub(X_fwd.mean(axis=1), axis=0)
    b = (zc * xc).mean(axis=1) / (zc * zc).mean(axis=1)

    # summary
    out = {
        "n_assets": int(rets.shape[1]),
        "ls_mean": float(ls.mean()),
        "ls_std": float(ls.std(ddof=1)),
        "b_mean": float(np.nanmean(b)),
        "b_std": float(np.nanstd(b, ddof=1)),
        "ls_series": ls,
        "b_series": b,
    }
    return out

if __name__ == "__main__":
    exchange = ccxt.okx({"enableRateLimit": True})
    u_liq = define_universe_by_liquidity(exchange, number_of_tickers=200, timeframe="5m", lookback_bars=288)
    caps = load_or_fetch_caps()
    u = restrict_universe_by_max_market_cap(u_liq, max_market_cap_usd=10e9, caps_by_base_symbol=caps)

    u.to_csv("data/universe.csv", index=False)
    print('successfully saved universe')

    bt = run_simple_xsmr_backtest(
        exchange,
        u,
        caps,
        tick_size="5m",
        days=30,
        max_symbols=120,
        K=288,        # 1 day lookback
        H=24,         # 2 hour horizon
        q=0.2,
        weight_mode="equal",  # start equal to avoid cap noise
    )

    print("assets:", bt["n_assets"])
    print("LS mean:", bt["ls_mean"], "LS std:", bt["ls_std"])
    print("b mean:", bt["b_mean"], "b std:", bt["b_std"])
    print(bt["ls_series"].head())
    print(bt["b_series"].head())

