import time
import random
from pathlib import Path

import ccxt
import pandas as pd

TICK_SIZE = "5m"
TRAIN_DAYS = 365
TEST_DAYS = 90

MIN_COVERAGE_FRAC = 0.98
LIMIT = 300
MAX_RETRIES = 8
BASE_SLEEP = 0.75


def full_fetch_ohlcv(exchange, sym, tf, since_ms, until_ms=None, lim=LIMIT):
    out = []
    until_ms = int(time.time() * 1000) if until_ms is None else int(until_ms)

    while True:
        batch = None
        for k in range(MAX_RETRIES):
            try:
                batch = exchange.fetch_ohlcv(symbol=sym, timeframe=tf, since=since_ms, limit=lim)
                break
            except (ccxt.NetworkError, ccxt.ExchangeError, Exception) as e:
                msg = str(e)
                if ("Invalid" in msg) or ("does not exist" in msg) or ("Parameter" in msg):
                    raise
                time.sleep(BASE_SLEEP * (2**k) + random.random() * 0.25)

        if batch is None:
            raise RuntimeError(f"fetch failed for {sym} at since_ms={since_ms}")
        if not batch:
            break

        trimmed = [c for c in batch if c[0] < until_ms]
        if trimmed:
            out.extend(trimmed)

        if len(trimmed) < len(batch):
            break

        since_ms = batch[-1][0] + 1
        if len(batch) < lim:
            break

    return pd.DataFrame(out, columns=["ts", "open", "high", "low", "close", "volume"])


def read_universe_csv(path: str) -> list[str]:
    s = pd.read_csv(path, usecols=[0]).iloc[:, 0].astype(str).str.strip()
    seen = set()
    out = []
    for x in s:
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def load_or_fetch_close_series(exchange, sym: str, since_ms: int, end_ms: int, parts_dir: Path) -> pd.Series:
    fname = parts_dir / f"{sym.replace('/', '_')}.parquet"

    if fname.exists():
        asset = pd.read_parquet(fname)
    else:
        raw = full_fetch_ohlcv(exchange, sym, TICK_SIZE, since_ms, end_ms)[["ts", "close"]].copy()
        raw.drop_duplicates(subset=["ts"], keep="last", inplace=True)
        asset = raw.rename(columns={"close": sym})
        asset.to_parquet(fname, index=False, engine="pyarrow")

    if asset is None or asset.empty or sym not in asset.columns:
        return pd.Series(dtype="float64", name=sym)

    # force numeric ms
    ts_ms = pd.to_numeric(asset["ts"], errors="coerce").astype("Int64")
    mask = ts_ms.notna()
    if mask.sum() == 0:
        return pd.Series(dtype="float64", name=sym)

    idx = pd.to_datetime(ts_ms[mask].astype("int64"), unit="ms", utc=True)
    vals = pd.to_numeric(asset.loc[mask, sym], errors="coerce").to_numpy()

    s = pd.Series(vals, index=idx, name=sym).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s.dropna()

    return s


def build_close_panel():
    exchange = ccxt.okx({"enableRateLimit": True})

    universe_csv = "data/derived/universe.csv"
    parts_dir = Path("data/raw/close_5m_parts")
    out_parquet = "data/derived/close_5m.parquet"

    parts_dir.mkdir(parents=True, exist_ok=True)
    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)

    symbols = read_universe_csv(universe_csv)

    # optional: filter to symbols OKX actually knows
    markets = exchange.load_markets()
    symbols = [s for s in symbols if s in markets]

    end_ms = int(time.time() * 1000)
    since_ms = end_ms - (TRAIN_DAYS + TEST_DAYS) * 24 * 3600 * 1000

    series_list = []
    for i, sym in enumerate(symbols, 1):
        try:
            s = load_or_fetch_close_series(exchange, sym, since_ms, end_ms, parts_dir)
            if s.empty:
                print(f"{i}/{len(symbols)} empty: {sym}")
                continue
            series_list.append(s)
            print(f"{i}/{len(symbols)} ok: {sym}  n={len(s):,}")
        except Exception as e:
            print(f"{i}/{len(symbols)} fail: {sym} err={e}")

    if not series_list:
        raise RuntimeError("no non-empty series collected")

    U = pd.concat(series_list, axis=1)

    # hard assertions + debug
    print("U.index type:", type(U.index))
    print("U.index dtype:", getattr(U.index, "dtype", None))
    print("U time min/max:", U.index.min(), U.index.max())

    if not isinstance(U.index, pd.DatetimeIndex):
        # last-resort coercion
        U.index = pd.to_datetime(U.index, utc=True, errors="coerce")
        U = U[~U.index.isna()].sort_index()

    if not isinstance(U.index, pd.DatetimeIndex):
        raise RuntimeError(f"still not DatetimeIndex after coercion: {type(U.index)}")

    close = U.resample("5min").last()

    min_non_na = int(MIN_COVERAGE_FRAC * len(close))
    close = close.dropna(axis=1, thresh=min_non_na)
    close = close.ffill()

    print("panel shape:", close.shape)
    print("panel time min/max:", close.index.min(), close.index.max())

    close.to_parquet(out_parquet, engine="pyarrow")
    print("wrote:", out_parquet)


if __name__ == "__main__":
    build_close_panel()
