from src.ingest.ingest import full_fetch_ohlcv
from src.models.regime_detection_hmm import GaussianHMM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
Cross-sectional mean-reversion backtest on the test universe.

Signal is a cross-sectionally demeaned, volatility-standardized return z-score.
Entries are conditioned on the HMM regime state and gated by a momentum filter,
sized by a capped Kelly fraction, and charged transaction costs on turnover.
"""

TICK_SIZE = "5m"
DURATION = 90  # days
PRINCIPAL = 100.0
LOOKBACK = 36  # ticks
TC_BPS = 12
TC = TC_BPS / 1e4
OPTIMAL_Z = 3.75


def compute_xsmr(
    z: pd.DataFrame,
    entry_z: float,
    exit_z: float,
    mom_max: float | None = None,
) -> pd.DataFrame:
    # momentum proxy: |Δz| small => "momentum low"
    delta = z.sub(z.shift(1)).fillna(0.0)

    if mom_max is None:
        mom_ok = pd.DataFrame(True, index=z.index, columns=z.columns)
    else:
        mom_ok = delta.abs() <= mom_max

    # gate entries by low momentum
    enter_long = (z <= -entry_z) & mom_ok
    enter_short = (z >= entry_z) & mom_ok

    # exit on sign flip (crossing 0) or returning inside exit band
    exit_cross = (np.sign(z) * np.sign(z.shift(1))) < 0
    exit_band = z.abs() <= exit_z
    exit_any = exit_cross | exit_band

    raw = pd.DataFrame(np.nan, index=z.index, columns=z.columns)
    raw[enter_long] = 1.0
    raw[enter_short] = -1.0
    raw[exit_any] = 0.0

    return raw.ffill().fillna(0.0)



def apply_kelly(
    pos: pd.DataFrame, mu: pd.DataFrame, var: pd.DataFrame, frac: float, cap: float
) -> pd.DataFrame:
    f = frac * mu.div(var)
    f = f.clip(-cap, cap)
    return pos * f


def make_z(close_slice: pd.DataFrame, lookback: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    returns:
      z_score: cross-sectional mean-reversion z-score per asset
      r: returns used to compute z
    """
    r = close_slice.pct_change().fillna(0.0)

    r_recent = r.rolling(lookback).mean()
    r_cs_mean = r_recent.mean(axis=1)
    r_rel = r_recent.sub(r_cs_mean, axis=0)

    vol = r_rel.rolling(lookback).std(ddof=0).replace(0.0, np.nan)
    z_score = r_rel.div(vol)

    return z_score, r


def backtest_single_asset(
    r_asset: pd.DataFrame, signal: pd.DataFrame, principal: float, tc: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    r_asset and signal must share the same index.
    signal is in {-1,0,1}.
    returns:
      equity_curve (DataFrame)
      portfolio_returns (DataFrame)
    """
    # use prior position to avoid lookahead
    pos_prev = signal.shift(1).fillna(0.0)

    turnover = (signal - pos_prev).abs()
    costs = tc * turnover

    portfolio = (r_asset * pos_prev) - costs
    portfolio = portfolio.fillna(0.0)

    equity_curve = principal * (1.0 + portfolio).cumprod()
    return equity_curve, portfolio


if __name__ == "__main__":
    # load prices
    close = pd.read_parquet("data/derived/close_5m.parquet")

    # split train/test by last DURATION days at 5m frequency (12 bars/hour * 24 hours/day)
    n_test = 12 * 24 * DURATION
    train = close.iloc[:-n_test].copy()
    test = close.iloc[-n_test:].copy()

    # build z-scores separately on train and test (critical: index alignment)
    z_train, r_train = make_z(train, LOOKBACK)
    z_test, r_test = make_z(test, LOOKBACK)

    asset = "PEPE/USDT"
    micro_u_train = z_train[[asset]].copy()

    # --- grid search on train ---
    opt_in_z = 0.0
    opt_out_z = 0.0
    best_value = -np.inf

    r_asset_train = r_train[[asset]].copy()

    for entry_z in np.arange(2.0, 10.01, 0.1):
        for exit_z in np.arange(0.0, 1.0, 0.02):
            if exit_z > entry_z:
                continue

            signal_train = compute_xsmr(micro_u_train, entry_z, exit_z,)
            equity_train, _ = backtest_single_asset(
                r_asset=r_asset_train, signal=signal_train, principal=PRINCIPAL, tc=TC
            )

            final_val = float(equity_train.iloc[-1, 0])
            if final_val > best_value:
                best_value = final_val
                opt_in_z = float(entry_z)
                opt_out_z = float(exit_z)

    print("opt_in_z:", opt_in_z)
    print("opt_out_z:", opt_out_z)
    print("best_train_final:", best_value)

    # --- test with optimal thresholds ---
    micro_u_test = z_test[[asset]].copy()
    signal_test = compute_xsmr(micro_u_test, opt_in_z, opt_out_z)

    r_asset_test = r_test[[asset]].copy()
    equity_curve_test, portfolio_test = backtest_single_asset(
        r_asset=r_asset_test, signal=signal_test, principal=PRINCIPAL, tc=TC
    )

    # plot explicitly (stable behavior)
    plt.figure()
    plt.plot(equity_curve_test.index, equity_curve_test[asset])
    plt.title(f"{asset} XSMR equity (test)")
    plt.tight_layout()
    plt.show()

    # if you run headless and show() does nothing, uncomment:
    # plt.savefig("equity_curve_test.png", dpi=200)
