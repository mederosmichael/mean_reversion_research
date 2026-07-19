# Cross-Sectional Mean Reversion with Regime Conditioning

Research on whether cross-sectional mean reversion (XSMR) in a liquid crypto universe
survives realistic transaction costs, and whether conditioning entries on a detected
volatility regime improves the result.

## Research question

Assets that underperform their cross-section over a short lookback tend to revert. The
question is not whether that signal exists in-sample, but whether the edge is large
enough to clear costs once positions are sized and turnover is charged for, and whether
a latent regime variable identifies the periods where reversion actually pays.

## Approach

**Universe and data.** 5-minute OHLCV bars pulled through `ccxt`, 365 training days and
90 test days, with symbols dropped below 98% bar coverage so the cross-section is not
distorted by thinly traded names.

**Signal.** For each asset, take the rolling mean return over a 36-bar lookback,
subtract the cross-sectional mean at each timestamp to get a relative return, and
standardize by that series' rolling volatility. The resulting z-score measures how far
an asset has diverged from its peers rather than from its own history.

**Entry and exit.** Enter long below `-entry_z` and short above `+entry_z`, gated by a
momentum filter that skips names whose z-score is moving quickly (a large `|Δz|` means
the divergence is still widening, not reverting). Exit on a z-score sign flip or on
return inside the exit band.

**Sizing.** Kelly fraction `f = frac · μ/σ²`, clipped to a hard cap so a low-variance
estimate cannot produce an unbounded position.

**Costs.** 12 bps charged per unit of turnover. This is the constraint that most of the
parameter space fails against.

**Regime detection.** A Gaussian hidden Markov model implemented from scratch in
`src/models/regime_detection_hmm.py` (Baum-Welch for fitting, Viterbi for state
decoding, numpy only, log-space throughout for numerical stability). The decoded state
conditions whether the strategy trades at all.

## Layout

| Path | Role |
| --- | --- |
| `src/ingest/ingest.py` | Paged OHLCV fetch with retry/backoff and coverage filtering |
| `src/ingest/preliminary.py` | Universe construction |
| `src/models/regime_detection_hmm.py` | From-scratch Gaussian HMM (Baum-Welch, Viterbi) |
| `src/backtest/xsmr_comprehensive_backtest.py` | Signal, Kelly sizing, cost model, backtest |
| `data/derived/` | Cached close panel and universe |
| `models/` | Fitted HMM and feature scaler |

## Status

Ongoing. Ingestion, signal construction, the HMM, and the cost-aware backtest pipeline
are implemented. Remaining work is out-of-sample validation on the held-out 90-day
window and cross-validated hyperparameter selection. The current `entry_z` and lookback
were chosen in-sample, so present figures should be read as in-sample until that lands.

## Run

```bash
pip install -r requirements.txt
python -m src.ingest.ingest
python -m src.backtest.xsmr_comprehensive_backtest
```
