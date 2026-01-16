from src.ingest.ingest import full_fetch_ohlcv
import pandas as pd
import numpy as np
import time 
"""
Goal for tomorrow:
Design and formalize a trading strategy on the defined test universe that exploits XSMR.
Implement regime detection and a trade-decision rule conditional on regime state.
Use Kelly fraction for position sizing.
Exclude momentum components at this stage. Add later.
Incorporate transaction costs.
Build a fully specified backtest pipeline.
Explore limited cross-validation for hyperparameter selection.
"""

TICK_SIZE = '5m'
DURATION = 90 #days
PRINCIPAL = 100
LOOKBACK = 12 #ticks

def compute_xsmr(z: pd.DataFrame, entry_z: float, exit_z: float) -> pd.DataFrame:
    enter_long=z<=-entry_z
    enter_short=z>=entry_z
    exit_any=z.abs()<=exit_z
    raw=pd.DataFrame(np.nan,index=z.index,columns=z.columns)
    raw[enter_long]=1.0
    raw[enter_short]=-1.0
    raw[exit_any]=0.0
    return raw.ffill().fillna(0.0)

def apply_kelly(pos: pd.DataFrame, mu: pd.DataFrame, var: pd.DataFrame, frac: float, cap: float) -> pd.DataFrame:
    f=frac*mu.div(var)
    f=f.clip(-cap,cap)
    return pos*f
    
if __name__ == "__main__":
    close = pd.read_parquet("data/derived/close_5m.parquet")
    r = close.pct_change()
    r = r.fillna(0.0)
    r_recent = r.rolling(LOOKBACK).mean()
    r_cs_mean = r_recent.mean(axis=1)
    r_rel = r_recent.sub(r_cs_mean,axis=0)
    vol = r_rel.rolling(LOOKBACK).std(ddof=0)
    vol = vol.replace(0.0,np.nan)
    z_score = r_rel.div(vol)

    
