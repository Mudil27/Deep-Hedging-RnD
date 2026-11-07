"""
Step 1 - Market data.

Downloads S&P500 daily closes, turns them into log-returns, and cuts out the
(train, test) windows used in Section 5.5 of Buehler et al. (2019).

Paper's window: returns from 5 Mar 2013 to 2 Jan 2018 -> n_ret = 1217 points.
The first n_ret - T_days returns are "the past" (used to fit GARCH and train),
the last T_days returns are "the future" (only used for testing).
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_FILE = DATA_DIR / "sp500_gspc_daily.csv"

# Wide enough to cover all 300 windows of Experiment B (earliest starts mid-2007)
# plus some slack on both ends.
DEFAULT_START = "2006-01-01"
DEFAULT_END = "2018-02-01"


def download_sp500(start=DEFAULT_START, end=DEFAULT_END, path=RAW_FILE):
    """Download ^GSPC daily closes from Yahoo Finance and save them as CSV."""
    import yfinance as yf

    df = yf.download("^GSPC", start=start, end=end, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):  # newer yfinance returns (field, ticker) columns
        df.columns = df.columns.get_level_values(0)
    df = df[["Close", "Adj Close"]].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def load_prices(path=RAW_FILE):
    """Closing prices as a pd.Series indexed by date (downloads on first use)."""
    if not path.exists():
        download_sp500(path=path)
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df["Adj Close"].rename("S")


def log_returns(prices):
    """r_k = log(S_k / S_{k-1}); the return is stamped with the date of S_k."""
    return np.log(prices / prices.shift(1)).dropna().rename("r")


def get_window(end_date, n_ret=1217, T_days=240, prices=None):
    """
    Cut one experiment window ending at `end_date` (inclusive).

    Returns a dict with
      train_returns : first n_ret - T_days returns (fit GARCH / train network)
      test_returns  : last T_days returns (historical test path)
      S0            : price on the last training day ("today", k = 0)
      test_prices   : S_0, S_1, ..., S_T along the real future path
    """
    if prices is None:
        prices = load_prices()
    rets = log_returns(prices)
    rets = rets.loc[:pd.Timestamp(end_date)].iloc[-n_ret:]
    if len(rets) < n_ret:
        raise ValueError(f"only {len(rets)} returns available before {end_date}")

    train, test = rets.iloc[: n_ret - T_days], rets.iloc[n_ret - T_days:]
    today = train.index[-1]
    test_prices = prices.loc[today:test.index[-1]]
    return {
        "train_returns": train,
        "test_returns": test,
        "today": today,
        "S0": float(prices.loc[today]),
        "test_prices": test_prices,
    }
