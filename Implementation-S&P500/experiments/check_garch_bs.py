"""
Sanity check for Steps 1-3 on the Experiment A window (T = 240 days, ends 2 Jan 2018):
  * fit GJR-GARCH-t on the training returns
  * Black-Scholes premium q at sigma_0  (paper: q ~ 70.5)
  * simulate paths and compare their statistics to the historical data
"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import get_window
from garch import fit_gjr_garch_t, simulate_paths
from black_scholes import call_price, TRADING_DAYS

T_DAYS = 240
w = get_window("2018-01-02", n_ret=1217, T_days=T_DAYS)
print(f"training returns : {w['train_returns'].index[0].date()} .. {w['today'].date()} "
      f"({len(w['train_returns'])} days)")
print(f"test returns     : {w['test_returns'].index[0].date()} .. {w['test_returns'].index[-1].date()} "
      f"({len(w['test_returns'])} days)")
print(f"S0 (today)       : {w['S0']:.2f}\n")

params, sigma0, eps0, sigma_hist = fit_gjr_garch_t(w["train_returns"])
print("GJR-GARCH-t fit:")
for k, v in params.as_dict().items():
    print(f"  {k:6s} = {v:.6g}")
print(f"  persistence alpha+gamma/2+beta = {params.persistence:.4f}")
print(f"  unconditional vol = {params.uncond_daily_vol*math.sqrt(TRADING_DAYS):.2%} p.a.")
print(f"  sigma_0 (today)   = {sigma0:.5f} daily = {sigma0*math.sqrt(TRADING_DAYS):.2%} p.a.")
print(f"  eps_0  (today)    = {eps0:.5f}\n")

K = w["S0"]
q = float(call_price(w["S0"], K, sigma0 * math.sqrt(TRADING_DAYS), T_DAYS / TRADING_DAYS))
print(f"Black-Scholes premium q = {q:.2f}   (paper: ~70.5)\n")

t0 = time.time()
S, sig = simulate_paths(params, w["S0"], sigma0, eps0, T=T_DAYS, n_paths=200_000, keep_every=1)
print(f"simulated 200k x {T_DAYS} days in {time.time()-t0:.1f}s on "
      f"{'GPU' if torch.cuda.is_available() else 'CPU'}")
r_sim = torch.log(S[:, 1:] / S[:, :-1]).numpy().ravel()
r_hist = w["train_returns"].to_numpy()


def describe(name, r):
    from scipy.stats import kurtosis, skew
    print(f"  {name:10s} mean {r.mean():+.5f}  vol {r.std()*math.sqrt(TRADING_DAYS):6.2%} p.a.  "
          f"skew {skew(r):+.2f}  excess kurt {kurtosis(r):5.2f}")


print("daily log-return statistics:")
describe("historical", r_hist)
describe("simulated", r_sim)
ST = S[:, -1].numpy()
print(f"\nS_T: mean {ST.mean():.1f}  5%/50%/95% = {np.percentile(ST, [5, 50, 95]).round(1)}")
print(f"MC call value E[(S_T-K)+] under the GARCH (real-world) measure = {np.maximum(ST-K,0).mean():.2f}")
