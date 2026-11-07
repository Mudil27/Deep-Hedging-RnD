"""
Experiment A - reproduces Figure 12 of Buehler et al. (2019), Section 5.5.

  * window: S&P500 returns ending 2 Jan 2018, n_ret = 1217, T = 240 days
  * fit GJR-GARCH-t on the first 977 returns, simulate paths from "today"
  * rebalance every 10 days (24 trades), objective E[(q - Z + (delta.S)_T)^2], p0 = q
  * compare deep hedge vs daily-recalibrated Black-Scholes on fresh simulated paths

Paper: q ~ 70.5, deep hedge MSE ~20% lower than Black-Scholes.

usage: python exp_A_240d.py --n_train 2000000
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import get_window
from garch import fit_gjr_garch_t, simulate_paths
from black_scholes import call_price, TRADING_DAYS
from train import train_deep_hedge
from evaluate import hedging_errors, compare_table, plot_histogram

ap = argparse.ArgumentParser()
ap.add_argument("--n_train", type=int, default=2_000_000)
ap.add_argument("--n_val", type=int, default=200_000)
ap.add_argument("--n_test", type=int, default=1_000_000)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--max_steps", type=int, default=300_000)
ap.add_argument("--norm", default="input", choices=["input", "batch"])
ap.add_argument("--tag", default="")
args = ap.parse_args()

T_DAYS, STEP = 240, 10
OUT = ROOT / "results" / f"exp_A{args.tag}"
OUT.mkdir(parents=True, exist_ok=True)

# 1. data + GARCH fit --------------------------------------------------------
w = get_window("2018-01-02", n_ret=1217, T_days=T_DAYS)
params, sigma0, eps0, _ = fit_gjr_garch_t(w["train_returns"])
S0 = K = w["S0"]
q = float(call_price(S0, K, sigma0 * math.sqrt(TRADING_DAYS), T_DAYS / TRADING_DAYS))
print(f"today {w['today'].date()}  S0 {S0:.2f}  sigma0 {sigma0*math.sqrt(252):.2%}  q {q:.2f}")

# 2. simulate train / validation / test paths (different seeds = independent) --
t0 = time.time()
sim = lambda n, seed: simulate_paths(params, S0, sigma0, eps0, T_DAYS, n, keep_every=STEP, seed=seed)
S_tr, _ = sim(args.n_train, seed=1)
S_va, _ = sim(args.n_val, seed=2)
S_te, sig_te = sim(args.n_test, seed=3)
print(f"simulated {args.n_train + args.n_val + args.n_test:,} paths in {time.time()-t0:.0f}s")

# 3. train ----------------------------------------------------------------------
model, info = train_deep_hedge(S_tr, S_va, K, q, kind="mse", lr=args.lr, norm=args.norm,
                               batch_size=args.batch, max_steps=args.max_steps)
del S_tr

# 4. evaluate on fresh simulated paths ----------------------------------------
pl_dh, pl_bs = hedging_errors(model, S_te, sig_te, K, q, T_DAYS, STEP)
table, m_dh, m_bs = compare_table(pl_dh, pl_bs)
print("\n" + table)
print(f"\nMSE reduction of deep hedge vs BS: {1 - m_dh['MSE']/m_bs['MSE']:.1%}   (paper: ~20%)")

plot_histogram(pl_dh, pl_bs, OUT / "fig12_hist.png",
               f"Exp A: T=240d, simulated test paths (q = {q:.1f})", xlim=(-260, 110))
torch.save(model.state_dict(), OUT / "model.pt")
json.dump({"args": vars(args), "garch": params.as_dict(), "sigma0": sigma0, "q": q,
           "train": info, "deep_hedge": {k: float(v) for k, v in m_dh.items()},
           "black_scholes": {k: float(v) for k, v in m_bs.items()}},
          open(OUT / "results.json", "w"), indent=2)
print(f"saved to {OUT}")
