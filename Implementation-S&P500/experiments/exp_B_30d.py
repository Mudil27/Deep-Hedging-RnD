"""
Experiment B - reproduces Figure 13 of Buehler et al. (2019), Section 5.5.

For each of 300 windows (window i ends 2 Jan 2018 minus i weeks):
  * n_ret = 1217 returns; the last T = 30 are the (real) future, the rest the past
  * fit GJR-GARCH-t on the past, simulate paths from "today"
  * train a deep hedge for an ATM call, CVaR objective with alpha = 0.75,
    weekly rebalancing (every 5 trading days -> 6 trades)
  * hedge the REAL next 30 days of the S&P500 with the deep hedge and with
    Black-Scholes (vol = GARCH filter along the realised path); both charged q

Paper: mean P&L 4.3 (deep) vs 3.2 (BS); a loss figure of -7.9 vs -11.6.
Results are appended to a CSV after every window, so the run can be resumed.

usage: python exp_B_30d.py --n_train 500000 [--warm_start] [--windows 0:300]
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import load_prices, get_window
from garch import fit_gjr_garch_t, simulate_paths, filter_sigma
from black_scholes import call_price, bs_hedge_positions, TRADING_DAYS
from network import trading_gains, call_payoff
from train import train_deep_hedge
from evaluate import hedging_errors

ap = argparse.ArgumentParser()
ap.add_argument("--n_train", type=int, default=500_000)
ap.add_argument("--n_val", type=int, default=100_000)
ap.add_argument("--n_test", type=int, default=100_000)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--max_steps", type=int, default=60_000)
ap.add_argument("--eval_every", type=int, default=2_000)
ap.add_argument("--patience", type=int, default=5)
ap.add_argument("--alpha", type=float, default=0.75)
ap.add_argument("--warm_start", action="store_true")
ap.add_argument("--min_steps", type=int, default=30_000,
                help="no early stopping before this many steps (cold start)")
ap.add_argument("--warm_steps", type=int, default=15_000,
                help="max training steps for warm-started windows")
ap.add_argument("--warm_min_steps", type=int, default=6_000)
ap.add_argument("--windows", default="0:300", help="python-style slice of window indices")
ap.add_argument("--tag", default="")
args = ap.parse_args()

T_DAYS, STEP, N_RET = 30, 5, 1217
OUT = ROOT / "results" / f"exp_B{args.tag}"
OUT.mkdir(parents=True, exist_ok=True)
CSV = OUT / "windows.csv"
done = set(pd.read_csv(CSV)["window"]) if CSV.exists() else set()

prices = load_prices()
lo, hi = (int(x) for x in args.windows.split(":"))
prev_state = None

for i in range(lo, hi):
    if i in done:
        continue
    t_start = time.time()
    end_date = pd.Timestamp("2018-01-02") - pd.Timedelta(weeks=i)
    w = get_window(end_date, n_ret=N_RET, T_days=T_DAYS, prices=prices)

    # 1. fit GARCH on the past, premium q = BS price at today's vol --------------
    params, sigma0, eps0, _ = fit_gjr_garch_t(w["train_returns"])
    S0 = K = w["S0"]
    q = float(call_price(S0, K, sigma0 * math.sqrt(TRADING_DAYS), T_DAYS / TRADING_DAYS))

    # 2. simulate + train (CVaR) ---------------------------------------------------
    sim = lambda n, seed: simulate_paths(params, S0, sigma0, eps0, T_DAYS, n,
                                         keep_every=STEP, seed=seed)
    S_tr, _ = sim(args.n_train, seed=1000 + i)
    S_va, _ = sim(args.n_val, seed=2000 + i)
    warm = args.warm_start and prev_state is not None
    model, info = train_deep_hedge(
        S_tr, S_va, K, q, kind="cvar", alpha=args.alpha, lr=args.lr,
        batch_size=args.batch, max_steps=args.warm_steps if warm else args.max_steps,
        eval_every=args.eval_every, patience=args.patience, verbose=False,
        min_steps=args.warm_min_steps if warm else args.min_steps,
        init_state=prev_state if warm else None)
    prev_state = {k: v.clone() for k, v in model.state_dict().items()}
    del S_tr

    # 3. in-model check on fresh simulated paths -------------------------------------
    S_te, sig_te = sim(args.n_test, seed=3000 + i)
    sim_dh, sim_bs = hedging_errors(model, S_te, sig_te, K, q, T_DAYS, STEP)

    # 4. the real test: hedge the actual next 30 days of the S&P500 -------------------
    real = torch.tensor(w["test_prices"].to_numpy(), dtype=torch.float32)  # S_0..S_30
    sig_real = filter_sigma(params, w["test_returns"].to_numpy(), sigma0, eps0)
    S_h = real[::STEP].unsqueeze(0)                                          # (1, 7)
    sig_h = torch.tensor(sig_real[::STEP], dtype=torch.float32).unsqueeze(0)
    dev = next(model.parameters()).device
    with torch.no_grad():
        S_d = S_h.to(dev)
        z = call_payoff(S_d, K)
        pl_dh = float(q - z + trading_gains(model(S_d), S_d))
        pl_bs = float(q - z + trading_gains(
            bs_hedge_positions(S_d, sig_h.to(dev), K, T_DAYS, STEP).float(), S_d))

    row = {
        "window": i, "today": w["today"].date(), "end": w["test_returns"].index[-1].date(),
        "S0": S0, "S_T": float(real[-1]), "q": q, "sigma0_ann": sigma0 * math.sqrt(TRADING_DAYS),
        "payoff": float(z), "pl_deep": pl_dh, "pl_bs": pl_bs,
        "sim_mean_deep": sim_dh.mean(), "sim_mean_bs": sim_bs.mean(),
        "sim_cvar_deep": np.sort(-sim_dh)[-int(len(sim_dh) * (1 - args.alpha)):].mean(),
        "sim_cvar_bs": np.sort(-sim_bs)[-int(len(sim_bs) * (1 - args.alpha)):].mean(),
        "train_steps": info["history"][-1]["step"], "w_var": info["w"],
        "seconds": time.time() - t_start,
        **{f"g_{k}": v for k, v in params.as_dict().items()},
    }
    pd.DataFrame([row]).to_csv(CSV, mode="a", header=not CSV.exists(), index=False)
    print(f"window {i:3d}  {row['today']}  q {q:6.2f}  payoff {row['payoff']:6.2f}  "
          f"PL deep {pl_dh:+7.2f}  BS {pl_bs:+7.2f}  | sim CVaR deep {row['sim_cvar_deep']:.2f} "
          f"BS {row['sim_cvar_bs']:.2f}  | {row['train_steps']} steps {row['seconds']:.0f}s", flush=True)
