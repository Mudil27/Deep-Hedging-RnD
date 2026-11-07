"""
Experiment B, vectorised: identical method to exp_B_30d.py, but the deep hedges of
`--group` windows are trained simultaneously (see src/multi.py). Same CSV format,
resumable, ~10x faster.

usage: python exp_B_30d_fast.py --tag _fast --group 25 [--windows 0:300]
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
from multi import train_multi, multi_pl

ap = argparse.ArgumentParser()
ap.add_argument("--n_train", type=int, default=500_000)
ap.add_argument("--n_val", type=int, default=100_000)
ap.add_argument("--n_test", type=int, default=100_000)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--batch", type=int, default=4096)
ap.add_argument("--steps", type=int, default=60_000)
ap.add_argument("--alpha", type=float, default=0.75)
ap.add_argument("--group", type=int, default=25)
ap.add_argument("--windows", default="0:300")
ap.add_argument("--tag", default="_fast")
args = ap.parse_args()

T_DAYS, STEP, N_RET = 30, 5, 1217
dev = "cuda"
OUT = ROOT / "results" / f"exp_B{args.tag}"
OUT.mkdir(parents=True, exist_ok=True)
CSV = OUT / "windows.csv"
done = set(pd.read_csv(CSV)["window"]) if CSV.exists() else set()
prices = load_prices()
lo, hi = (int(x) for x in args.windows.split(":"))
todo = [i for i in range(lo, hi) if i not in done]
cvar = lambda pl: np.sort(-pl)[-int(len(pl) * (1 - args.alpha)):].mean()

for g in range(0, len(todo), args.group):
    group = todo[g:g + args.group]
    t0 = time.time()
    # 1. per window: data, GARCH fit, premium, simulated paths --------------------
    W = []
    for i in group:
        w = get_window(pd.Timestamp("2018-01-02") - pd.Timedelta(weeks=i),
                       n_ret=N_RET, T_days=T_DAYS, prices=prices)
        params, sigma0, eps0, _ = fit_gjr_garch_t(w["train_returns"])
        S0 = w["S0"]
        q = float(call_price(S0, S0, sigma0 * math.sqrt(TRADING_DAYS), T_DAYS / TRADING_DAYS))
        sim = lambda n, seed: simulate_paths(params, S0, sigma0, eps0, T_DAYS, n,
                                             keep_every=STEP, seed=seed)
        S_tr, _ = sim(args.n_train, 1000 + i)
        S_va, _ = sim(args.n_val, 2000 + i)
        S_te, sig_te = sim(args.n_test, 3000 + i)
        W.append(dict(i=i, w=w, params=params, sigma0=sigma0, eps0=eps0, S0=S0, q=q,
                      S_tr=S_tr, S_va=S_va, S_te=S_te, sig_te=sig_te))
    t_prep = time.time() - t0

    # 2. train all windows of the group at once ------------------------------------
    K = torch.tensor([x["S0"] for x in W], device=dev)
    q = torch.tensor([x["q"] for x in W], device=dev)
    S_tr = torch.stack([x.pop("S_tr") for x in W]).to(dev)
    S_va = torch.stack([x.pop("S_va") for x in W]).to(dev)
    model, best_val = train_multi(S_tr, S_va, K, q, alpha=args.alpha, lr=args.lr,
                                  batch_size=args.batch, steps=args.steps, verbose=False)
    del S_tr, S_va
    model.eval()

    # 3. evaluate: simulated test paths + the real S&P path, deep vs BS ----------------
    with torch.no_grad():
        S_te = torch.stack([x["S_te"] for x in W]).to(dev)
        sim_dh = multi_pl(model, S_te, K, q).cpu().numpy()
        real = torch.stack([torch.tensor(x["w"]["test_prices"].to_numpy()[::STEP],
                                         dtype=torch.float32) for x in W]).unsqueeze(1).to(dev)
        real_dh = multi_pl(model, real, K, q)[:, 0].cpu().numpy()

    rows = []
    for n, x in enumerate(W):
        Kn, qn = float(K[n]), float(q[n])
        s_te, sg_te = S_te[n], x["sig_te"].to(dev)
        z = torch.clamp(s_te[:, -1] - Kn, min=0)
        bs_te = qn - z + (bs_hedge_positions(s_te, sg_te, Kn, T_DAYS, STEP).float()
                          * (s_te[:, 1:] - s_te[:, :-1])).sum(1)
        sig_real = filter_sigma(x["params"], x["w"]["test_returns"].to_numpy(), x["sigma0"], x["eps0"])
        s_h = real[n]
        sg_h = torch.tensor(sig_real[::STEP], dtype=torch.float32, device=dev)[None]
        z_h = float(torch.clamp(s_h[0, -1] - Kn, min=0))
        pl_bs = float(qn - z_h + (bs_hedge_positions(s_h, sg_h, Kn, T_DAYS, STEP).float()
                                  * (s_h[:, 1:] - s_h[:, :-1])).sum())
        bs_te = bs_te.cpu().numpy()
        rows.append({
            "window": x["i"], "today": x["w"]["today"].date(),
            "end": x["w"]["test_returns"].index[-1].date(),
            "S0": x["S0"], "S_T": float(s_h[0, -1]), "q": qn,
            "sigma0_ann": x["sigma0"] * math.sqrt(TRADING_DAYS), "payoff": z_h,
            "pl_deep": float(real_dh[n]), "pl_bs": pl_bs,
            "sim_mean_deep": sim_dh[n].mean(), "sim_mean_bs": bs_te.mean(),
            "sim_cvar_deep": cvar(sim_dh[n]), "sim_cvar_bs": cvar(bs_te),
            "val_cvar_deep": float(best_val[n]), "train_steps": args.steps,
            **{f"g_{k}": v for k, v in x["params"].as_dict().items()},
        })
    pd.DataFrame(rows).to_csv(CSV, mode="a", header=not CSV.exists(), index=False)
    d = pd.DataFrame(rows)
    print(f"windows {group[0]}..{group[-1]}: prep {t_prep:.0f}s, total {time.time()-t0:.0f}s | "
          f"real PL deep {d.pl_deep.mean():+.2f} BS {d.pl_bs.mean():+.2f} | "
          f"sim CVaR deep {d.sim_cvar_deep.mean():.2f} BS {d.sim_cvar_bs.mean():.2f}", flush=True)
