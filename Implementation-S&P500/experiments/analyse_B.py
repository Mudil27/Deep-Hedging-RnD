"""
Analyse Experiment B (Figure 13): terminal hedging errors on the REAL S&P500
paths of all finished windows, deep hedge vs Black-Scholes.

Paper: mean P&L 4.3 (deep) vs 3.2 (BS), and a loss figure of -7.9 vs -11.6
(called "mean square hedging loss" there, but it is negative, so we print
several candidate metrics to see which one matches).

usage: python analyse_B.py [--tag _final]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluate import metrics, plot_histogram

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="_final")
args = ap.parse_args()
OUT = ROOT / "results" / f"exp_B{args.tag}"
d = pd.read_csv(OUT / "windows.csv").sort_values("window")
print(f"{len(d)} windows: {d['today'].min()} .. {d['today'].max()}\n")

dh, bs = d["pl_deep"].to_numpy(), d["pl_bs"].to_numpy()
m_dh, m_bs = metrics(dh), metrics(bs)
print(f"{'metric (real S&P paths)':32s} {'Deep hedge':>12s} {'Black-Scholes':>14s}")
for k in m_dh:
    print(f"{k:32s} {m_dh[k]:12.3f} {m_bs[k]:14.3f}")
print(f"{'CVaR0.75 of P&L (= -loss)':32s} {-m_dh['CVaR0.75 (loss)']:12.3f} {-m_bs['CVaR0.75 (loss)']:14.3f}")

diff = dh - bs
se = diff.std(ddof=1) / np.sqrt(len(diff))
print(f"\nmean(deep - BS) = {diff.mean():+.3f}   naive s.e. {se:.3f}   "
      f"(windows overlap, so the true s.e. is larger)")
print(f"deep hedge better in {np.mean(diff > 0):.0%} of windows")

print("\nin-model (simulated test paths, averaged over windows):")
print(f"  CVaR0.75 loss  deep {d['sim_cvar_deep'].mean():.3f}   BS {d['sim_cvar_bs'].mean():.3f}")
print(f"  mean P&L       deep {d['sim_mean_deep'].mean():.3f}   BS {d['sim_mean_bs'].mean():.3f}")

plot_histogram(dh, bs, OUT / "fig13_hist.png",
               f"Exp B: T=30d, {len(d)} historical S&P500 windows", bins=32, xlim=(-80, 80))
print(f"\nsaved {OUT / 'fig13_hist.png'}")
