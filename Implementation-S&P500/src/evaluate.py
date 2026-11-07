"""
Step 6 - Compare deep hedge vs Black-Scholes on the same paths.

Both strategies are charged the same premium q (the Black-Scholes price), exactly
as in the paper, so differences come only from the trading strategy.
"""
import numpy as np
import torch

from black_scholes import bs_hedge_positions
from network import trading_gains, call_payoff


@torch.no_grad()
def hedging_errors(model, S, sigma, K, q, T_days, step, chunk=500_000, device=None):
    """Terminal hedging errors q - Z + (delta . S)_T for the deep hedge and for BS."""
    device = device or next(model.parameters()).device
    dh, bs = [], []
    for i in range(0, len(S), chunk):
        s = S[i:i + chunk].to(device)
        sg = sigma[i:i + chunk].to(device)
        z = call_payoff(s, K)
        dh.append(q - z + trading_gains(model(s), s))
        bs.append(q - z + trading_gains(bs_hedge_positions(s, sg, K, T_days, step).float(), s))
    return torch.cat(dh).cpu().numpy(), torch.cat(bs).cpu().numpy()


def metrics(pl, alpha=0.75):
    loss = -pl
    tail = np.sort(loss)[-max(1, int(round((1 - alpha) * len(loss)))):]
    return {
        "mean P&L": pl.mean(),
        "std": pl.std(),
        "MSE": (pl**2).mean(),
        "RMSE": np.sqrt((pl**2).mean()),
        f"CVaR{alpha:g} (loss)": tail.mean(),
        "mean of losses E[min(PL,0)]": np.minimum(pl, 0).mean(),
        "1% quantile": np.quantile(pl, 0.01),
        "5% quantile": np.quantile(pl, 0.05),
        "median": np.median(pl),
    }


def compare_table(pl_dh, pl_bs, alpha=0.75):
    m_dh, m_bs = metrics(pl_dh, alpha), metrics(pl_bs, alpha)
    lines = [f"{'metric':32s} {'Deep hedge':>12s} {'Black-Scholes':>14s}"]
    for k in m_dh:
        lines.append(f"{k:32s} {m_dh[k]:12.3f} {m_bs[k]:14.3f}")
    return "\n".join(lines), m_dh, m_bs


def plot_histogram(pl_dh, pl_bs, path, title, bins=60, xlim=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, hi = xlim or (np.quantile(np.r_[pl_dh, pl_bs], 0.001), np.quantile(np.r_[pl_dh, pl_bs], 0.999))
    edges = np.linspace(lo, hi, bins + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pl_dh, edges, alpha=0.6, label="Deep hedge")
    ax.hist(pl_bs, edges, alpha=0.6, label="Black-Scholes")
    ax.set_xlabel("terminal hedging error  q - Z + (δ·S)_T")
    ax.set_ylabel("number of paths")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
