"""
Step 3 - Black-Scholes benchmark (r = 0, no dividends).

The practitioner's strategy of Section 5.5: at every rebalancing date k, plug the
GARCH-filtered volatility into the Black-Scholes delta and hold that many shares.
The premium q is the Black-Scholes price at the volatility of "today" (sigma_0).
"""
import math

import torch

TRADING_DAYS = 252  # daily vol -> annual vol:  sigma_ann = sigma_daily * sqrt(252)


def _d1_d2(S, K, sigma, tau):
    vol_sqrt_t = sigma * torch.sqrt(tau)
    d1 = (torch.log(S / K) + 0.5 * vol_sqrt_t**2) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def _N(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def call_price(S, K, sigma, tau):
    """BS call price; sigma annualised, tau in years. Accepts floats or tensors."""
    S, K, sigma, tau = (torch.as_tensor(x, dtype=torch.float64) for x in (S, K, sigma, tau))
    d1, d2 = _d1_d2(S, K, sigma, tau)
    return S * _N(d1) - K * _N(d2)


def call_delta(S, K, sigma, tau):
    """BS delta N(d1); tensors of matching (broadcastable) shape."""
    d1, _ = _d1_d2(S, K, sigma, tau)
    return _N(d1)


def bs_hedge_positions(S, sigma_daily, K, T_days, step):
    """
    Black-Scholes delta at each rebalancing date.

    S, sigma_daily : (n_paths, n_dates) prices and GARCH vols at the rebalancing
                     dates t_j = j * step days, j = 0..n_dates-1 (last one = maturity)
    returns        : (n_paths, n_dates - 1) positions delta_j held over [t_j, t_{j+1}]
    """
    n_trade = S.shape[1] - 1
    days_left = T_days - step * torch.arange(n_trade, device=S.device, dtype=S.dtype)
    tau = (days_left / TRADING_DAYS).expand(S.shape[0], -1)
    sigma_ann = sigma_daily[:, :n_trade] * math.sqrt(TRADING_DAYS)
    return call_delta(S[:, :n_trade], K, sigma_ann, tau)
