"""
Step 2 - GJR-GARCH(1,1) with Student-t innovations, eq. (37) of the paper:

    log(S_k / S_{k-1}) = mu + eps_k
    eps_k              = sigma_k * z_k,        z_k iid standardised t(nu)  (mean 0, var 1)
    sigma_k^2          = omega + alpha*eps_{k-1}^2 + beta*sigma_{k-1}^2
                               + gamma*eps_{k-1}^2 * 1{eps_{k-1} < 0}

fit_gjr_garch_t : maximum-likelihood fit on the training returns (via `arch`)
filter_sigma    : run the variance recursion along a given return path
simulate_paths  : Monte Carlo paths of (S_k, sigma_k), k = 0..T, on GPU if available
"""
from dataclasses import dataclass, asdict

import numpy as np
import torch

SCALE = 100.0  # fit on percentage returns for numerical stability


@dataclass
class GarchParams:
    mu: float
    omega: float
    alpha: float
    gamma: float
    beta: float
    nu: float

    def as_dict(self):
        return asdict(self)

    @property
    def persistence(self):
        # E[sigma^2_k] evolves like (alpha + gamma/2 + beta) per step; must be < 1
        return self.alpha + 0.5 * self.gamma + self.beta

    @property
    def uncond_daily_vol(self):
        return float(np.sqrt(self.omega / (1.0 - self.persistence)))


def fit_gjr_garch_t(train_returns):
    """
    MLE fit of eq. (37) on daily log-returns (decimal, e.g. 0.01 = 1%).

    Returns (params, sigma0, eps0, sigma_hist):
      sigma0     : conditional daily vol of the last training day (k = 0, "today")
      eps0       : residual of the last training day, eps_0 = r_0 - mu
      sigma_hist : filtered daily vol over the whole training period
    """
    from arch import arch_model

    am = arch_model(SCALE * np.asarray(train_returns), mean="Constant",
                    vol="GARCH", p=1, o=1, q=1, dist="t")
    res = am.fit(disp="off")
    p = res.params
    params = GarchParams(
        mu=p["mu"] / SCALE,
        omega=p["omega"] / SCALE**2,
        alpha=p["alpha[1]"],
        gamma=p["gamma[1]"],
        beta=p["beta[1]"],
        nu=p["nu"],
    )
    sigma_hist = np.asarray(res.conditional_volatility) / SCALE
    eps_hist = np.asarray(res.resid) / SCALE
    return params, float(sigma_hist[-1]), float(eps_hist[-1]), sigma_hist


def next_variance(params, sigma2_prev, eps_prev):
    """sigma_k^2 from (sigma_{k-1}^2, eps_{k-1}); works on floats, numpy arrays and torch tensors."""
    neg = (eps_prev < 0) * 1.0
    return (params.omega
            + (params.alpha + params.gamma * neg) * eps_prev**2
            + params.beta * sigma2_prev)


def filter_sigma(params, returns, sigma0, eps0):
    """
    Filtered daily vol along an observed return path r_1..r_T, starting from
    today's (sigma_0, eps_0). Output has length T+1: sigma_0, sigma_1, ..., sigma_T.
    This is what a practitioner would compute from the realised S&P500 path.
    """
    sig2, eps = sigma0**2, eps0
    out = [sigma0]
    for r in np.asarray(returns):
        sig2 = next_variance(params, sig2, eps)
        eps = r - params.mu
        out.append(np.sqrt(sig2))
    return np.array(out)


@torch.no_grad()
def simulate_paths(params, S0, sigma0, eps0, T, n_paths, keep_every=1,
                   device=None, seed=0, chunk=1_000_000):
    """
    Simulate n_paths trajectories of eq. (37) for k = 1..T, starting at today's
    (S0, sigma0, eps0). Only every `keep_every`-th day is stored, i.e. the
    rebalancing dates, which saves a lot of memory for long maturities.

    Returns CPU float32 tensors S, sigma of shape (n_paths, T // keep_every + 1).
    sigma[:, j] is the conditional daily vol in force on stored day j.
    """
    assert T % keep_every == 0
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(seed)
    n_keep = T // keep_every + 1
    t_scale = float(np.sqrt((params.nu - 2.0) / params.nu))  # t(nu) -> unit variance

    S_out = torch.empty(n_paths, n_keep)
    sig_out = torch.empty(n_paths, n_keep)

    for start in range(0, n_paths, chunk):
        n = min(chunk, n_paths - start)
        logS = torch.full((n,), float(np.log(S0)), device=device)
        sig2 = torch.full((n,), sigma0**2, device=device)
        eps = torch.full((n,), eps0, device=device)
        S_out[start:start + n, 0] = float(S0)
        sig_out[start:start + n, 0] = sigma0

        for k in range(1, T + 1):
            sig2 = next_variance(params, sig2, eps)
            z = _student_t(n, params.nu, gen, device) * t_scale
            eps = torch.sqrt(sig2) * z
            logS = logS + params.mu + eps
            if k % keep_every == 0:
                j = k // keep_every
                S_out[start:start + n, j] = torch.exp(logS).cpu()
                sig_out[start:start + n, j] = torch.sqrt(sig2).cpu()
    return S_out, sig_out


def _student_t(n, nu, gen, device):
    """
    Draw n samples of a (non-standardised) t(nu) using a seeded generator:
    t = Z / sqrt(V / nu) with Z ~ N(0,1), V ~ chi^2(nu) = 2 * Gamma(nu/2).
    (torch.distributions doesn't accept a Generator, hence this helper.)
    """
    z = torch.randn(n, generator=gen, device=device)
    # torch._standard_gamma has no generator arg either -> use Marsaglia-Tsang via uniforms/normals
    v = 2.0 * _gamma(nu / 2.0, n, gen, device)
    return z / torch.sqrt(v / nu)


def _gamma(shape, n, gen, device):
    """Marsaglia-Tsang Gamma(shape, 1) sampler (shape >= 1), vectorised with rejection."""
    d = shape - 1.0 / 3.0
    c = 1.0 / np.sqrt(9.0 * d)
    out = torch.empty(n, device=device)
    todo = torch.arange(n, device=device)
    while todo.numel() > 0:
        m = todo.numel()
        x = torch.randn(m, generator=gen, device=device)
        v = (1.0 + c * x) ** 3
        u = torch.rand(m, generator=gen, device=device)
        ok = (v > 0) & (torch.log(u) < 0.5 * x**2 + d - d * v + d * torch.log(v.clamp_min(1e-12)))
        out[todo[ok]] = (d * v)[ok]
        todo = todo[~ok]
    return out
