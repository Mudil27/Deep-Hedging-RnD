"""
Step 4 - The deep hedging network (Section 5.1 / 5.5 of the paper).

Semi-recurrent structure: one feed-forward net F_j per rebalancing date t_j,

    delta_j = F_j( log S_j , delta_{j-1} ),     delta_{-1} = 0

Each F_j: input (d + d = 2) -> 20 -> 20 -> 1   (N1 = N2 = 20 in Section 5.5),
ReLU activations, linear output layer.

Normalisation (norm=...):
  "batch" : batch normalisation right before every hidden ReLU (as in the paper)
  "input" : no BatchNorm; log S_j is standardised with FIXED per-date mean/std
            computed once from the training paths (our default, see README:
            BatchNorm's running statistics are unreliable with fat-tailed paths)
"""
import torch
from torch import nn


class StepNet(nn.Module):
    def __init__(self, n_in=2, hidden=20, n_out=1, batchnorm=False):
        super().__init__()
        bn = (lambda: nn.BatchNorm1d(hidden)) if batchnorm else nn.Identity
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), bn(), nn.ReLU(),
            nn.Linear(hidden, hidden), bn(), nn.ReLU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


class DeepHedge(nn.Module):
    """n_trade StepNets chained through time via the previous position."""

    def __init__(self, n_trade, hidden=20, norm="input"):
        super().__init__()
        assert norm in ("input", "batch")
        self.norm = norm
        self.nets = nn.ModuleList(StepNet(2, hidden, 1, batchnorm=(norm == "batch"))
                                  for _ in range(n_trade))
        # fixed input statistics (saved with the model); identity until fitted
        self.register_buffer("x_mean", torch.zeros(n_trade))
        self.register_buffer("x_std", torch.ones(n_trade))

    @torch.no_grad()
    def fit_input_stats(self, S):
        """Per-date mean / std of log S_j over the training paths (only for norm='input')."""
        if self.norm == "input":
            logS = torch.log(S[:, :-1].to(self.x_mean.device))
            self.x_mean.copy_(logS.mean(0))
            self.x_std.copy_(logS.std(0).clamp_min(1e-3))  # t_0: all paths equal -> std 0

    def forward(self, S):
        """
        S : (batch, n_trade + 1) prices at the rebalancing dates t_0..t_n (t_n = maturity)
        returns positions delta : (batch, n_trade), delta[:, j] held over [t_j, t_{j+1}]
        """
        x = (torch.log(S[:, :-1]) - self.x_mean) / self.x_std
        delta_prev = torch.zeros_like(S[:, :1])
        deltas = []
        for j, net in enumerate(self.nets):
            delta_prev = net(torch.cat([x[:, j:j + 1], delta_prev], dim=1))
            deltas.append(delta_prev)
        return torch.cat(deltas, dim=1)


def trading_gains(delta, S):
    """(delta . S)_T = sum_j delta_j (S_{j+1} - S_j)  for each path."""
    return (delta * (S[:, 1:] - S[:, :-1])).sum(dim=1)


def call_payoff(S, K):
    return torch.clamp(S[:, -1] - K, min=0.0)
