"""
Speed-up for Experiment B: train the deep hedges of W windows *simultaneously*.

Each window still has its own networks, paths, premium and strike. We only stack
the weights of all windows into tensors with a leading "window" dimension, so that
one batched matmul (torch.baddbmm) evaluates W networks in a single GPU call.

The total loss is the SUM of the per-window CVaRs. Windows share no parameters
and Adam updates every parameter element-wise, so each window is trained exactly
as if it were trained alone - the method is unchanged, only the bookkeeping.
(Training tiny nets is launch-overhead-bound: one window or 25 cost the same time.)
"""
import math
import time

import torch
from torch import nn


class MultiDeepHedge(nn.Module):
    """W independent copies of network.DeepHedge(norm='input'), 2 -> H -> H -> 1 per date."""

    def __init__(self, n_windows, n_trade, hidden=20):
        super().__init__()
        W, n, H = n_windows, n_trade, hidden

        def lin(fan_in, fan_out):  # PyTorch nn.Linear default init, U(-1/sqrt(fan_in), +)
            b = 1.0 / math.sqrt(fan_in)
            return (nn.Parameter(torch.empty(W, n, fan_in, fan_out).uniform_(-b, b)),
                    nn.Parameter(torch.empty(W, n, 1, fan_out).uniform_(-b, b)))

        self.W1, self.b1 = lin(2, H)
        self.W2, self.b2 = lin(H, H)
        self.W3, self.b3 = lin(H, 1)
        self.register_buffer("x_mean", torch.zeros(W, 1, n))
        self.register_buffer("x_std", torch.ones(W, 1, n))
        self.n_trade = n

    @torch.no_grad()
    def fit_input_stats(self, S):
        """S : (W, N, n+1). Per-window, per-date mean/std of log S_j."""
        logS = torch.log(S[:, :, :-1])
        self.x_mean.copy_(logS.mean(1, keepdim=True))
        self.x_std.copy_(logS.std(1, keepdim=True).clamp_min(1e-3))

    def forward(self, S):
        """S : (W, B, n+1) -> positions (W, B, n)."""
        x = (torch.log(S[:, :, :-1]) - self.x_mean) / self.x_std
        delta = torch.zeros_like(S[:, :, :1])
        out = []
        for j in range(self.n_trade):
            h = torch.cat([x[:, :, j:j + 1], delta], dim=2)                     # (W, B, 2)
            h = torch.relu(torch.baddbmm(self.b1[:, j], h, self.W1[:, j]))
            h = torch.relu(torch.baddbmm(self.b2[:, j], h, self.W2[:, j]))
            delta = torch.baddbmm(self.b3[:, j], h, self.W3[:, j])              # (W, B, 1)
            out.append(delta)
        return torch.cat(out, dim=2)


def multi_pl(model, S, K, p0):
    """S (W,B,n+1), K and p0 (W,) -> hedging errors p0 - Z + (delta.S)_T, shape (W,B)."""
    delta = model(S)
    gains = (delta * (S[:, :, 1:] - S[:, :, :-1])).sum(2)
    payoff = torch.clamp(S[:, :, -1] - K[:, None], min=0.0)
    return p0[:, None] - payoff + gains


def multi_cvar(pl, alpha):
    """Per-window OCE/CVaR with w = exact batch VaR (as in train.objective)."""
    loss = -pl
    k = math.ceil(alpha * loss.shape[1])
    w = torch.sort(loss.detach(), dim=1).values[:, k - 1:k]
    return (w[:, 0] + torch.relu(loss - w).mean(1) / (1.0 - alpha))


@torch.no_grad()
def multi_empirical_cvar(model, S, K, p0, alpha, chunk=50_000):
    pl = torch.cat([multi_pl(model, S[:, i:i + chunk], K, p0)
                    for i in range(0, S.shape[1], chunk)], dim=1)
    n_tail = max(1, round((1 - alpha) * pl.shape[1]))
    return torch.topk(-pl, n_tail, dim=1).values.mean(1)


def train_multi(S_train, S_val, K, p0, alpha=0.75, hidden=20, lr=1e-4, batch_size=4096,
                steps=60_000, eval_every=2_000, seed=0, verbose=True):
    """
    S_train, S_val : (W, N, n+1) on GPU; K, p0 : (W,) tensors.
    Fixed number of steps; for each window the weights with the best validation
    CVaR are kept (per-window checkpointing).
    """
    dev = S_train.device
    torch.manual_seed(seed)
    W, N, _ = S_train.shape
    model = MultiDeepHedge(W, S_train.shape[2] - 1, hidden).to(dev)
    model.fit_input_stats(S_train)
    opt = torch.optim.Adam(model.parameters(), lr=lr, capturable=True)
    gen = torch.Generator(device=dev).manual_seed(seed)
    rows = torch.arange(W, device=dev)[:, None]

    def step_fn(Sb):
        loss = multi_cvar(multi_pl(model, Sb, K, p0), alpha).sum()
        loss.backward()
        opt.step()
        return loss

    # CUDA graph (same trick as train.py)
    static_S = S_train[:, :batch_size].clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            step_fn(static_S)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    opt.zero_grad(set_to_none=True)
    with torch.cuda.graph(graph):
        static_loss = step_fn(static_S)

    best = torch.full((W,), float("inf"), device=dev)
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.randint(0, N, (W, batch_size), device=dev, generator=gen)
        static_S.copy_(S_train[rows, idx])
        graph.replay()
        if step % eval_every == 0:
            val = multi_empirical_cvar(model, S_val, K, p0, alpha)
            better = val < best
            best = torch.where(better, val, best)
            for k, v in model.state_dict().items():  # copy improved windows only
                m = better.view(-1, *([1] * (v.dim() - 1)))
                best_state[k] = torch.where(m, v.detach(), best_state[k])
            if verbose:
                print(f"  step {step:6d}  loss/window {static_loss.item()/W:8.3f}  "
                      f"val CVaR mean {val.mean():.3f}  improved {int(better.sum())}/{W}  "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
    model.load_state_dict(best_state)
    return model, best
