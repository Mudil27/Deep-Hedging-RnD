"""
Step 5 - Training the deep hedge.

Objectives (Section 3 / 5.5 of the paper), with PL = p0 - Z + (delta . S)_T:

  "mse"  : E[ PL^2 ]                                   (eq. (4), l(x) = x^2, p0 = q)
  "cvar" : min_w { w + E[ (-PL - w)^+ ] / (1 - alpha) }  (OCE form of eq. (28))
           The paper trains w with Adam. For a fixed network the minimising w is
           the alpha-quantile (VaR) of the loss -PL, so we set w to the exact
           batch quantile at every step instead. Same objective, but w can't get
           stuck far from its optimum (with lr = 1e-4 a trainable w moves only
           ~1e-4 per step while the optimal VaR drops from ~70 to ~5 in training).

Optimiser: Adam, mini-batches of random paths, early stopping on a validation set.
"""
import copy
import math
import time

import torch

from network import DeepHedge, trading_gains, call_payoff


def objective(pl, kind, w=None, alpha=None):
    if kind == "mse":
        return (pl**2).mean()
    if kind == "cvar":
        loss = -pl
        if w is None:  # exact inner minimisation: w = batch VaR_alpha (no gradient through w)
            k = math.ceil(alpha * loss.numel())
            # sort + index rather than torch.kthvalue: kthvalue crashed inside the
            # CUDA graph (illegal memory access); a full sort is graph-safe.
            w = torch.sort(loss.detach()).values[k - 1]
        return w + torch.relu(loss - w).mean() / (1.0 - alpha)
    raise ValueError(kind)


def hedge_pl(model, S, K, p0):
    """Terminal hedging error p0 - Z + (delta . S)_T on every path in S."""
    delta = model(S)
    return p0 - call_payoff(S, K) + trading_gains(delta, S)


@torch.no_grad()
def evaluate_loss(model, S, K, p0, kind, w=None, alpha=None, chunk=500_000):
    model.eval()
    pl = torch.cat([hedge_pl(model, S[i:i + chunk], K, p0) for i in range(0, len(S), chunk)])
    model.train()
    if kind == "cvar":  # exact empirical CVaR, not the OCE upper bound at the current w
        return float(empirical_cvar(-pl, alpha))
    return float(objective(pl, kind))


@torch.no_grad()
def recalibrate_batchnorm(model, S_train, batch_size=8192, n_batches=100, gen=None):
    """
    "Precise BatchNorm": in eval mode BN uses running averages of the batch
    mean/variance (momentum 0.1 = roughly the last ~10 batches). With fat-tailed
    paths those are noisy, which made validation scores jump around. Here we
    reset them and recompute an exact average over many training batches.
    """
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    old_momentum = [bn.momentum for bn in bns]
    for bn in bns:
        bn.reset_running_stats()   # in place: keeps the CUDA-graph memory valid
        bn.momentum = None         # None = cumulative (equal-weight) average
    model.train()
    for _ in range(n_batches):
        idx = torch.randint(0, len(S_train), (batch_size,), device=S_train.device, generator=gen)
        model(S_train[idx])
    for bn, m in zip(bns, old_momentum):
        bn.momentum = m


def empirical_cvar(loss, alpha):
    """Average of the worst (1 - alpha) fraction of losses (eq. (28) applied to X = -loss)."""
    n_tail = max(1, int(round((1 - alpha) * loss.numel())))
    return torch.topk(loss, n_tail).values.mean()


def train_deep_hedge(S_train, S_val, K, p0, kind="mse", alpha=None, hidden=20, norm="input",
                     lr=1e-4, batch_size=256, max_steps=300_000, eval_every=5_000,
                     patience=6, min_steps=0, device=None, seed=0, verbose=True,
                     init_state=None):
    """
    S_train, S_val : (n_paths, n_trade + 1) price paths at the rebalancing dates.
    Returns (model, history). The returned model holds the best validation weights.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    S_train, S_val = S_train.to(device), S_val.to(device)
    n_trade = S_train.shape[1] - 1

    model = DeepHedge(n_trade, hidden, norm).to(device)
    if init_state is not None:  # warm start (e.g. from the neighbouring window)
        model.load_state_dict(init_state)
    model.fit_input_stats(S_train)
    params = list(model.parameters())
    w = None
    use_graph = device == "cuda"
    opt = torch.optim.Adam(params, lr=lr, capturable=use_graph)
    gen = torch.Generator(device=device).manual_seed(seed)

    def train_step(S_batch):
        loss = objective(hedge_pl(model, S_batch, K, p0), kind, w, alpha)
        loss.backward()
        opt.step()
        return loss

    # CUDA graph: record forward + backward + Adam once, then replay it as a single
    # GPU call. Same maths, but removes the Python / kernel-launch overhead of
    # running 24 tiny networks one after another (~10x faster here).
    static_S = S_train[:batch_size].clone()
    if use_graph:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm-up iterations (required before capture)
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                train_step(static_S)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        opt.zero_grad(set_to_none=True)
        with torch.cuda.graph(graph):
            static_loss = train_step(static_S)

    best, best_state, bad, history = float("inf"), None, 0, []
    t0, run = time.time(), 0.0
    for step in range(1, max_steps + 1):
        idx = torch.randint(0, len(S_train), (batch_size,), device=device, generator=gen)
        if use_graph:
            static_S.copy_(S_train[idx])
            graph.replay()
            loss = static_loss
        else:
            opt.zero_grad(set_to_none=True)
            loss = train_step(S_train[idx])
        run += loss.item() if step % 100 == 0 else 0.0

        if step % eval_every == 0:
            if norm == "batch":
                recalibrate_batchnorm(model, S_train, gen=gen)
            val = evaluate_loss(model, S_val, K, p0, kind, w, alpha)
            history.append({"step": step, "train": run / (eval_every / 100), "val": val,
                            "minutes": (time.time() - t0) / 60})
            run = 0.0
            if verbose:
                h = history[-1]
                print(f"  step {step:7d}  train {h['train']:10.3f}  val {val:10.3f}  "
                      f"({h['minutes']:.1f} min)", flush=True)
            if best_state is None or val < best - 1e-4 * abs(best):
                best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= patience and step >= min_steps:
                    break

    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": history, "best_val": best,
                   "w": None if w is None else float(w.detach())}
