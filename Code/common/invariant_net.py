import os
import time

SEED = int(os.environ.get("C2_SEED", "42"))

import numpy as np
import torch
import torch.nn as nn


def _prep(X, mu=None, sd=None):
    Z = np.sign(X) * np.log1p(np.abs(X))
    if mu is None:
        mu, sd = Z.mean(0), Z.std(0) + 1e-6
    return ((Z - mu) / sd).astype(np.float32), mu, sd


def fit_predict(ctx, objective, beta, steps, batch, lr, wd, eta, threads):
    torch.set_num_threads(threads)
    torch.manual_seed(SEED + ctx.fold_index)
    rng = np.random.default_rng(SEED + ctx.fold_index)
    Z, mu, sd = _prep(ctx.X_tr)
    y, prof = ctx.y_tr, ctx.prof_tr
    envs = sorted(set(prof[y == 1]))
    neg = np.flatnonzero(y == 0)
    rng.shuffle(neg)
    neg_chunks = np.array_split(neg, len(envs))
    Xp = [torch.from_numpy(Z[(y == 1) & (prof == e)]) for e in envs]
    Xn = [torch.from_numpy(Z[c]) for c in neg_chunks]
    d = Z.shape[1]
    net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    bce = nn.BCEWithLogitsLoss()
    q = torch.ones(len(envs)) / len(envs)
    half = batch // 2
    hist = []
    t0 = time.time()
    for s in range(steps):
        risks = []
        for e in range(len(envs)):
            ip = torch.from_numpy(rng.integers(0, len(Xp[e]), half))
            ineg = torch.from_numpy(rng.integers(0, len(Xn[e]), half))
            xb = torch.cat([Xp[e][ip], Xn[e][ineg]])
            yb = torch.cat([torch.ones(half), torch.zeros(half)])
            risks.append(bce(net(xb).squeeze(1), yb))
        r = torch.stack(risks)
        if objective == "erm":
            loss = r.mean()
        elif objective == "vrex":
            b = beta if s >= steps // 4 else 0.0
            loss = r.mean() + b * r.var()
        else:
            q = q * torch.exp(eta * r.detach())
            q = q / q.sum()
            loss = (q * r).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if s % 200 == 0 or s == steps - 1:
            hist.append({"held_out": ctx.held_out, "step": s, "mean_risk": float(r.mean()),
                         "worst_risk": float(r.max()), "env_risk_var": float(r.var()),
                         "elapsed_min": (time.time() - t0) / 60})
            print(f"    step {s}/{steps}: mean={float(r.mean()):.4f} worst={float(r.max()):.4f}", flush=True)
    net.eval()
    Zt, _, _ = _prep(ctx.X_te, mu, sd)
    out = []
    with torch.no_grad():
        for i in range(0, len(Zt), 200_000):
            out.append(torch.sigmoid(net(torch.from_numpy(Zt[i:i + 200_000])).squeeze(1)).numpy())
    return np.concatenate(out), hist
