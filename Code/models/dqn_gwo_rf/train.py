#!/usr/bin/env python3
"""DQN-guided Grey Wolf feature selection + Random Forest, leave-profiles-out."""
import argparse
import os
import pickle
import platform
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "common"))
import lpo_common as C

torch.set_num_threads(1)

RHO = 0.01
ACTIONS = ["exploit x0.5", "balanced x1.0", "explore x1.5", "OBL"]
A_SCALE = {0: 0.5, 1: 1.0, 2: 1.5}
OBL_ACTION = 3


class DuelingQ(nn.Module):
    def __init__(self, n_in, n_act):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.v = nn.Linear(64, 1)
        self.a = nn.Linear(64, n_act)

    def forward(self, x):
        h = self.body(x)
        a = self.a(h)
        return self.v(h) + a - a.mean(dim=-1, keepdim=True)


class PER:
    """Proportional prioritized experience replay."""
    def __init__(self, cap=2000, alpha=0.6):
        self.cap, self.alpha = cap, alpha
        self.data, self.prio, self.pos = [], np.zeros(cap), 0

    def add(self, tr):
        mx = self.prio[:len(self.data)].max() if self.data else 1.0
        mx = mx if mx > 0 else 1.0
        if len(self.data) < self.cap:
            self.data.append(tr)
        else:
            self.data[self.pos] = tr
        self.prio[self.pos] = mx
        self.pos = (self.pos + 1) % self.cap

    def sample(self, k, beta, rng):
        n = len(self.data)
        p = self.prio[:n] ** self.alpha
        p /= p.sum()
        idx = rng.choice(n, size=min(k, n), p=p)
        w = (n * p[idx]) ** (-beta)
        return idx, [self.data[i] for i in idx], (w / w.max()).astype(np.float32)

    def update(self, idx, td):
        self.prio[idx] = np.abs(td) + 1e-5


class Agent:
    def __init__(self, seed):
        torch.manual_seed(seed)
        self.q, self.tq = DuelingQ(3, len(ACTIONS)), DuelingQ(3, len(ACTIONS))
        self.tq.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=1e-3)
        self.buf = PER()
        self.eps = 1.0

    def act(self, s, rng):
        if rng.random() < self.eps:
            return int(rng.integers(len(ACTIONS)))
        with torch.no_grad():
            return int(self.q(torch.tensor(s)).argmax())

    def learn(self, rng, steps=8, batch=32, gamma=0.99, tau=0.005, beta=0.4):
        if len(self.buf.data) < 8:
            return float("nan")
        losses = []
        for _ in range(steps):
            idx, trs, w = self.buf.sample(batch, beta, rng)
            s = torch.tensor(np.stack([t[0] for t in trs]))
            a = torch.tensor([t[1] for t in trs])
            r = torch.tensor([t[2] for t in trs], dtype=torch.float32)
            s2 = torch.tensor(np.stack([t[3] for t in trs]))
            d = torch.tensor([t[4] for t in trs], dtype=torch.float32)
            q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                tgt = r + gamma * (1 - d) * self.tq(s2).max(1)[0]
            loss = (torch.tensor(w) * nn.functional.smooth_l1_loss(q, tgt, reduction="none")).mean()
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
            self.opt.step()
            self.buf.update(idx, (tgt - q).detach().numpy())
            for p, tp in zip(self.q.parameters(), self.tq.parameters()):
                tp.data.mul_(1 - tau).add_(tau * p.data)
            losses.append(float(loss.detach()))
        return float(np.mean(losses))


def binarize(pos, rng):
    return (rng.random(pos.shape) < 1 / (1 + np.exp(-np.clip(pos, -30, 30)))).astype(np.int8)


def rl_state(fit):
    return np.array([fit.max(), fit.mean(), fit.std() * 10], dtype=np.float32)


class Fitness:
    """Lightweight RF on a subsample (all positives + capped negatives); F1 on the validation
    split minus a feature-count penalty. Cached per feature mask."""
    def __init__(self, X_tr, y_tr, X_val, y_val, rng, neg_cap):
        pos, neg = np.flatnonzero(y_tr == 1), np.flatnonzero(y_tr == 0)
        if len(neg) > neg_cap:
            neg = rng.choice(neg, neg_cap, replace=False)
        idx = np.concatenate([pos, neg])
        self.X, self.y, self.Xv, self.yv = X_tr[idx], y_tr[idx], X_val, y_val
        self.cache, self.n_evals = {}, 0

    def __call__(self, mask):
        if mask.sum() == 0:
            return -1.0
        key = mask.tobytes()
        if key in self.cache:
            return self.cache[key]
        cols = np.flatnonzero(mask)
        clf = RandomForestClassifier(n_estimators=40, max_depth=16, min_samples_leaf=2, n_jobs=-1,
                                     random_state=C.RANDOM_STATE, class_weight="balanced")
        clf.fit(self.X[:, cols], self.y)
        p = clf.predict(self.Xv[:, cols])
        tp = int(((self.yv == 1) & (p == 1)).sum())
        fp = int(((self.yv == 0) & (p == 1)).sum())
        fn = int(((self.yv == 1) & (p == 0)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        val = f1 - RHO * mask.sum() / len(mask)
        self.cache[key] = val
        self.n_evals += 1
        return val


def run_rl_gwo(ev, n_features, rng, agent, gens, n_wolves, deadline, held_out):
    pos = rng.uniform(-1, 1, (n_wolves, n_features))
    binm = np.array([binarize(p, rng) for p in pos])
    fit = np.array([ev(b) for b in binm])
    best_mask, best_fit = binm[int(fit.argmax())].copy(), float(fit.max())
    prev_best = float(fit.max())
    hist, t0 = [], time.time()
    for g in range(gens):
        if time.time() > deadline:
            print("  fold time budget reached; keeping the best subset found so far", flush=True)
            break
        s = rl_state(fit)
        a = agent.act(s, rng)
        if a == OBL_ACTION:
            opp_pos, opp_bin = -pos, (1 - binm).astype(np.int8)
            opp_fit = np.array([ev(b) for b in opp_bin])
            all_pos, all_bin = np.vstack([pos, opp_pos]), np.vstack([binm, opp_bin])
            all_fit = np.concatenate([fit, opp_fit])
            keep = np.argsort(-all_fit)[:n_wolves]
            pos, binm, fit = all_pos[keep], all_bin[keep], all_fit[keep]
        else:
            a_par = (2 - 2 * g / gens) * A_SCALE[a]
            leaders = pos[np.argsort(-fit)[:3]].copy()
            new_pos = np.empty_like(pos)
            for i in range(n_wolves):
                acc = np.zeros(n_features)
                for L in leaders:
                    r1, r2 = rng.random(n_features), rng.random(n_features)
                    acc += L - (2 * a_par * r1 - a_par) * np.abs(2 * r2 * L - pos[i])
                new_pos[i] = acc / 3
            pos = new_pos
            binm = np.array([binarize(p, rng) for p in pos])
            fit = np.array([ev(b) for b in binm])
        b = int(fit.argmax())
        if fit[b] > best_fit:
            best_fit, best_mask = float(fit[b]), binm[b].copy()
        reward = (float(fit.max()) - prev_best) * 100.0
        prev_best = float(fit.max())
        agent.buf.add((s, a, reward, rl_state(fit), float(g == gens - 1)))
        loss = agent.learn(rng)
        eps_used = agent.eps
        agent.eps = max(0.05, agent.eps * 0.97)
        hist.append({"held_out": held_out, "generation": g + 1, "best_fitness": float(fit.max()),
                     "mean_fitness": float(fit.mean()), "best_n_features": int(binm[b].sum()),
                     "action": ACTIONS[a], "reward": reward, "epsilon": eps_used, "loss": loss,
                     "fitness_evals": ev.n_evals, "elapsed_min": (time.time() - t0) / 60})
        print(f"    gen {g + 1}/{gens}: {ACTIONS[a]:<14} best={fit.max():.4f} "
              f"features={int(binm[b].sum())} reward={reward:+.3f} eps={eps_used:.2f} "
              f"evals={ev.n_evals}", flush=True)
    return best_mask, hist


def plot_dqn(h, path, title):
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    counts = h.groupby(["held_out", "action"]).size().unstack(fill_value=0).reindex(
        columns=ACTIONS, fill_value=0)
    counts.plot(kind="bar", ax=a1)
    a1.tick_params(axis="x", rotation=20)
    a1.set_title("RL agent action choices per fold")
    x = np.arange(len(h))
    a2.plot(x, h["reward"], label="reward (x100)")
    a2.plot(x, h["epsilon"], label="epsilon")
    a2.set_xlabel("learning step (all folds)")
    a2.legend()
    a2.set_title("Reward and exploration rate")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--name", default="dataset")
    ap.add_argument("--variant", default="mixed", choices=["mixed", "https"])
    ap.add_argument("--wolves", type=int, default=12)
    ap.add_argument("--gens", type=int, default=100)
    ap.add_argument("--max-hours", type=float, default=66.0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.wolves, args.gens = 4, 3
    neg_cap = 20_000 if args.smoke else 250_000
    final_trees = 50 if args.smoke else 300

    run_dir = args.run_dir
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)
    ck_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    label = os.path.basename(os.path.normpath(run_dir))
    t_start = time.time()

    df, feature_cols, dropped = C.load_data(args.data, args.variant, args.smoke)
    n_features = len(feature_cols)
    X_all = df[feature_cols].to_numpy(dtype=np.float32)
    y_all = df["label"].to_numpy()
    profiles = df["profile"].to_numpy()
    malicious, normal_test_idx, n_normal_train = C.build_folds_time(df)

    zero = sorted(set(profiles) - set(malicious) - {"normal"})
    print(f"[{label}] {len(df):,} rows, {n_features} features, dataset={args.name}, variant={args.variant}")
    print(f"Constant features dropped: {dropped}")
    print(f"Malicious profiles (one fold each): {malicious}")
    if zero:
        print(f"*** WARNING: profiles with ZERO C2 flows (treated as benign): {zero} ***")
    print(f"Normal: {n_normal_train:,} train rows, {len(normal_test_idx):,} held-out (last 20% in time)")

    C.write_json(os.path.join(run_dir, "run_config.json"), {
        "model": "EnhancedRLGWO (DuelingDQN+PER+OBL) + RF", "dataset": args.name,
        "variant": args.variant, "data": args.data, "rows": len(df), "features": feature_cols,
        "dropped_constant_features": dropped, "malicious_profiles": malicious,
        "wolves": args.wolves, "generations": args.gens, "rho": RHO, "fitness_neg_cap": neg_cap,
        "final_rf_trees": final_trees, "max_hours": args.max_hours, "random_state": C.RANDOM_STATE,
        "smoke": args.smoke, "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "")})

    agent = Agent(C.RANDOM_STATE)
    agent_path = os.path.join(ck_dir, "agent.pt")
    if os.path.exists(agent_path):
        st = torch.load(agent_path, map_location="cpu")
        agent.q.load_state_dict(st["q"])
        agent.tq.load_state_dict(st["q"])
        agent.eps = st["eps"]
        print(f"Resumed RL agent (epsilon={agent.eps:.2f})")

    rows, history, selected_by_fold, importances, curves = [], [], {}, {}, {}

    def ckpt(fi):
        return os.path.join(ck_dir, f"fold_{fi}.pkl")

    def refresh():
        if not rows:
            return
        fig = os.path.join(run_dir, "figures")
        C.write_common_outputs(run_dir, label, rows, curves, "DQN-GWO+RF", "RF all features")
        h = pd.DataFrame(history)
        h.to_csv(os.path.join(run_dir, "gwo_history.csv"), index=False)
        C.plot_gwo_history(h, os.path.join(fig, "optimizer_convergence.png"), label)
        plot_dqn(h, os.path.join(fig, "dqn_actions_reward.png"), label)
        C.plot_selection_heatmap(selected_by_fold, feature_cols,
                                 os.path.join(fig, "feature_selection_heatmap.png"), label)
        imp = pd.DataFrame(importances).T
        imp.to_csv(os.path.join(run_dir, "feature_importance.csv"))
        C.plot_importance(imp, os.path.join(fig, "feature_importance.png"), label)

    for fi, held_out in enumerate(malicious):
        print(f"\n=== Fold {fi + 1}/{len(malicious)}: holding out {held_out} ===", flush=True)
        if os.path.exists(ckpt(fi)):
            d = pickle.load(open(ckpt(fi), "rb"))
            rows.append(d["row"])
            curves[held_out] = d["curve"]
            selected_by_fold[held_out] = d["selected"]
            importances[held_out] = d["importance"]
            history.extend(d["history"])
            print("  already done in a previous run; skipping")
            continue

        tr_mask, te_mask = C.split_fold(df, held_out, normal_test_idx)
        X_tr_full, y_tr_full = X_all[tr_mask], y_all[tr_mask]
        X_te, y_te = X_all[te_mask], y_all[te_mask]
        normal_te = profiles[te_mask] == "normal"
        rng = np.random.default_rng(C.RANDOM_STATE + fi)

        val = np.zeros(len(X_tr_full), dtype=bool)
        val[rng.choice(len(X_tr_full), size=max(1, len(X_tr_full) // 5), replace=False)] = True
        ev = Fitness(X_tr_full[~val], y_tr_full[~val], X_tr_full[val], y_tr_full[val], rng, neg_cap)

        n_left = sum(1 for j in range(fi, len(malicious)) if not os.path.exists(ckpt(j)))
        remaining = t_start + args.max_hours * 3600 - time.time()
        fold_deadline = time.time() + 0.85 * remaining / max(n_left, 1)

        t_gwo = time.time()
        best_mask, fold_hist = run_rl_gwo(ev, n_features, rng, agent, args.gens, args.wolves,
                                          fold_deadline, held_out)
        gwo_min = (time.time() - t_gwo) / 60
        history.extend(fold_hist)
        cols = np.flatnonzero(best_mask)
        if len(cols) == 0:
            cols = np.arange(n_features)
        selected = [feature_cols[i] for i in cols]
        selected_by_fold[held_out] = selected
        print(f"  Selected {len(selected)} features: {selected}", flush=True)

        clf = RandomForestClassifier(n_estimators=final_trees, n_jobs=-1,
                                     random_state=C.RANDOM_STATE, class_weight="balanced")
        clf.fit(X_tr_full[:, cols], y_tr_full)
        pred = clf.predict(X_te[:, cols])
        score = clf.predict_proba(X_te[:, cols])[:, 1]
        importances[held_out] = pd.Series(clf.feature_importances_, index=selected)
        del clf

        base = RandomForestClassifier(n_estimators=final_trees, n_jobs=-1,
                                      random_state=C.RANDOM_STATE, class_weight="balanced")
        base.fit(X_tr_full, y_tr_full)
        bpred = base.predict(X_te)
        del base

        m = C.evaluate(y_te, pred, normal_te, rng)
        bm = C.evaluate(y_te, bpred, normal_te, rng)
        curves[held_out] = C.roc_pr_curves(y_te, score)

        row = {"held_out": held_out, "n_train": int(tr_mask.sum()), "n_selected": len(selected),
               "selected": ";".join(selected), "roc_auc": curves[held_out]["roc_auc"],
               "ap": curves[held_out]["ap"], "fitness_evals": ev.n_evals, "gwo_minutes": gwo_min}
        row.update(m)
        row.update({"base_" + k: v for k, v in bm.items()})
        rows.append(row)

        print(f"  TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}  F1={m['f1']:.4f} "
              f"[{m['f1_lo']:.4f},{m['f1_hi']:.4f}]  P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"FPR(normal)={m['fpr_normal']:.5f}")
        print(f"  [baseline, all features] F1={bm['f1']:.4f}  P={bm['precision']:.4f}  "
              f"R={bm['recall']:.4f}  FPR(normal)={bm['fpr_normal']:.5f}", flush=True)

        pickle.dump({"row": row, "curve": curves[held_out], "selected": selected,
                     "importance": importances[held_out], "history": fold_hist},
                    open(ckpt(fi), "wb"))
        torch.save({"q": agent.q.state_dict(), "eps": agent.eps}, agent_path)
        refresh()

    refresh()
    invariant = sorted(set.intersection(*[set(s) for s in selected_by_fold.values()])) \
        if selected_by_fold else []
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write(f"{label}\nrows={len(df)} elapsed_h={(time.time() - t_start) / 3600:.2f}\n\n")
        f.write(pd.DataFrame(rows)[["held_out", "f1", "precision", "recall", "fpr_normal",
                                    "roc_auc", "n_selected"]].to_string(index=False))
        f.write(f"\n\nFeatures selected in EVERY fold: {invariant}\n")
    open(os.path.join(run_dir, "DONE"), "w").write("complete\n")
    print(f"\nInvariant features: {invariant}\nDone in {(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
