#!/usr/bin/env python3
"""Grey Wolf feature selection + Random Forest, leave-profiles-out."""
import argparse
import os
import pickle
import platform
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "common"))
import lpo_common as C

RHO = 0.01


def binarize(pos, rng):
    return (rng.random(pos.shape) < 1 / (1 + np.exp(-np.clip(pos, -30, 30)))).astype(np.int8)


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


def run_gwo(ev, n_features, rng, gens, n_wolves, deadline, held_out):
    pos = rng.uniform(-1, 1, (n_wolves, n_features))
    binm = np.array([binarize(p, rng) for p in pos])
    fit = np.array([ev(b) for b in binm])
    best_mask, best_fit = binm[int(fit.argmax())].copy(), float(fit.max())
    hist, t0 = [], time.time()
    for g in range(gens):
        if time.time() > deadline:
            print("  fold time budget reached; keeping the best subset found so far", flush=True)
            break
        a_par = 2 - 2 * g / gens
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
        hist.append({"held_out": held_out, "generation": g + 1, "best_fitness": float(fit.max()),
                     "mean_fitness": float(fit.mean()), "best_n_features": int(binm[b].sum()),
                     "fitness_evals": ev.n_evals, "elapsed_min": (time.time() - t0) / 60})
        print(f"    gen {g + 1}/{gens}: best={fit.max():.4f} features={int(binm[b].sum())} "
              f"evals={ev.n_evals}", flush=True)
    return best_mask, hist


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
        "model": "Plain GWO + RF", "dataset": args.name, "variant": args.variant, "data": args.data,
        "rows": len(df), "features": feature_cols, "dropped_constant_features": dropped,
        "malicious_profiles": malicious, "wolves": args.wolves, "generations": args.gens,
        "rho": RHO, "fitness_neg_cap": neg_cap, "final_rf_trees": final_trees,
        "max_hours": args.max_hours, "random_state": C.RANDOM_STATE, "smoke": args.smoke,
        "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID", "")})

    rows, history, selected_by_fold, importances, curves = [], [], {}, {}, {}

    def ckpt(fi):
        return os.path.join(ck_dir, f"fold_{fi}.pkl")

    def refresh():
        if not rows:
            return
        fig = os.path.join(run_dir, "figures")
        C.write_common_outputs(run_dir, label, rows, curves, "GWO+RF", "RF all features")
        h = pd.DataFrame(history)
        h.to_csv(os.path.join(run_dir, "gwo_history.csv"), index=False)
        C.plot_gwo_history(h, os.path.join(fig, "optimizer_convergence.png"), label)
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
        best_mask, fold_hist = run_gwo(ev, n_features, rng, args.gens, args.wolves,
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
