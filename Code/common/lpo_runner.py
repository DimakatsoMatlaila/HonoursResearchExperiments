#!/usr/bin/env python3
"""Shared leave-profiles-out driver: folds, metrics, checkpoints and figures."""
import argparse
import os
import pickle
import platform
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

import lpo_common as C


def parse_args(description, add_args=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--data", required=True, help="internal/external/master dataset CSV")
    ap.add_argument("--run-dir", required=True, help="output folder for this run")
    ap.add_argument("--name", default="dataset", help="internal | external | master")
    ap.add_argument("--variant", default="mixed", choices=["mixed", "https"])
    ap.add_argument("--max-hours", type=float, default=66.0, help="wall-clock budget for the whole run")
    ap.add_argument("--smoke", action="store_true", help="tiny quick test (3%% of benign rows)")
    if add_args:
        add_args(ap)
    return ap.parse_args()


def _atomic_pickle(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


# fold_fn(ctx) returns {pred, score, optional selected / importance / extra / history}
def run_lpo(title, model_label, fold_fn, args, config_extra=None, history_plot=None):
    run_dir = args.run_dir
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)
    ck_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    label = os.path.basename(os.path.normpath(run_dir))
    t_start = time.time()

    df, feature_cols, dropped = C.load_data(args.data, args.variant, args.smoke)
    X_all = df[feature_cols].to_numpy(dtype=np.float32)
    y_all = df["label"].to_numpy()
    profiles = df["profile"].astype(str).to_numpy()
    malicious, normal_test_idx, n_normal_train = C.build_folds_time(df)
    zero = sorted(set(profiles) - set(malicious) - {"normal"})

    print(f"[{title} | {label}] {len(df):,} rows, {len(feature_cols)} features, "
          f"dataset={args.name}, variant={args.variant}", flush=True)
    print(f"Constant features dropped: {dropped}")
    print(f"Malicious profiles (one fold each, {len(malicious)}): {malicious}")
    if zero:
        print(f"*** WARNING: profiles with ZERO C2 flows in this variant (treated as benign): {zero} ***")
    print(f"Normal: {n_normal_train:,} train rows, {len(normal_test_idx):,} held-out (last 20% in time)",
          flush=True)

    cfg = {"model": title, "dataset": args.name, "variant": args.variant, "data": args.data,
           "rows": len(df), "features": feature_cols, "dropped_constant_features": dropped,
           "malicious_profiles": malicious, "profiles_without_c2": zero,
           "random_state": C.RANDOM_STATE, "smoke": args.smoke, "max_hours": args.max_hours,
           "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID", "")}
    cfg.update(config_extra or {})
    C.write_json(os.path.join(run_dir, "run_config.json"), cfg)

    rows, history, selected_by_fold, importances, curves = [], [], {}, {}, {}

    def ckpt(fi):
        return os.path.join(ck_dir, f"fold_{fi}.pkl")

    def refresh():
        if not rows:
            return
        fig = os.path.join(run_dir, "figures")
        C.write_common_outputs(run_dir, label, rows, curves, model_label)
        if history:
            h = pd.DataFrame(history)
            h.to_csv(os.path.join(run_dir, "history.csv"), index=False)
            if history_plot is not None:
                history_plot(h, os.path.join(fig, "training_history.png"), label)
        if selected_by_fold:
            C.plot_selection_heatmap(selected_by_fold, feature_cols,
                                     os.path.join(fig, "feature_selection_heatmap.png"), label)
        if importances:
            imp = pd.DataFrame(importances).T
            imp.to_csv(os.path.join(run_dir, "feature_importance.csv"))
            C.plot_importance(imp, os.path.join(fig, "feature_importance.png"), label)

    for fi, held_out in enumerate(malicious):
        print(f"\n=== Fold {fi + 1}/{len(malicious)}: holding out {held_out} ===", flush=True)
        if os.path.exists(ckpt(fi)):
            with open(ckpt(fi), "rb") as fh:
                d = pickle.load(fh)
            rows.append(d["row"])
            curves[held_out] = d["curve"]
            if d.get("selected") is not None:
                selected_by_fold[held_out] = d["selected"]
            if d.get("importance") is not None:
                importances[held_out] = d["importance"]
            history.extend(d.get("history") or [])
            print("  already done in a previous run; skipping", flush=True)
            continue

        tr_mask, te_mask = C.split_fold(df, held_out, normal_test_idx)
        X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
        X_te, y_te = X_all[te_mask], y_all[te_mask]
        normal_te = profiles[te_mask] == "normal"
        rng = np.random.default_rng(C.RANDOM_STATE + fi)

        n_left = sum(1 for j in range(fi, len(malicious)) if not os.path.exists(ckpt(j)))
        remaining = t_start + args.max_hours * 3600 - time.time()
        deadline = time.time() + 0.85 * remaining / max(n_left, 1)

        ctx = SimpleNamespace(X_tr=X_tr, y_tr=y_tr, X_te=X_te, y_te=y_te, feature_cols=feature_cols,
                              rng=rng, args=args, deadline=deadline, held_out=held_out,
                              fold_index=fi, smoke=args.smoke, prof_tr=profiles[tr_mask])
        t_fit = time.time()
        out = fold_fn(ctx)
        fit_min = (time.time() - t_fit) / 60

        pred = np.asarray(out["pred"]).astype(int)
        score = np.asarray(out["score"], dtype=float)
        m = C.evaluate(y_te, pred, normal_te, rng)
        curves[held_out] = C.roc_pr_curves(y_te, score)
        selected = out.get("selected")
        importance = out.get("importance")
        fold_hist = out.get("history") or []

        row = {"held_out": held_out, "n_train": int(tr_mask.sum()), "n_test": int(te_mask.sum()),
               "n_selected": len(selected) if selected is not None else len(feature_cols),
               "selected": ";".join(selected) if selected is not None else ";".join(feature_cols),
               "roc_auc": curves[held_out]["roc_auc"], "ap": curves[held_out]["ap"],
               "fit_minutes": fit_min}
        row.update(m)
        row.update(out.get("extra") or {})
        rows.append(row)
        if selected is not None:
            selected_by_fold[held_out] = selected
        if importance is not None:
            importances[held_out] = importance
        history.extend(fold_hist)

        print(f"  TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}  F1={m['f1']:.4f} "
              f"[{m['f1_lo']:.4f},{m['f1_hi']:.4f}]  P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"FPR(normal)={m['fpr_normal']:.5f}  AUC={row['roc_auc']:.4f}  ({fit_min:.1f} min)", flush=True)

        _atomic_pickle({"row": row, "curve": curves[held_out], "selected": selected,
                        "importance": importance, "history": fold_hist}, ckpt(fi))
        refresh()

    refresh()
    invariant = sorted(set.intersection(*[set(s) for s in selected_by_fold.values()])) \
        if selected_by_fold else []
    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write(f"{label}\nrows={len(df)} elapsed_h={(time.time() - t_start) / 3600:.2f}\n\n")
        f.write(pd.DataFrame(rows)[["held_out", "f1", "precision", "recall", "fpr_normal",
                                    "roc_auc", "n_selected"]].to_string(index=False))
        if selected_by_fold:
            f.write(f"\n\nFeatures selected in EVERY fold: {invariant}\n")
        else:
            f.write("\n\n(this model uses all features; no feature selection)\n")
    open(os.path.join(run_dir, "DONE"), "w").write("complete\n")
    print(f"\nInvariant features: {invariant}\nDone in {(time.time() - t_start) / 3600:.2f} h", flush=True)
