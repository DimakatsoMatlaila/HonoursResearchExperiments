#!/usr/bin/env python3
"""Data loading, folds, metrics with bootstrap CIs and plots."""
import json
import math
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plot_style as PS
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

NON_FEATURE_COLS = ["profile", "capture_side", "proto", "ip_a", "port_a", "ip_b", "port_b",
                    "start_time", "http_method", "label"]
RANDOM_STATE = int(os.environ.get("C2_SEED", "42"))
NORMAL_TEST_FRACTION = 0.2
N_BOOTSTRAP = 2000


def load_data(path, variant="mixed", smoke=False):
    """Memory-lean loader: only the columns a model needs, float32 features.
    Returns (df, feature_cols, dropped_constant_cols). df has: profile, start_time, label + features."""
    head = pd.read_csv(path, nrows=0).columns.tolist()
    feats = [c for c in head if c not in NON_FEATURE_COLS]
    dtypes = {c: np.float32 for c in feats}
    dtypes.update({"label": np.int8, "profile": "category", "start_time": np.float64})
    df = pd.read_csv(path, usecols=["profile", "start_time", "label"] + feats, dtype=dtypes)
    if variant == "https":
        df = df[df["is_tls"] == 1]
    if smoke:
        rng = np.random.default_rng(RANDOM_STATE)
        df = df[(df["label"].to_numpy() == 1) | (rng.random(len(df)) < 0.03)]
    df = df.reset_index(drop=True)
    df[feats] = df[feats].fillna(0)
    dropped = [c for c in feats if df[c].nunique() <= 1]
    return df, [c for c in feats if c not in dropped], dropped


def build_folds_time(df):
    """Malicious profiles + held-out normal slice = LAST 20% of the normal capture in time
    (so a flow's internal/external twin never straddles the train/test boundary)."""
    malicious = sorted(df.loc[df["label"] == 1, "profile"].unique())
    idx = np.flatnonzero((df["profile"] == "normal").to_numpy())
    if len(idx) < 10:
        raise SystemExit("No (or too few) rows with profile == 'normal' in this dataset/variant.")
    if len(malicious) == 0:
        raise SystemExit("No malicious (label==1) rows in this dataset/variant.")
    t = df["start_time"].to_numpy()[idx]
    cut = np.quantile(t, 1 - NORMAL_TEST_FRACTION)
    test = idx[t >= cut]
    return malicious, test, len(idx) - len(test)


def split_fold(df, held_out, normal_test_idx):
    held = (df["profile"] == held_out).to_numpy()
    nt = np.zeros(len(df), dtype=bool)
    nt[normal_test_idx] = True
    return (~held & ~nt), (held | nt)


def evaluate(y_true, y_pred, normal_mask, rng, n_boot=N_BOOTSTRAP):
    """Confusion counts, P/R/F1, FPR, with 95% bootstrap CIs (multinomial over confusion cells)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    n = tp + fp + fn + tn
    n_norm = int(normal_mask.sum())
    fp_norm = int(((y_pred == 1) & normal_mask).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    out = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_test": n, "n_pos": tp + fn, "n_neg": fp + tn,
           "n_normal_test": n_norm, "precision": prec, "recall": rec, "f1": f1,
           "fpr_all": fp / (fp + tn) if (fp + tn) else float("nan"),
           "fpr_normal": fp_norm / n_norm if n_norm else float("nan")}
    p = np.array([tp, fp, fn, tn], dtype=float)
    d = rng.multinomial(n, p / p.sum(), size=n_boot)
    btp, bfp, bfn, btn = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    z = np.zeros(n_boot)
    bp = np.divide(btp, btp + bfp, out=z.copy(), where=(btp + bfp) > 0)
    br = np.divide(btp, btp + bfn, out=z.copy(), where=(btp + bfn) > 0)
    bf = np.divide(2 * bp * br, bp + br, out=z.copy(), where=(bp + br) > 0)
    bfpr = np.divide(bfp, bfp + btn, out=z.copy(), where=(bfp + btn) > 0)
    cis = {"precision": bp, "recall": br, "f1": bf, "fpr_all": bfpr}
    if n_norm:
        cis["fpr_normal"] = rng.binomial(n_norm, out["fpr_normal"], size=n_boot) / n_norm
    for k in ("precision", "recall", "f1", "fpr_all", "fpr_normal"):
        if k in cis:
            out[k + "_lo"], out[k + "_hi"] = np.percentile(cis[k], [2.5, 97.5])
        else:
            out[k + "_lo"] = out[k + "_hi"] = float("nan")
    return out


def roc_pr_curves(y_true, scores, max_points=3000):
    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)

    def thin(a, b):
        if len(a) <= max_points:
            return a, b
        i = np.linspace(0, len(a) - 1, max_points).astype(int)
        return a[i], b[i]

    fpr, tpr = thin(fpr, tpr)
    prec, rec = thin(prec, rec)
    return {"fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec,
            "roc_auc": float(roc_auc_score(y_true, scores)),
            "ap": float(average_precision_score(y_true, scores))}


def write_json(path, obj):
    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=default)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _err(df, m):
    v = df[m].to_numpy(dtype=float)
    lo, hi = df[m + "_lo"].to_numpy(dtype=float), df[m + "_hi"].to_numpy(dtype=float)
    return v, [np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)]


def plot_metric_bars(df, path, title):
    x, w = np.arange(len(df)), 0.25
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(max(10, 2.4 * len(df) + 4), 5),
                                 gridspec_kw={"width_ratios": [3, 1.3]})
    for i, m in enumerate(["f1", "precision", "recall"]):
        v, e = _err(df, m)
        a1.bar(x + (i - 1) * w, v, w, yerr=e, capsize=3, label=m)
    a1.set_xticks(x)
    a1.set_xticklabels(df["held_out"], rotation=20, ha="right")
    a1.set_ylim(0, 1.05)
    a1.set_title("Held-out profile performance (95% bootstrap CI)")
    a1.legend(loc="lower right")
    v, e = _err(df, "fpr_normal")
    a2.bar(x, v, 0.5, yerr=e, capsize=3, color=PS.PALETTE[3])
    a2.set_xticks(x)
    a2.set_xticklabels(df["held_out"], rotation=20, ha="right")
    a2.set_title("FPR vs normal baseline")
    fig.suptitle(title)
    _save(fig, path)


def plot_baseline_comparison(df, path, title, model_name, baseline_name):
    x, w = np.arange(len(df)), 0.2
    fig, ax = plt.subplots(figsize=(max(9, 2.4 * len(df) + 3), 5))
    for i, (col, lab) in enumerate([("f1", f"{model_name} F1"), ("base_f1", f"{baseline_name} F1"),
                                    ("recall", f"{model_name} recall"),
                                    ("base_recall", f"{baseline_name} recall")]):
        ax.bar(x + (i - 1.5) * w, df[col], w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(df["held_out"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"{title}\nmodel vs reference baseline")
    _save(fig, path)


def plot_confusion(df, path, title):
    n = len(df)
    ncols = min(n, 4)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), df.iterrows()):
        ax.axis("on")
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]], dtype=float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.grid(False)
        ax.imshow(norm, vmin=0, vmax=1, cmap=PS.CM_SEQ)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i, j]):,}", ha="center", va="center",
                        color="#FFFFFF" if norm[i, j] > 0.6 else PS.INK)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["pred benign", "pred C2"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["true benign", "true C2"])
        ax.set_title(r["held_out"])
    fig.suptitle(title + " (colour = row-normalised)")
    _save(fig, path)


def plot_roc_pr(curves, path, title):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
    for ho, c in curves.items():
        a1.plot(c["fpr"], c["tpr"], label=f"{ho} (AUC {c['roc_auc']:.4f})")
        a2.plot(c["rec"], c["prec"], label=f"{ho} (AP {c['ap']:.4f})")
    a1.set_xscale("log")
    a1.set_xlim(1e-6, 1)
    a1.set_ylim(0, 1.02)
    a1.set_xlabel("False positive rate (log)")
    a1.set_ylabel("True positive rate")
    a1.set_title("ROC")
    a1.legend(fontsize=8)
    a2.set_xlim(0, 1)
    a2.set_ylim(0, 1.02)
    a2.set_xlabel("Recall")
    a2.set_ylabel("Precision")
    a2.set_title("Precision-Recall")
    a2.legend(fontsize=8)
    fig.suptitle(title)
    _save(fig, path)


def plot_gwo_history(h, path, title):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for ho, g in h.groupby("held_out", sort=False):
        a1.plot(g["generation"], g["best_fitness"], marker="o", ms=3, label=ho)
        a2.plot(g["generation"], g["best_n_features"], marker="o", ms=3, label=ho)
    a1.set_xlabel("Generation")
    a1.set_ylabel("Best fitness (F1 - feature penalty)")
    a1.set_title("Optimizer convergence")
    a1.legend(fontsize=8)
    a2.set_xlabel("Generation")
    a2.set_ylabel("Features in best subset")
    a2.set_title("Subset size")
    fig.suptitle(title)
    _save(fig, path)


def plot_selection_heatmap(sel, feats, path, title):
    folds = list(sel)
    mat = np.array([[1 if f in sel[k] else 0 for f in feats] for k in folds])
    fig, ax = plt.subplots(figsize=(max(10, 0.45 * len(feats)), 1.2 + 0.7 * len(folds)))
    ax.grid(False)
    ax.imshow(mat, cmap=PS.CM_GREEN, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=75, ha="right", fontsize=8)
    ax.set_yticks(range(len(folds)))
    ax.set_yticklabels(folds)
    ax.set_title(title + ": features selected per fold (dark = selected)")
    _save(fig, path)


def plot_importance(imp, path, title):
    mean = imp.mean(axis=0, skipna=True).dropna().sort_values()
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(mean))))
    ax.barh(mean.index, mean.values)
    ax.set_xlabel("Mean RF importance (over folds that selected the feature)")
    ax.set_title(title)
    _save(fig, path)


def write_common_outputs(run_dir, label, rows, curves, model_name, baseline_name=None):
    """Per-run tables and figures. The model-vs-baseline plot is only made when baseline columns exist."""
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(run_dir, "metrics_per_fold.csv"), index=False)
    plot_metric_bars(df, os.path.join(fig_dir, "metrics_per_fold.png"), label)
    if baseline_name and "base_f1" in df.columns:
        plot_baseline_comparison(df, os.path.join(fig_dir, "model_vs_baseline.png"), label,
                                 model_name, baseline_name)
    plot_confusion(df, os.path.join(fig_dir, "confusion_matrices.png"), label)
    plot_roc_pr(curves, os.path.join(fig_dir, "roc_pr_curves.png"), label)
    return df
