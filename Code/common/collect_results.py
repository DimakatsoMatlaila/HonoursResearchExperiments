#!/usr/bin/env python3
"""Combine every model's results into comparison tables and figures."""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_style

METRICS = ["f1", "precision", "recall", "fpr_normal", "roc_auc", "ap"]


def main():
    pipe = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(pipe, "results_summary")
    os.makedirs(out, exist_ok=True)
    frames = []
    for f in sorted(glob.glob(os.path.join(pipe, "models", "*", "results*", "*", "metrics_per_fold.csv"))):
        if "results_smoke" in f:
            continue
        run = os.path.basename(os.path.dirname(f))
        model = f.split(os.sep)[-4] + (" (serial)" if "results_serial" in f else "")
        name, _, variant = run.partition("_")
        d = pd.read_csv(f)
        d.insert(0, "model", model); d.insert(1, "dataset", name); d.insert(2, "variant", variant)
        d["complete"] = os.path.exists(os.path.join(os.path.dirname(f), "DONE"))
        frames.append(d)
    if not frames:
        sys.exit("No results found yet.")
    allr = pd.concat(frames, ignore_index=True)
    allr.to_csv(os.path.join(out, "all_folds_long.csv"), index=False)
    cols = [m for m in METRICS if m in allr.columns]
    g = allr.groupby(["model", "dataset", "variant"])
    summ = g[cols].mean().round(4)
    summ["folds"] = g.size()
    summ["complete"] = g["complete"].all()
    summ = summ.reset_index()
    summ.to_csv(os.path.join(out, "summary_mean_over_folds.csv"), index=False)
    print(summ.to_string(index=False))

    for variant in sorted(allr["variant"].unique()):
        sub = summ[summ["variant"] == variant]
        if sub.empty:
            continue
        piv = sub.pivot(index="model", columns="dataset", values="f1")
        ax = piv.plot(kind="bar", figsize=(9, 5))
        ax.set_ylabel("Mean F1 over held-out profiles"); ax.set_ylim(0, 1.02)
        ax.set_title(f"Model comparison ({variant})"); plt.xticks(rotation=30, ha="right")
        plt.tight_layout(); plt.savefig(os.path.join(out, f"compare_f1_{variant}.png"), dpi=200); plt.close()
        for ds in sorted(allr["dataset"].unique()):
            s = allr[(allr["variant"] == variant) & (allr["dataset"] == ds)]
            if s.empty:
                continue
            p = s.pivot_table(index="held_out", columns="model", values="f1")
            ax = p.plot(kind="bar", figsize=(11, 5))
            ax.set_ylabel("F1"); ax.set_ylim(0, 1.02); ax.set_title(f"F1 per held-out profile - {ds}, {variant}")
            plt.xticks(rotation=30, ha="right"); plt.tight_layout()
            plt.savefig(os.path.join(out, f"per_profile_f1_{ds}_{variant}.png"), dpi=200); plt.close()
    print(f"\nWrote tables and figures to {out}")


if __name__ == "__main__":
    main()
