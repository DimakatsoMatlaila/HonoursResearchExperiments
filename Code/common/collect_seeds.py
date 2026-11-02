#!/usr/bin/env python3
"""Mean and spread across seeds.  python collect_seeds.py PIPE [OUT]"""
import glob, json, os, sys
import pandas as pd

pipe = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(pipe, "results_seeds")
os.makedirs(out, exist_ok=True)
fr = []
for f in glob.glob(os.path.join(pipe, "models", "*", "results*", "*", "metrics_per_fold.csv")):
    d = os.path.dirname(f)
    p = d.split(os.sep)
    if "smoke" in p[-2] or not os.path.exists(os.path.join(d, "DONE")):
        continue
    seed = json.load(open(os.path.join(d, "run_config.json"))).get("random_state", 42)
    x = pd.read_csv(f)
    name, _, var = p[-1].partition("_")
    x.insert(0, "model", p[-3]); x.insert(1, "dataset", name); x.insert(2, "variant", var); x.insert(3, "seed", seed)
    fr.append(x)
if not fr:
    sys.exit("No finished runs found.")
a = pd.concat(fr, ignore_index=True)
a.to_csv(os.path.join(out, "all_folds_all_seeds.csv"), index=False)
run = a.groupby(["model", "dataset", "variant", "seed"])[["f1", "precision", "recall", "fpr_normal"]].mean().reset_index()
run.to_csv(os.path.join(out, "per_run_per_seed.csv"), index=False)
g = run.groupby(["model", "dataset", "variant"])
s = g["f1"].agg(["mean", "std", "min", "max", "count"]).rename(columns={"count": "n_seeds"}).round(4)
s.to_csv(os.path.join(out, "f1_mean_sd_over_seeds.csv"))
print("F1 per model/dataset/variant, mean and sd over seeds\n", s.to_string())
m = run.groupby(["model", "seed"])[["f1", "recall"]].mean().reset_index()
t = m.groupby("model")[["f1", "recall"]].agg(["mean", "std"]).round(4)
t["n_seeds"] = m.groupby("model").size()
t.to_csv(os.path.join(out, "model_overall_mean_sd_over_seeds.csv"))
print("\nOverall (mean over the 6 configurations, then over seeds)\n", t.to_string())
pp = a.groupby(["model", "held_out", "seed"])[["f1", "recall"]].mean().reset_index()
pq = pp.groupby(["model", "held_out"])[["f1", "recall"]].agg(["mean", "std"]).round(4)
pq.to_csv(os.path.join(out, "per_profile_mean_sd_over_seeds.csv"))
print("\ndukes_apt29 across seeds\n", pq.xs("dukes_apt29", level="held_out").to_string())
print("\nwritten to", out)
