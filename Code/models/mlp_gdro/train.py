#!/usr/bin/env python3
"""MLP with Group DRO over profile environments, leave-profiles-out."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))

import lpo_runner as R
from invariant_net import fit_predict

OBJECTIVE = "gdro"


def add_args(ap):
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=20.0)
    ap.add_argument("--eta", type=float, default=0.01)


def fold_fn(ctx):
    a = ctx.args
    steps = 40 if ctx.smoke else a.steps
    threads = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count() or 4))
    score, hist = fit_predict(ctx, OBJECTIVE, a.beta, steps, a.batch, a.lr, a.wd, a.eta, threads)
    return {"pred": (score >= 0.5).astype(int), "score": score, "history": hist}


if __name__ == "__main__":
    args = R.parse_args(__doc__, add_args)
    R.run_lpo("MLP with Group DRO over profile environments", "MLP-gdro", fold_fn, args,
              config_extra={"objective": OBJECTIVE, "steps": args.steps, "beta": args.beta,
                            "eta": args.eta, "environments": "one per malicious training profile"})
