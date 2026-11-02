#!/usr/bin/env python3
"""Histogram gradient boosting, leave-profiles-out."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import lpo_common as C
import lpo_runner as R


def add_args(ap):
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.1)


def fold_fn(ctx):
    iters = 30 if ctx.smoke else ctx.args.iters
    n_pos = max(int((ctx.y_tr == 1).sum()), 1)
    n_neg = max(int((ctx.y_tr == 0).sum()), 1)
    w = np.where(ctx.y_tr == 1, 0.5 * len(ctx.y_tr) / n_pos, 0.5 * len(ctx.y_tr) / n_neg)
    clf = HistGradientBoostingClassifier(max_iter=iters, learning_rate=ctx.args.lr, max_leaf_nodes=63,
                                         early_stopping=False, random_state=C.RANDOM_STATE)
    clf.fit(ctx.X_tr, ctx.y_tr, sample_weight=w)
    score = clf.predict_proba(ctx.X_te)[:, 1]
    return {"pred": (score >= 0.5).astype(int), "score": score}


if __name__ == "__main__":
    args = R.parse_args(__doc__, add_args)
    R.run_lpo("Gradient boosting (HistGB, all features)", "HistGB", fold_fn, args,
              config_extra={"max_iter": 30 if args.smoke else args.iters, "learning_rate": args.lr,
                            "class_weighting": "balanced sample weights"})
