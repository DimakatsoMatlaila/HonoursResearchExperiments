#!/usr/bin/env python3
"""XGBoost, leave-profiles-out."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))

import numpy as np
from xgboost import XGBClassifier

import lpo_common as C
import lpo_runner as R


def add_args(ap):
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.1)


def fold_fn(ctx):
    trees = 30 if ctx.smoke else ctx.args.trees
    n_pos = max(int((ctx.y_tr == 1).sum()), 1)
    n_neg = max(int((ctx.y_tr == 0).sum()), 1)
    clf = XGBClassifier(n_estimators=trees, max_depth=ctx.args.depth, learning_rate=ctx.args.lr,
                        tree_method="hist", subsample=0.8, colsample_bytree=0.8,
                        scale_pos_weight=n_neg / n_pos, n_jobs=-1, random_state=C.RANDOM_STATE)
    clf.fit(ctx.X_tr, ctx.y_tr)
    score = clf.predict_proba(ctx.X_te)[:, 1]
    return {"pred": (score >= 0.5).astype(int), "score": score}


if __name__ == "__main__":
    args = R.parse_args(__doc__, add_args)
    R.run_lpo("XGBoost (all features)", "XGBoost", fold_fn, args,
              config_extra={"n_estimators": 30 if args.smoke else args.trees, "max_depth": args.depth,
                            "learning_rate": args.lr, "class_weighting": "scale_pos_weight = n_neg/n_pos"})
