#!/usr/bin/env python3
"""Random Forest on all features, leave-profiles-out."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import lpo_common as C
import lpo_runner as R


def add_args(ap):
    ap.add_argument("--trees", type=int, default=300)


def fold_fn(ctx):
    trees = 50 if ctx.smoke else ctx.args.trees
    clf = RandomForestClassifier(n_estimators=trees, n_jobs=-1, random_state=C.RANDOM_STATE,
                                 class_weight="balanced")
    clf.fit(ctx.X_tr, ctx.y_tr)
    return {"pred": clf.predict(ctx.X_te), "score": clf.predict_proba(ctx.X_te)[:, 1],
            "importance": pd.Series(clf.feature_importances_, index=ctx.feature_cols)}


if __name__ == "__main__":
    args = R.parse_args(__doc__, add_args)
    R.run_lpo("RF baseline (all features)", "RF all features", fold_fn, args,
              config_extra={"trees": 50 if args.smoke else args.trees, "class_weight": "balanced"})
