#!/usr/bin/env python3
"""Isolation Forest trained on benign flows only, leave-profiles-out."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

import lpo_common as C
import lpo_runner as R

YIL = ["mean_iat", "std_iat", "total_packets"]


def add_args(ap):
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--max-samples", type=int, default=4096)


def _fit_score(Xb, Xte, trees, max_samples):
    clf = IsolationForest(n_estimators=trees, max_samples=min(max_samples, len(Xb)),
                          contamination="auto", random_state=C.RANDOM_STATE, n_jobs=-1)
    clf.fit(Xb)
    pred = (clf.predict(Xte) == -1).astype(int)
    score = -clf.decision_function(Xte)
    return pred, score


def _iat_cv(X, cols):
    mi, si = cols.index("mean_iat"), cols.index("std_iat")
    mean = X[:, mi].astype(np.float64)
    return np.divide(X[:, si], mean, out=np.zeros(len(X)), where=mean != 0)


def fold_fn(ctx):
    trees = 30 if ctx.smoke else ctx.args.trees
    benign = ctx.y_tr == 0
    pred, score = _fit_score(ctx.X_tr[benign], ctx.X_te, trees, ctx.args.max_samples)
    extra = {}
    cols = ctx.feature_cols
    if all(c in cols for c in YIL):
        idx = [cols.index(c) for c in YIL]
        Xtr_y = np.column_stack([ctx.X_tr[:, idx], _iat_cv(ctx.X_tr, cols)])
        Xte_y = np.column_stack([ctx.X_te[:, idx], _iat_cv(ctx.X_te, cols)])
        yp, ys = _fit_score(Xtr_y[benign], Xte_y, trees, ctx.args.max_samples)
        tp = int(((ctx.y_te == 1) & (yp == 1)).sum())
        fp = int(((ctx.y_te == 0) & (yp == 1)).sum())
        fn = int(((ctx.y_te == 1) & (yp == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        extra = {"yil_precision": prec, "yil_recall": rec,
                 "yil_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                 "yil_roc_auc": float(roc_auc_score(ctx.y_te, ys))}
    return {"pred": pred, "score": score, "extra": extra}


if __name__ == "__main__":
    args = R.parse_args(__doc__, add_args)
    R.run_lpo("Isolation Forest (unsupervised, benign-only training)", "IsolationForest", fold_fn, args,
              config_extra={"trees": 30 if args.smoke else args.trees, "max_samples": args.max_samples,
                            "yilmaz_subset": YIL + ["iat_cv"]})
