#!/usr/bin/env python3
"""LSH + Random Forest, leave-profiles-out."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import lpo_common as C
import lpo_runner as R

ALPHA = 1.0


def add_args(ap):
    ap.add_argument("--tables", type=int, default=8)
    ap.add_argument("--bits", type=int, default=16)
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--oof-folds", type=int, default=5)


def hash_ids(Z, planes):
    bits = (Z @ planes.T >= 0).astype(np.int64)
    return bits @ (1 << np.arange(bits.shape[1], dtype=np.int64))


# training rows get out-of-fold bucket ratios so a row's own label never leaks into its feature
def lsh_features(Ztr, ytr, Zte, n_tables, n_bits, n_oof, rng):
    n_buckets = 1 << n_bits
    base = float(ytr.mean())
    ftr = np.zeros((len(Ztr), n_tables), dtype=np.float32)
    fte = np.zeros((len(Zte), n_tables), dtype=np.float32)
    folds = rng.integers(0, n_oof, size=len(Ztr))
    for t in range(n_tables):
        planes = rng.normal(size=(n_bits, Ztr.shape[1]))
        btr, bte = hash_ids(Ztr, planes), hash_ids(Zte, planes)
        tot = np.bincount(btr, minlength=n_buckets).astype(np.float64)
        pos = np.bincount(btr, weights=ytr.astype(np.float64), minlength=n_buckets)
        ratio = (pos + ALPHA) / (tot + 2 * ALPHA)
        fte[:, t] = np.where(tot[bte] > 0, ratio[bte], base)
        for k in range(n_oof):
            m = folds == k
            tot_k = np.bincount(btr[m], minlength=n_buckets).astype(np.float64)
            pos_k = np.bincount(btr[m], weights=ytr[m].astype(np.float64), minlength=n_buckets)
            r_k = (pos - pos_k + ALPHA) / (tot - tot_k + 2 * ALPHA)
            ftr[m, t] = r_k[btr[m]]
    return ftr, fte


def fold_fn(ctx):
    a = ctx.args
    mean = ctx.X_tr.mean(axis=0, dtype=np.float64)
    std = ctx.X_tr.std(axis=0, dtype=np.float64)
    std[std == 0] = 1.0
    Ztr = ((ctx.X_tr - mean) / std).astype(np.float32)
    Zte = ((ctx.X_te - mean) / std).astype(np.float32)
    ftr, fte = lsh_features(Ztr, ctx.y_tr, Zte, a.tables, a.bits, a.oof_folds, ctx.rng)
    clf = RandomForestClassifier(n_estimators=50 if ctx.smoke else a.trees, n_jobs=-1,
                                 random_state=C.RANDOM_STATE, class_weight="balanced")
    clf.fit(ftr, ctx.y_tr)
    return {"pred": clf.predict(fte), "score": clf.predict_proba(fte)[:, 1]}


if __name__ == "__main__":
    args = R.parse_args(__doc__, add_args)
    R.run_lpo("LSH + Random Forest", "LSH+RF", fold_fn, args,
              config_extra={"tables": args.tables, "bits": args.bits, "trees": args.trees,
                            "oof_folds": args.oof_folds, "laplace_alpha": ALPHA})
