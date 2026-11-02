import numpy as np
from sklearn.ensemble import RandomForestClassifier

import os
SEED = int(os.environ.get("C2_SEED", "42"))
RHO = 0.01


class NestedFitness:
    """Fitness for feature search that holds profiles out inside the training set: for each of k inner
    profiles, fit on the other training profiles and score F1 on the inner profile plus unseen benign rows."""

    def __init__(self, X, y, prof, rng, neg_cap, k_inner=3, val_neg_cap=100_000):
        mal = sorted(set(prof[y == 1]))
        inner = list(rng.choice(mal, min(k_inner, len(mal)), replace=False))
        neg = np.flatnonzero(y == 0)
        rng.shuffle(neg)
        cut = max(1, len(neg) // 5)
        val_neg, fit_neg = neg[:cut][:val_neg_cap], neg[cut:][:neg_cap]
        pos = np.flatnonzero(y == 1)
        self.inner = inner
        self.folds = []
        for p in inner:
            tr_pos = pos[prof[pos] != p]
            va_pos = pos[prof[pos] == p]
            tr = np.concatenate([tr_pos, fit_neg])
            va = np.concatenate([va_pos, val_neg])
            self.folds.append((X[tr], y[tr], X[va], y[va]))
        self.cache, self.n_evals = {}, 0

    def __call__(self, mask):
        if mask.sum() == 0:
            return -1.0
        key = mask.tobytes()
        if key in self.cache:
            return self.cache[key]
        cols = np.flatnonzero(mask)
        f1s = []
        for Xf, yf, Xv, yv in self.folds:
            clf = RandomForestClassifier(n_estimators=40, max_depth=16, min_samples_leaf=2, n_jobs=-1,
                                         random_state=SEED, class_weight="balanced")
            clf.fit(Xf[:, cols], yf)
            p = clf.predict(Xv[:, cols])
            tp = int(((yv == 1) & (p == 1)).sum())
            fp = int(((yv == 0) & (p == 1)).sum())
            fn = int(((yv == 1) & (p == 0)).sum())
            f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
        val = float(np.mean(f1s)) - RHO * mask.sum() / len(mask)
        self.cache[key] = val
        self.n_evals += 1
        return val
