#!/usr/bin/env python3
"""Re-run one held-out fold, save the missed flows and describe them against benign traffic."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
import lpo_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out-dir", required=True)
ap.add_argument("--variant", default="mixed", choices=["mixed", "https"])
ap.add_argument("--held-out", default="dukes_apt29")
ap.add_argument("--benign-sample", type=int, default=50000)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
seed = C.RANDOM_STATE
rng = np.random.default_rng(seed)

df = pd.read_csv(a.data)
if a.variant == "https":
    df = df[df["is_tls"] == 1]
if a.smoke:
    df = df[(df["label"] == 1) | (rng.random(len(df)) < 0.03)]
df = df.reset_index(drop=True)
meta = [c for c in C.NON_FEATURE_COLS if c in df.columns and c not in ("profile", "label", "start_time")]
feats = [c for c in df.columns if c not in C.NON_FEATURE_COLS]
df[feats] = df[feats].fillna(0)
feats = [c for c in feats if df[c].nunique() > 1]
mal, test_norm, _ = C.build_folds_time(df)
tr, te = C.split_fold(df, a.held_out, test_norm)
X = df[feats].to_numpy(np.float32)
y = df["label"].to_numpy()
clf = RandomForestClassifier(n_estimators=30 if a.smoke else 300, n_jobs=-1, class_weight="balanced", random_state=seed)
clf.fit(X[tr], y[tr])
idx_te = np.flatnonzero(te)
score = clf.predict_proba(X[idx_te])[:, 1]
pred = (score >= 0.5).astype(int)
is_c2 = y[idx_te] == 1
fn = is_c2 & (pred == 0)
tp = is_c2 & (pred == 1)
norm_idx = idx_te[~is_c2]
print(f"[{a.variant} seed {seed}] {a.held_out}: {is_c2.sum()} C2 flows, caught {tp.sum()}, missed {fn.sum()}, "
      f"recall {tp.sum() / max(is_c2.sum(), 1):.4f}; benign test {len(norm_idx):,}", flush=True)

held = df.iloc[idx_te[is_c2]][["profile", "start_time"] + meta + feats].copy()
held["score"] = score[is_c2]; held["caught"] = (pred[is_c2] == 1).astype(int)
held.to_csv(os.path.join(a.out_dir, "heldout_c2_flows.csv.gz"), index=False)
bs = rng.choice(norm_idx, min(a.benign_sample, len(norm_idx)), replace=False)
ben = df.iloc[bs][["profile", "start_time"] + meta + feats].copy()
ben["score"] = clf.predict_proba(X[bs])[:, 1]
ben.to_csv(os.path.join(a.out_dir, "benign_test_sample.csv.gz"), index=False)

miss, caught = held[held.caught == 0], held[held.caught == 1]
rep = [f"{a.held_out}, {a.variant}, seed {seed}: C2 {len(held)}, caught {len(caught)}, missed {len(miss)} "
       f"(recall {len(caught) / max(len(held), 1):.4f}); score of missed flows: median {miss.score.median():.3f}, "
       f"max {miss.score.max():.3f}" if len(miss) else "no missed flows"]
if len(miss) >= 5:
    rows = []
    for f in feats:
        b = ben[f].to_numpy()
        lo, hi = np.percentile(b, [5, 95])
        q = lambda s: np.percentile(s, [25, 50, 75])
        rows.append({"feature": f,
                     "ks_missed_vs_benign": ks_2samp(miss[f], b).statistic,
                     "ks_caught_vs_benign": ks_2samp(caught[f], b).statistic if len(caught) else np.nan,
                     "missed_in_benign_5_95": float(((miss[f] >= lo) & (miss[f] <= hi)).mean()),
                     "caught_in_benign_5_95": float(((caught[f] >= lo) & (caught[f] <= hi)).mean()) if len(caught) else np.nan,
                     "missed_q25": q(miss[f])[0], "missed_median": q(miss[f])[1], "missed_q75": q(miss[f])[2],
                     "caught_median": np.median(caught[f]) if len(caught) else np.nan,
                     "benign_q25": q(b)[0], "benign_median": q(b)[1], "benign_q75": q(b)[2]})
    fo = pd.DataFrame(rows).sort_values("ks_missed_vs_benign")
    fo.to_csv(os.path.join(a.out_dir, "feature_overlap.csv"), index=False)
    rep.append("\nFeatures where the MISSED flows look most like benign (small KS = same distribution):")
    rep.append(fo.head(8)[["feature", "ks_missed_vs_benign", "ks_caught_vs_benign", "missed_in_benign_5_95"]].round(3).to_string(index=False))
    top = fo.sort_values("ks_caught_vs_benign", ascending=False).feature.head(5).tolist()
    inside = np.ones(len(miss), bool)
    for f in top:
        lo, hi = np.percentile(ben[f], [5, 95]); inside &= ((miss[f] >= lo) & (miss[f] <= hi)).to_numpy()
    rep.append(f"\nFraction of missed flows inside the benign 5-95 percentile band on ALL of the 5 most "
               f"discriminative features {top}: {inside.mean():.3f}")
    med = ben[top].median(); iqr = (ben[top].quantile(.75) - ben[top].quantile(.25)).replace(0, 1)
    z = lambda d: ((d[top] - med) / iqr).to_numpy()
    pool = np.vstack([z(ben), z(caught)]) if len(caught) else z(ben)
    lab = np.r_[np.zeros(len(ben)), np.ones(len(caught))] if len(caught) else np.zeros(len(ben))
    nn = NearestNeighbors(n_neighbors=11).fit(pool)
    _, ix = nn.kneighbors(z(miss), n_neighbors=10)
    rep.append(f"kNN (k=10, robust-scaled top-5 features): share of a missed flow's neighbours that are benign = "
               f"{1 - lab[ix].mean():.3f}")
    if len(caught) >= 5:
        _, ix = nn.kneighbors(z(caught.sample(min(len(caught), 2000), random_state=seed)), n_neighbors=11)
        rep.append(f"same measure for CAUGHT flows (self excluded) = {1 - lab[ix[:, 1:]].mean():.3f}")
    # metadata and timing ("low and slow")
    for c in [c for c in ("capture_side", "proto", "port_b", "http_method") if c in held.columns]:
        rep.append(f"\n{c}: missed vs caught counts\n" + pd.concat([miss[c].value_counts().rename("missed"),
                   caught[c].value_counts().rename("caught")], axis=1).fillna(0).astype(int).head(8).to_string())
    if {"ip_a", "ip_b"}.issubset(held.columns):
        h = held.sort_values("start_time")
        gap = h.groupby(["ip_a", "ip_b"])["start_time"].diff()
        h = h.assign(gap_s=gap)
        rep.append("\nSeconds since the previous flow of the same host pair (beacon spacing):\n" +
                   h.groupby("caught")["gap_s"].describe(percentiles=[.25, .5, .75]).round(2).to_string())
open(os.path.join(a.out_dir, "report.txt"), "w").write("\n".join(rep) + "\n")
print("\n".join(rep))
