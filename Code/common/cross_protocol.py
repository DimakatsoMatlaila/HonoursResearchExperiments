#!/usr/bin/env python3
"""Protocol-switch experiments: train on one protocol, test on another (with and without a held-out profile)."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
import lpo_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out-dir", required=True)
ap.add_argument("--min-pos-test", type=int, default=20); ap.add_argument("--min-pos-train", type=int, default=50)
ap.add_argument("--neg-cap", type=int, default=1_000_000); ap.add_argument("--count-only", action="store_true")
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
os.makedirs(a.out_dir, exist_ok=True)
seed = C.RANDOM_STATE
rng = np.random.default_rng(seed)

df = pd.read_csv(a.data)
if a.smoke:
    df = df[(df["label"] == 1) | (rng.random(len(df)) < 0.03)]
df = df.reset_index(drop=True)
cat = np.where(df.is_tls == 1, "tls", np.where(df.is_quic == 1, "quic", np.where(df.is_http == 1, "http", "other")))
df["cat"] = cat
cnt = pd.crosstab(df.profile.where(df.label == 1, "normal"), df.cat)
print("flows per profile and protocol category\n", cnt.to_string(), flush=True)
cnt.to_csv(os.path.join(a.out_dir, "protocol_counts.csv"))
if a.count_only:
    sys.exit(0)

proto_cols = ["is_http", "is_tls", "is_quic", "tls_handshake"]
feats = [c for c in df.columns if c not in C.NON_FEATURE_COLS + ["cat"] + proto_cols]
df[feats] = df[feats].fillna(0)
X = df[feats].to_numpy(np.float32); y = df.label.to_numpy(); prof = df.profile.astype(str).to_numpy()
mal, test_norm, _ = C.build_folds_time(df)
is_tn = np.zeros(len(df), bool); is_tn[test_norm] = True
is_norm = (prof == "normal")
pos_cat = df[df.label == 1].cat.value_counts()
cats = [c for c in pos_cat.index if pos_cat[c] >= a.min_pos_train]
print("categories with enough C2 flows:", cats, flush=True)

rows = []
def run(S, T, q, mode):
    trm = (df.cat.to_numpy() == S) & ~is_tn
    if mode == "lpo":
        trm &= (prof != q)
    tem = ((prof == q) & (df.cat.to_numpy() == T)) | (is_tn & (df.cat.to_numpy() == T))
    npos_te = int(((prof == q) & (df.cat.to_numpy() == T)).sum())
    ntr = int((trm & (y == 1)).sum())
    if npos_te < a.min_pos_test or ntr < a.min_pos_train:
        return
    neg = np.flatnonzero(trm & (y == 0))
    if len(neg) > a.neg_cap:
        neg = rng.choice(neg, a.neg_cap, replace=False)
    tri = np.concatenate([np.flatnonzero(trm & (y == 1)), neg]); tei = np.flatnonzero(tem)
    clf = RandomForestClassifier(n_estimators=30 if a.smoke else 200, n_jobs=-1, class_weight="balanced", random_state=seed)
    clf.fit(X[tri], y[tri])
    s = clf.predict_proba(X[tei])[:, 1]; p = (s >= 0.5).astype(int)
    m = C.evaluate(y[tei], p, is_norm[tei], rng)
    m.update({"mode": mode, "train_proto": S, "test_proto": T, "held_out": q, "n_train_pos": ntr, "seed": seed})
    rows.append(m)
    print(f"{mode:5s} train {S:5s} -> test {T:5s} held {q:14s} F1={m['f1']:.4f} R={m['recall']:.4f} "
          f"P={m['precision']:.4f} FPR={m['fpr_normal']:.2e} (pos test {npos_te}, train pos {ntr})", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "cross_protocol_results.csv"), index=False)

for S in cats:
    for T in cats:
        for q in mal:
            run(S, T, q, "lpo")      # profile AND protocol unseen (S==T is the same-protocol control)
            if S != T:
                run(S, T, q, "proto")  # protocol unseen, profile seen
if rows:
    d = pd.DataFrame(rows)
    sm = d.groupby(["mode", "train_proto", "test_proto"])[["f1", "recall", "precision", "fpr_normal"]].mean().round(4)
    sm.to_csv(os.path.join(a.out_dir, "cross_protocol_summary.csv"))
    print("\nmean over held-out profiles\n", sm.to_string())
else:
    print("No protocol pair had enough C2 flows; see protocol_counts.csv")
