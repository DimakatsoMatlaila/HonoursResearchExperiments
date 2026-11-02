#!/usr/bin/env python3
"""A: how common are dukes_apt29-style check-in flows (11 packets, ~2.2 kB) in every profile and in benign traffic.
B: RF fold of dukes_apt29 with and without the first half of its own flows in training; scored on the second half."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
import lpo_common as C

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--out-dir", required=True)
ap.add_argument("--trees", type=int, default=300)
a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)

df, feats, _ = C.load_data(a.data, "mixed")
chk = ((df["total_packets"] == 11) & df["total_bytes"].between(2100, 2300)).to_numpy()   # check-in-like flow
rows = []
for p, g in df.groupby("profile", observed=True):
    ix = g.index.to_numpy()
    rows.append(dict(profile=p, flows=len(g), checkin_like=int(chk[ix].sum()), share=chk[ix].mean(),
                     median_bytes=g["total_bytes"].median(), median_packets=g["total_packets"].median(),
                     median_duration=g["duration"].median()))
g = df[df["profile"] == "dukes_apt29"].sort_values("start_time"); tm = (g["start_time"] - g["start_time"].min()) / 60
ck = chk[g.index.to_numpy()]
print("dukes_apt29 capture minutes: check-in-like flows first/last at", round(tm[ck].min(), 1), round(tm[ck].max(), 1),
      "| other flows first/last at", round(tm[~ck].min(), 1), round(tm[~ck].max(), 1), "| capture length", round(tm.max(), 1))
A = pd.DataFrame(rows); A.to_csv(f"{a.out_dir}/checkin_by_profile.csv", index=False)
print(A.to_string(index=False))

malicious, test_idx, _ = C.build_folds_time(df)
held = (df["profile"] == "dukes_apt29").to_numpy()
nt = np.zeros(len(df), bool); nt[test_idx] = True
d_idx = np.flatnonzero(held)
d_idx = d_idx[np.argsort(df["start_time"].to_numpy()[d_idx])]
chunk = np.arange(len(d_idx)) * 20 // len(d_idx)      # 20 consecutive time blocks; even blocks -> training, odd -> test
first, second = d_idx[chunk % 2 == 0], d_idx[chunk % 2 == 1]
X = df[feats].to_numpy(np.float32); y = df["label"].to_numpy()
res = []
for name, extra in (("without_dukes_in_training", np.array([], int)), ("with_alternate_blocks_of_dukes", first)):
    tr = (~held & ~nt); tr[extra] = True
    te = nt.copy(); te[second] = True
    clf = RandomForestClassifier(n_estimators=a.trees, n_jobs=-1, random_state=C.RANDOM_STATE, class_weight="balanced")
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te]); yt = y[te]; ck = chk[te]
    pos = yt == 1; fp = int(((yt == 0) & (pred == 1)).sum()); nneg = int((yt == 0).sum())
    res.append(dict(setting=name, c2_test=int(pos.sum()), recall_all=float(pred[pos].mean()),
                    checkin_test=int((pos & ck).sum()), recall_checkin=float(pred[pos & ck].mean()) if (pos & ck).any() else np.nan,
                    recall_other=float(pred[pos & ~ck].mean()) if (pos & ~ck).any() else np.nan, fp=fp, fpr=fp / nneg))
B = pd.DataFrame(res); B.to_csv(f"{a.out_dir}/dukes_train_ablation.csv", index=False)
print(B.to_string(index=False))
