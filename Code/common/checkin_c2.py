#!/usr/bin/env python3
"""C2-only check: how many check-in-like flows (11 packets, 2100-2300 bytes) each profile has, and when dukes_apt29 has them."""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import lpo_common as C
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out-dir", required=True)
a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
df, feats, _ = C.load_data(a.data, "mixed")
chk = ((df["total_packets"] == 11) & df["total_bytes"].between(2100, 2300)).to_numpy()
rows = []
for lab, name in ((1, "c2"), (0, "benign")):
    for p, g in df[df["label"] == lab].groupby("profile", observed=True):
        ix = g.index.to_numpy()
        rows.append(dict(kind=name, profile=p, flows=len(g), checkin_like=int(chk[ix].sum()), share=chk[ix].mean(),
                         median_bytes=g["total_bytes"].median(), median_packets=g["total_packets"].median()))
T = pd.DataFrame(rows); T.to_csv(f"{a.out_dir}/checkin_c2_vs_benign.csv", index=False); print(T.to_string(index=False))
g = df[(df["profile"] == "dukes_apt29") & (df["label"] == 1)].sort_values("start_time")
tm = (g["start_time"] - g["start_time"].min()) / 60; ck = chk[g.index.to_numpy()]
print("dukes C2 flows:", len(g), "check-in-like:", int(ck.sum()), "capture minutes:", round(tm.max(), 1))
print("check-in-like first/last at", round(tm[ck].min(), 1), round(tm[ck].max(), 1), "| other C2 first/last at", round(tm[~ck].min(), 1), round(tm[~ck].max(), 1))
print("C2 flows per 50-minute block, check-in-like vs other:")
for lo in range(0, int(tm.max()) + 50, 50):
    m = (tm >= lo) & (tm < lo + 50); print(lo, int((m & ck).sum()), int((m & ~ck).sum()))
