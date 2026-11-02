#!/usr/bin/env python3
"""Merge labeled captures into the internal, external and master datasets."""
import glob
import os
import sys

import pandas as pd

CHUNK = 500_000


def main():
    labeled, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    files = [f for f in sorted(glob.glob(f"{labeled}/**/*.csv", recursive=True))
             if not os.path.basename(f).startswith(".tmp_")]
    if not files:
        sys.exit(f"No labeled CSVs under {labeled}")
    names = {"internal": "internal_dataset.csv", "external": "external_dataset.csv", "master": "master_dataset.csv"}
    tmp = {k: os.path.join(out, v + ".tmp") for k, v in names.items()}
    for p in tmp.values():
        if os.path.exists(p):
            os.remove(p)
    started = {k: False for k in names}
    cols_ref = None
    stats = {}

    def emit(key, d):
        d.to_csv(tmp[key], mode="a", header=not started[key], index=False)
        started[key] = True

    for f in files:
        n = c2 = 0
        for chunk in pd.read_csv(f, chunksize=CHUNK):
            if cols_ref is None:
                cols_ref = list(chunk.columns)
            elif list(chunk.columns) != cols_ref:
                chunk = chunk.reindex(columns=cols_ref)
            emit("master", chunk)
            for side in ("internal", "external"):
                sub = chunk[chunk["capture_side"] == side]
                if len(sub):
                    emit(side, sub)
            g = chunk.groupby(["capture_side", "profile"])["label"].agg(["count", "sum"])
            for (side, prof), r in g.iterrows():
                s = stats.setdefault((side, prof), [0, 0])
                s[0] += int(r["count"]); s[1] += int(r["sum"])
            n += len(chunk); c2 += int(chunk["label"].sum())
        print(f"  {os.path.relpath(f, labeled)}: {n:,} rows, {c2:,} C2", flush=True)

    summ = pd.DataFrame([(s, p, v[0], v[1]) for (s, p), v in sorted(stats.items())],
                        columns=["capture_side", "profile", "flows", "c2_flows"])
    print("\nFlows / C2 flows per side and profile:\n" + summ.to_string(index=False))
    summ.to_csv(os.path.join(out, "dataset_summary.csv"), index=False)
    by_prof = summ.groupby("profile")[["flows", "c2_flows"]].sum()
    for prof, r in by_prof.iterrows():
        if prof != "normal" and r["c2_flows"] == 0:
            print(f"\n*** WARNING: profile '{prof}' has ZERO C2 flows. It probably used a different C2 IP. "
                  f"Run data_pipeline/diagnose.sh {prof}, fix c2_ips.txt, delete its labeled CSVs and re-run labelling. ***")
    if "normal" in by_prof.index and by_prof.loc["normal", "c2_flows"] > 0:
        print("\n*** WARNING: 'normal' contains C2-labeled flows. ***")
    for k, v in names.items():
        if started[k]:
            os.replace(tmp[k], os.path.join(out, v))
            print(f"Wrote {os.path.join(out, v)}")
        else:
            print(f"*** {v}: no rows (no captures with that side) ***")


if __name__ == "__main__":
    main()
