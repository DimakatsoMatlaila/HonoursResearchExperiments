#!/usr/bin/env python3
"""Show the most common IPs in a profile to help find its C2 server."""
import glob
import os
import sys
from collections import Counter

import pandas as pd


def count_ips(files):
    c = Counter()
    for f in files:
        for ch in pd.read_csv(f, usecols=["ip_a", "ip_b"], chunksize=500_000):
            c.update(ch["ip_a"]); c.update(ch["ip_b"])
    return c


def main():
    labeled, prof = sys.argv[1:3]
    pf = glob.glob(f"{labeled}/{prof}/*.csv")
    nf = glob.glob(f"{labeled}/normal/*.csv")
    if not pf:
        sys.exit(f"no labeled CSVs for {prof}")
    cp = count_ips(pf)
    cn = count_ips(nf) if nf else Counter()
    print(f"Top endpoint IPs in '{prof}' (flows), and flows with the same IP in 'normal':")
    for ip, n in cp.most_common(15):
        print(f"  {ip:<18} {n:>12,}   normal: {cn.get(ip, 0):>10,}")
    print("\nThe C2 server is usually the IP that is very frequent here and (near) absent from normal.")


if __name__ == "__main__":
    main()
