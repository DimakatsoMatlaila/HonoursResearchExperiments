#!/usr/bin/env python3
"""Make a small fake pcap tree for testing the pipeline."""
import os
import random
import socket
import sys

import dpkt

C2 = "13.246.195.48"
TREE = {"entropic": ["jquery", "mscrl"], "mimicry": ["amazon", "slack"], "stealth": ["havex"], "": ["normal"]}


def pkt(src, sport, dst, dport, payload, ts, flags=dpkt.tcp.TH_ACK | dpkt.tcp.TH_PUSH, udp=False):
    if udp:
        l4 = dpkt.udp.UDP(sport=sport, dport=dport, data=payload)
        l4.ulen = len(l4)
        proto = dpkt.ip.IP_PROTO_UDP
    else:
        l4 = dpkt.tcp.TCP(sport=sport, dport=dport, flags=flags, data=payload)
        proto = dpkt.ip.IP_PROTO_TCP
    ip = dpkt.ip.IP(src=socket.inet_aton(src), dst=socket.inet_aton(dst), p=proto, data=l4)
    ip.len = len(ip)
    eth = dpkt.ethernet.Ethernet(src=b"\x00" * 6, dst=b"\x01" * 6, type=dpkt.ethernet.ETH_TYPE_IP, data=ip)
    return ts, bytes(eth)


def tls(n):
    return b"\x17\x03\x03" + n.to_bytes(2, "big") + os.urandom(max(n - 5, 0))


def make_flow(rng, t0, client, server, malicious, prof_seed):
    out, sport = [], rng.randint(20000, 60000)
    if malicious:
        n, gap, sz = rng.randint(12, 30), 5 + prof_seed + rng.random() * 0.5, 120 + prof_seed * 15
        sizes = lambda: int(rng.gauss(sz, 12))
    else:
        n, gap = rng.randint(4, 60), rng.random() * 2.0
        sizes = lambda: rng.choice([60, 300, 900, 1400, 1448])
    t = t0
    kind = rng.random()
    for i in range(n):
        if kind < 0.25:
            first = b"GET /index HTTP/1.1\r\nHost: x\r\n\r\n" if i == 0 else tls(60)
            payload = first if i % 2 == 0 else b"HTTP/1.1 200 OK\r\n\r\n" + os.urandom(max(sizes(), 20))
        else:
            payload = tls(max(sizes(), 40))
        a, b = (client, server) if i % 2 == 0 else (server, client)
        sp, dp = (sport, 443) if i % 2 == 0 else (443, sport)
        out.append(pkt(a, sp, b, dp, payload, t))
        t += gap * (0.8 + 0.4 * rng.random()) / 4
    return out


def write(path, prof, side, rng, nflows, t_base, seed):
    pk = []
    for i in range(nflows):
        mal = prof != "normal" and rng.random() < 0.5
        server = C2 if mal else f"93.184.{rng.randint(0, 250)}.{rng.randint(1, 250)}"
        pk += make_flow(rng, t_base + i * 3.0 + rng.random(), f"10.0.{seed % 200}.{rng.randint(2, 250)}", server, mal, seed % 7)
    pk.append(pkt("10.0.0.2", 5353, "8.8.8.8", 53, b"\x12\x34\x01\x00", t_base, udp=True))
    pk.sort(key=lambda x: x[0])
    with open(path, "wb") as f:
        w = dpkt.pcap.Writer(f)
        for ts, b in pk:
            w.writepkt(b, ts=ts)


def main():
    root = sys.argv[1]
    rng = random.Random(7)
    seed = 0
    for cat, profs in TREE.items():
        for p in profs:
            d = os.path.join(root, cat, p) if cat else os.path.join(root, p)
            os.makedirs(d, exist_ok=True)
            seed += 1
            nfl = 700 if p == "normal" else 160
            for side in ("internal", "external"):
                name = f"{side}_{p}_2025-01-0{seed % 9 + 1}.pcap" if seed % 2 else f"{side}_{p}.pcap"
                write(os.path.join(d, name), p, side, rng, nfl, 1_700_000_000 + seed * 100000, seed)
    open(os.path.join(root, "mimicry", "nohup.out"), "w").write("stray\n")
    print("synthetic pcaps written to", root)


if __name__ == "__main__":
    main()
