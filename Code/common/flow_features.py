#!/usr/bin/env python3
"""Turn a cleaned pcap into per-flow features and C2 labels."""
import csv
import math
import os
import socket
import sys

import dpkt

DEFAULT_C2_IP = "13.246.195.48"

HTTP_METHODS = (b"GET ", b"POST", b"PUT ", b"HEAD", b"DELE", b"OPTI", b"PATC", b"CONN", b"TRAC")
HTTP_RESPONSE = b"HTTP/1"


def is_tls_record(p):
    return len(p) >= 5 and p[0] in (20, 21, 22, 23) and p[1] == 3 and 0 <= p[2] <= 4


def is_http_text(p):
    return len(p) >= 4 and (p.startswith(HTTP_METHODS) or p.startswith(HTTP_RESPONSE))


def is_quic_long_header(p):
    if len(p) < 5 or not (p[0] & 0x80):
        return False
    v = int.from_bytes(p[1:5], "big")
    return v == 0 or v in (0x00000001, 0x6b3343cf) or (v & 0xFF000000) == 0xFF000000


def get_ips(ip):
    if isinstance(ip, dpkt.ip.IP):
        fam = socket.AF_INET
    elif isinstance(ip, dpkt.ip6.IP6):
        fam = socket.AF_INET6
    else:
        return None, None
    return socket.inet_ntop(fam, ip.src), socket.inet_ntop(fam, ip.dst)


def open_reader(f):
    """pcap reader; falls back to pcapng if the file is really pcapng."""
    try:
        return dpkt.pcap.Reader(f)
    except (ValueError, dpkt.dpkt.UnpackError):
        f.seek(0)
        return dpkt.pcapng.Reader(f)


def iter_packets(reader):
    """Yield (ts, buf); stop quietly (with a warning) at a truncated or corrupt capture tail."""
    it = iter(reader)
    while True:
        try:
            yield next(it)
        except StopIteration:
            return
        except Exception as e:
            print(f"WARNING: stopped reading capture early ({e!r}); keeping packets read so far", flush=True)
            return


class RunningStats:
    __slots__ = ("n", "total", "total_sq", "vmin", "vmax")

    def __init__(self):
        self.n, self.total, self.total_sq, self.vmin, self.vmax = 0, 0.0, 0.0, None, None

    def update(self, x):
        self.n += 1
        self.total += x
        self.total_sq += x * x
        if self.vmin is None or x < self.vmin:
            self.vmin = x
        if self.vmax is None or x > self.vmax:
            self.vmax = x

    def mean(self):
        return self.total / self.n if self.n else 0.0

    def std(self):
        if self.n < 2:
            return 0.0
        var = self.total_sq / self.n - self.mean() ** 2
        return math.sqrt(var) if var > 0 else 0.0

    def min(self):
        return self.vmin if self.vmin is not None else 0.0

    def max(self):
        return self.vmax if self.vmax is not None else 0.0


class Flow:
    __slots__ = ("first_ts", "last_ts", "last_pkt_ts", "pk_ab", "pk_ba", "by_ab", "by_ba",
                 "iat", "size", "is_http", "is_tls", "is_quic", "tls_hs", "http_method",
                 "syn", "fin", "rst", "psh", "ack", "win")

    def __init__(self, ts):
        self.first_ts = self.last_ts = ts
        self.last_pkt_ts = None
        self.pk_ab = self.pk_ba = self.by_ab = self.by_ba = 0
        self.iat, self.size, self.win = RunningStats(), RunningStats(), RunningStats()
        self.is_http = self.is_tls = self.is_quic = self.tls_hs = False
        self.http_method = ""
        self.syn = self.fin = self.rst = self.psh = self.ack = 0


def process_pcap(path):
    flows = {}
    with open(path, "rb") as f:
        for ts, buf in iter_packets(open_reader(f)):
            try:
                ip = dpkt.ethernet.Ethernet(buf).data
                src, dst = get_ips(ip)
                if src is None:
                    continue
                if isinstance(ip.data, dpkt.tcp.TCP):
                    proto, l4 = "tcp", ip.data
                elif isinstance(ip.data, dpkt.udp.UDP):
                    proto, l4 = "udp", ip.data
                else:
                    continue
                a, b = (src, l4.sport), (dst, l4.dport)
            except Exception:
                continue
            ca, cb = (a, b) if a <= b else (b, a)
            key = (proto, ca, cb)
            fl = flows.get(key)
            if fl is None:
                fl = flows[key] = Flow(ts)
            if fl.last_pkt_ts is not None:
                fl.iat.update(ts - fl.last_pkt_ts)
            fl.last_pkt_ts = fl.last_ts = ts
            n = len(buf)
            fl.size.update(n)
            if a == ca:
                fl.pk_ab += 1
                fl.by_ab += n
            else:
                fl.pk_ba += 1
                fl.by_ba += n
            try:
                payload = bytes(l4.data)
            except Exception:
                payload = b""
            if proto == "tcp":
                fg = l4.flags
                fl.syn += bool(fg & dpkt.tcp.TH_SYN)
                fl.fin += bool(fg & dpkt.tcp.TH_FIN)
                fl.rst += bool(fg & dpkt.tcp.TH_RST)
                fl.psh += bool(fg & dpkt.tcp.TH_PUSH)
                fl.ack += bool(fg & dpkt.tcp.TH_ACK)
                fl.win.update(l4.win)
                if is_tls_record(payload):
                    fl.is_tls = True
                    if payload[0] == 22:
                        fl.tls_hs = True
                elif is_http_text(payload):
                    fl.is_http = True
                    if not fl.http_method:
                        fl.http_method = payload.split(b" ", 1)[0].decode("ascii", "ignore")
            elif is_quic_long_header(payload):
                fl.is_quic = True
    return flows


FIELDS = ["profile", "capture_side", "proto", "ip_a", "port_a", "ip_b", "port_b",
          "start_time", "duration", "total_packets", "total_bytes",
          "pkts_a_to_b", "pkts_b_to_a", "bytes_a_to_b", "bytes_b_to_a",
          "mean_iat", "std_iat", "min_iat", "max_iat",
          "mean_pkt_size", "std_pkt_size", "min_pkt_size", "max_pkt_size",
          "is_http", "is_tls", "is_quic", "tls_handshake", "http_method",
          "tcp_syn_count", "tcp_fin_count", "tcp_rst_count", "tcp_psh_count", "tcp_ack_count",
          "mean_tcp_window", "label"]


def write_csv(flows, out, profile, side, c2_ips):
    pos = 0
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for (proto, (ipa, pa), (ipb, pb)), fl in flows.items():
            label = int(ipa in c2_ips or ipb in c2_ips)
            pos += label
            w.writerow({
                "profile": profile, "capture_side": side, "proto": proto,
                "ip_a": ipa, "port_a": pa, "ip_b": ipb, "port_b": pb,
                "start_time": fl.first_ts, "duration": fl.last_ts - fl.first_ts,
                "total_packets": fl.pk_ab + fl.pk_ba, "total_bytes": fl.by_ab + fl.by_ba,
                "pkts_a_to_b": fl.pk_ab, "pkts_b_to_a": fl.pk_ba,
                "bytes_a_to_b": fl.by_ab, "bytes_b_to_a": fl.by_ba,
                "mean_iat": fl.iat.mean(), "std_iat": fl.iat.std(),
                "min_iat": fl.iat.min(), "max_iat": fl.iat.max(),
                "mean_pkt_size": fl.size.mean(), "std_pkt_size": fl.size.std(),
                "min_pkt_size": fl.size.min(), "max_pkt_size": fl.size.max(),
                "is_http": int(fl.is_http), "is_tls": int(fl.is_tls), "is_quic": int(fl.is_quic),
                "tls_handshake": int(fl.tls_hs), "http_method": fl.http_method,
                "tcp_syn_count": fl.syn, "tcp_fin_count": fl.fin, "tcp_rst_count": fl.rst,
                "tcp_psh_count": fl.psh, "tcp_ack_count": fl.ack,
                "mean_tcp_window": fl.win.mean(), "label": label})
    return pos


def main():
    if len(sys.argv) not in (5, 6):
        sys.exit("Usage: flow_features.py <in.pcap> <out.csv> <profile> <capture_side> [c2_ip,...]")
    inp, out, profile, side = sys.argv[1:5]
    raw_ips = sys.argv[5] if len(sys.argv) == 6 else os.environ.get("C2_IPS", DEFAULT_C2_IP)
    c2_ips = {x.strip() for x in raw_ips.split(",") if x.strip()}
    print(f"C2 IPs used for labelling: {sorted(c2_ips)}", flush=True)
    print(f"[1/2] Building flow table from {inp} ...", flush=True)
    flows = process_pcap(inp)
    print(f"      {len(flows):,} flows", flush=True)
    pos = write_csv(flows, out, profile, side, c2_ips)
    print(f"[2/2] Wrote {out}: {pos:,} C2 flows (label=1) of {len(flows):,}", flush=True)


if __name__ == "__main__":
    main()
