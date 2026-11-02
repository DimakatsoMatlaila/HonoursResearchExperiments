#!/usr/bin/env python3
"""Keep only HTTP and TLS/QUIC flows, matched on payload rather than port."""

import socket
import sys
from collections import namedtuple

import dpkt

HTTP_METHODS = (b"GET ", b"POST", b"PUT ", b"HEAD", b"DELE",
                b"OPTI", b"PATC", b"CONN", b"TRAC")
HTTP_RESPONSE = b"HTTP/1"

QUIC_V1 = 0x00000001
QUIC_V2 = 0x6b3343cf

FlowKey = namedtuple("FlowKey", ["proto", "a", "b"])


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


def flow_key(proto, src_ip, src_port, dst_ip, dst_port):
    a = (src_ip, src_port)
    b = (dst_ip, dst_port)
    if a > b:
        a, b = b, a
    return FlowKey(proto, a, b)


def is_tls_record(payload):
    if len(payload) < 5:
        return False
    content_type, major, minor = payload[0], payload[1], payload[2]
    return content_type in (20, 21, 22, 23) and major == 3 and 0 <= minor <= 4


def is_http_text(payload):
    if len(payload) < 4:
        return False
    return payload.startswith(HTTP_METHODS) or payload.startswith(HTTP_RESPONSE)


def is_quic_long_header(payload):
    if len(payload) < 5:
        return False
    first_byte = payload[0]
    if not (first_byte & 0x80):
        return False
    version = int.from_bytes(payload[1:5], "big")
    if version == 0:
        return True
    if version in (QUIC_V1, QUIC_V2):
        return True
    if (version & 0xFF000000) == 0xFF000000:
        return True
    return False


def get_ip_addrs(ip):
    if isinstance(ip, dpkt.ip.IP):
        fam = socket.AF_INET
    elif isinstance(ip, dpkt.ip6.IP6):
        fam = socket.AF_INET6
    else:
        return None, None
    return socket.inet_ntop(fam, ip.src), socket.inet_ntop(fam, ip.dst)


def classify_packet(eth):
    ip = eth.data
    src_ip, dst_ip = get_ip_addrs(ip)
    if src_ip is None:
        return None

    if isinstance(ip.data, dpkt.tcp.TCP):
        tcp = ip.data
        payload = bytes(tcp.data)
        if is_tls_record(payload) or is_http_text(payload):
            return ("tcp", src_ip, tcp.sport, dst_ip, tcp.dport)
        return None

    if isinstance(ip.data, dpkt.udp.UDP):
        udp = ip.data
        payload = bytes(udp.data)
        if is_quic_long_header(payload):
            return ("udp", src_ip, udp.sport, dst_ip, udp.dport)
        return None

    return None


def pass_one_identify_flows(path):
    flagged = set()
    with open(path, "rb") as f:
        reader = open_reader(f)
        for _, buf in iter_packets(reader):
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except Exception:
                continue
            result = classify_packet(eth)
            if result:
                proto, src_ip, src_port, dst_ip, dst_port = result
                flagged.add(flow_key(proto, src_ip, src_port, dst_ip, dst_port))
    return flagged


def pass_two_write_matching(in_path, out_path, flagged_flows):
    kept = 0
    total = 0
    with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
        reader = open_reader(fin)
        writer = dpkt.pcap.Writer(fout, linktype=reader.datalink())
        for ts, buf in iter_packets(reader):
            total += 1
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                ip = eth.data
                src_ip, dst_ip = get_ip_addrs(ip)
                if src_ip is None:
                    continue
                if isinstance(ip.data, dpkt.tcp.TCP):
                    proto = "tcp"
                    sport, dport = ip.data.sport, ip.data.dport
                elif isinstance(ip.data, dpkt.udp.UDP):
                    proto = "udp"
                    sport, dport = ip.data.sport, ip.data.dport
                else:
                    continue
            except Exception:
                continue

            key = flow_key(proto, src_ip, sport, dst_ip, dport)
            if key in flagged_flows:
                writer.writepkt(buf, ts=ts)
                kept += 1
    return kept, total


def main():
    if len(sys.argv) != 3:
        print("Usage: python filter_pcaps.py <input.pcap> <output.pcap>")
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]
    print(f"[1/2] Scanning {in_path} for HTTP/TLS/QUIC flows...", flush=True)
    flagged = pass_one_identify_flows(in_path)
    print(f"      Identified {len(flagged)} matching flows.", flush=True)

    print(f"[2/2] Writing matching packets to {out_path}...", flush=True)
    kept, total = pass_two_write_matching(in_path, out_path, flagged)
    pct = (kept / total * 100) if total else 0.0
    print(f"      Kept {kept}/{total} packets ({pct:.1f}%).", flush=True)


if __name__ == "__main__":
    main()
