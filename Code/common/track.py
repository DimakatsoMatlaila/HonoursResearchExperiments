#!/usr/bin/env python3
"""Progress dashboard: Slurm queue, submitted jobs, data stages, per-run training progress, failures."""
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter

PIPE = os.environ["PIPE"]
MANIFEST = os.environ["MANIFEST"]
CLEAN, LABELED, DATASETS = os.environ["CLEAN"], os.environ["LABELED"], os.environ["DATASETS"]
JOBLOG, FAILLOG = os.environ["JOBLOG"], os.environ["FAILLOG"]
EXCLUDED = [n for n in os.environ.get("EXCLUDE_NODES", "").split(",") if n]
USER = os.environ.get("USER", "")
STALL_MIN = 120

COLOR = sys.stdout.isatty() and "--no-color" not in sys.argv
C = {"red": "\033[38;5;167m", "grn": "\033[38;5;108m", "yel": "\033[38;5;179m", "blu": "\033[38;5;74m",
     "dim": "\033[38;5;245m", "bold": "\033[1m", "end": "\033[0m"}
STATE_COLOR = {"RUNNING": "grn", "COMPLETED": "grn", "PENDING": "yel", "FAILED": "red", "TIMEOUT": "red",
               "OUT_OF_MEMORY": "red", "NODE_FAIL": "red", "CANCELLED": "yel", "DONE": "grn", "STALLED": "red",
               "QUEUED": "yel"}


def paint(text, name):
    return f"{C[name]}{text}{C['end']}" if COLOR else str(text)


def state(s, width=0):
    return paint(s.ljust(width), STATE_COLOR.get(s.split()[0], "dim"))


def head(title):
    print("\n" + paint(f"== {title} ", "blu") + paint("=" * max(4, 66 - len(title)), "dim"))


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return ""


def age(path):
    try:
        return (time.time() - os.path.getmtime(path)) / 60
    except OSError:
        return None


def fmt_age(m):
    if m is None:
        return "-"
    return f"{m:.0f}m" if m < 120 else f"{m / 60:.1f}h"


def read_joblog():
    rows = []
    if os.path.exists(JOBLOG):
        for line in open(JOBLOG):
            p = line.rstrip("\n").split("\t")
            if len(p) == 4:
                rows.append(p)
    return rows


def queue():
    head("Slurm queue")
    out = sh(["squeue", "-u", USER, "-h", "-o", "%i|%j|%T|%M|%D|%R"])
    if not out.strip():
        print(paint("nothing queued or running", "dim"))
        return
    print(f"{'JOBID':<18}{'NAME':<26}{'STATE':<10}{'TIME':<12}{'NODES':<6}WHERE/REASON")
    counts, totals = Counter(), Counter()
    for line in out.strip().split("\n"):
        jid, name, st, t, n, where = line.split("|")
        counts[(name, st)] += 1
        totals[st] += 1
        if st == "RUNNING" or "_" not in jid or counts[(name, st)] <= 3:
            print(f"{jid:<18}{name:<26}{state(st, 10)}{t:<12}{n:<6}{where}")
        bad = [x for x in EXCLUDED if x in where]
        if bad:
            print(paint(f"   ^ running on excluded node {bad}", "red"))
    summary = ", ".join(f"{n} {s.lower()}" for s, n in sorted(totals.items()))
    print(paint(f"(array tasks are listed 3 per name+state)  totals: {summary}", "dim"))


def submitted():
    head("Submitted jobs (from logs/jobs.tsv)")
    rows = read_joblog()[-14:]
    if not rows:
        print(paint("no jobs submitted through the pipeline scripts yet", "dim"))
        return
    ids = ",".join(r[3] for r in rows)
    acct = {}
    for line in sh(["sacct", "-X", "-n", "-P", "-j", ids, "--format=JobID,State,Elapsed,ExitCode"]).strip().split("\n"):
        p = line.split("|")
        if len(p) == 4:
            base = p[0].split("_")[0]
            acct.setdefault(base, []).append(p[1].split()[0])
    print(f"{'WHEN':<20}{'STAGE':<8}{'JOB':<10}{'WHAT':<34}STATUS")
    for when, stage, what, jid in rows:
        sts = Counter(acct.get(jid, []))
        text = "unknown (sacct has nothing yet)" if not sts else ", ".join(f"{n} {s}" for s, n in sts.items())
        worst = next((s for s in ("OUT_OF_MEMORY", "TIMEOUT", "NODE_FAIL", "FAILED", "RUNNING", "PENDING")
                      if s in sts), "COMPLETED" if sts else "dim")
        print(f"{when:<20}{stage:<8}{jid:<10}{what[:33]:<34}{paint(text, STATE_COLOR.get(worst, 'dim'))}")


def bar(done, total, width=24):
    filled = int(width * done / total) if total else 0
    return "[" + "#" * filled + "." * (width - filled) + f"] {done}/{total}"


def data_stages():
    head("Data stages")
    if not os.path.exists(MANIFEST):
        print(paint("no manifest yet - run the data pipeline first", "dim"))
        return
    rels = [l.strip() for l in open(MANIFEST) if l.strip()]
    labelable = [r for r in rels if re.match(r"(internal|external)_", os.path.basename(r))]
    cleaned = sum(os.path.isfile(os.path.join(CLEAN, r)) for r in rels)
    labeled = sum(os.path.isfile(os.path.join(LABELED, os.path.basename(os.path.dirname(r)),
                  os.path.splitext(os.path.basename(r))[0] + ".csv")) for r in labelable)
    print(f"cleaned   {bar(cleaned, len(rels))}")
    print(f"labelled  {bar(labeled, len(labelable))}")
    per = Counter()
    for r in labelable:
        prof = os.path.basename(os.path.dirname(r))
        csv = os.path.join(LABELED, prof, os.path.splitext(os.path.basename(r))[0] + ".csv")
        per[(prof, os.path.isfile(csv))] += 1
    profs = sorted({p for p, _ in per})
    waiting = [f"{p} ({per[(p, False)]})" for p in profs if per[(p, False)]]
    print("not labelled yet: " + (", ".join(waiting) if waiting else paint("none", "grn")))
    for name in ("internal", "external", "master"):
        f = os.path.join(DATASETS, f"{name}_dataset.csv")
        if os.path.isfile(f):
            print(f"{name:<9} {paint('ready', 'grn')}  {os.path.getsize(f) / 1e9:.1f} GB  ({fmt_age(age(f))} old)")
        else:
            print(f"{name:<9} {paint('not built', 'yel')}")


def last_progress(log):
    fold = gen = None
    try:
        size = os.path.getsize(log)
        with open(log, "rb") as f:
            f.seek(max(0, size - 20000))
            tail = f.read().decode("utf-8", "ignore").split("\n")
    except OSError:
        return None, None, "no log yet"
    for line in reversed(tail):
        if gen is None and re.search(r"gen \d+/\d+", line):
            gen = re.search(r"gen \d+/\d+", line).group(0)
        if fold is None and line.startswith("=== Fold"):
            fold = line.strip("= ").strip()
        if fold and gen is not None:
            break
    last = next((l.strip() for l in reversed(tail) if l.strip()), "")
    return fold, gen, last[:70]


def training():
    head("Training runs")
    runs = sorted(glob.glob(os.path.join(PIPE, "models", "*", "results*", "*_*")))
    runs = [r for r in runs if os.path.isdir(r)]
    if not runs:
        print(paint("no training runs yet", "dim"))
        return
    print(f"{'MODEL':<17}{'RUN':<20}{'STATUS':<10}{'FOLDS':<9}{'LOG':<6}CURRENT")
    for r in runs:
        model = r.split(os.sep)[-3]
        run = os.path.basename(r) + ("*" if "results_smoke" in r else "")
        cfg = {}
        try:
            cfg = json.load(open(os.path.join(r, "run_config.json")))
        except Exception:
            pass
        total = len(cfg.get("malicious_profiles", [])) or "?"
        done = len(glob.glob(os.path.join(r, "checkpoints", "fold_*.pkl")))
        log = os.path.join(r, "train.log")
        a = age(log)
        fold, gen, last = last_progress(log)
        if os.path.exists(os.path.join(r, "DONE")):
            st = "DONE"
        elif os.path.exists(os.path.join(r, "FAILED")):
            st = "FAILED"
        elif os.path.exists(os.path.join(r, "STARTED")):
            st = "STALLED" if a is not None and a > STALL_MIN else "RUNNING"
        else:
            st = "QUEUED"
        extra = f"{fold or ''} {gen or ''}".strip() if st in ("RUNNING", "STALLED") else ""
        if st == "FAILED":
            extra = open(os.path.join(r, "FAILED")).read().strip()[:70]
        print(f"{model:<17}{run:<20}{state(st, 9)} {f'{done}/{total}':<9}{fmt_age(a):<6}{extra}")
    print(paint("* = smoke test.  STALLED = running but the log has not changed for 2 h.", "dim"))


def failures():
    head("Failures needing attention")
    if not os.path.exists(FAILLOG):
        print(paint("none recorded", "grn"))
        return
    seen, shown = set(), 0
    for line in open(FAILLOG):
        p = line.rstrip("\n").split("\t")
        if len(p) != 4:
            continue
        when, stage, item, jid = p
        if (stage, item) in seen:
            continue
        seen.add((stage, item))
        if stage == "clean" and os.path.isfile(os.path.join(CLEAN, item)):
            continue
        if stage == "label" and os.path.isfile(os.path.join(LABELED, os.path.basename(os.path.dirname(item)),
                                                os.path.splitext(os.path.basename(item))[0] + ".csv")):
            continue
        if stage == "train":
            m, run = item.split("/", 1)
            if os.path.exists(os.path.join(PIPE, "models", m, "results", run, "DONE")):
                continue
        print(f"{paint('x', 'red')} {when}  {stage:<6} {item}  (job {jid})")
        shown += 1
    if not shown:
        print(paint("none outstanding (earlier failures have since been fixed)", "grn"))
    else:
        print(paint("re-run the same stage to retry; finished work is skipped.", "dim"))


def main():
    print(paint(time.strftime("%a %d %b %Y %H:%M:%S"), "bold") + paint(f"   node(s) excluded: {','.join(EXCLUDED) or 'none'}", "dim"))
    queue()
    submitted()
    data_stages()
    training()
    failures()


if __name__ == "__main__":
    main()
