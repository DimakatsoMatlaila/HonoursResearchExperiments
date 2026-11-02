#!/usr/bin/env python3
"""Check the python environment has what the pipeline needs."""
import importlib
import sys

ok = True
for m in ["numpy", "pandas", "sklearn", "scipy", "matplotlib", "dpkt", "torch"]:
    try:
        mod = importlib.import_module(m)
        print(f"ok   {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        ok = False
        print(f"FAIL {m}: {e}")
print("python", sys.version.split()[0])
sys.exit(0 if ok else 1)
