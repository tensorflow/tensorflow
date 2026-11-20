#!/usr/bin/env python3
"""Alternate base vs new TF kernel benchmarks for several rounds.

Usage:
  run3.py --base DIR --new DIR --rounds 7 --min_time 0.3s \
      --filter 'BM_Bucketize|BM_Unique_FLOAT|BM_InTopK'

Each DIR must contain the three binaries:
  bucketize_op_test, in_topk_op_test, unique_op_test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# google-benchmark lines look like:
#   BM_Bucketize/1024/100           31842 ns        31800 ns     22040
# or with real_time:
#   BM_Bucketize/1024/100_mean ...
BENCH_RE = re.compile(
    r"^(?P<name>\S+)\s+(?P<real>[0-9.]+)\s+(?P<real_unit>ns|us|ms|s)\s+"
    r"(?P<cpu>[0-9.]+)\s+(?P<cpu_unit>ns|us|ms|s)"
)

UNIT_TO_US = {
    "ns": 1e-3,
    "us": 1.0,
    "ms": 1e3,
    "s": 1e6,
}

BINARIES = ("bucketize_op_test", "in_topk_op_test", "unique_op_test")


def run_one(binary: Path, min_time: str, filt: str | None) -> dict[str, float]:
    cmd = [
        str(binary),
        f"--benchmark_min_time={min_time}",
        "--benchmark_counters_tabular=true",
    ]
    if filt:
        cmd.append(f"--benchmark_filter={filt}")
    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    # Do not set --gtest_filter=-*: TF's test_main then skips registering /
    # running google-benchmarks as well. GTests are cheap next to the benches.
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    out = proc.stdout + "\n" + proc.stderr
    results: dict[str, float] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "Benchmark" in line and "Time" in line:
            continue
        if any(s in line for s in ("_mean", "_median", "_stddev", "_cv")):
            continue
        m = BENCH_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        # Prefer the /real_time row when both are present.
        us = float(m.group("real")) * UNIT_TO_US[m.group("real_unit")]
        if name.endswith("/real_time") or name not in results:
            results[name.replace("/real_time", "")] = us
    if not results:
        sys.stderr.write(f"No benchmarks parsed from {binary}:\n{out[-2000:]}\n")
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--new", required=True, type=Path)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--min_time", default="0.3s")
    ap.add_argument("--filter", default=None)
    ap.add_argument("--json_out", type=Path, default=None)
    args = ap.parse_args()

    for label, d in (("base", args.base), ("new", args.new)):
        for b in BINARIES:
            p = d / b
            if not p.exists():
                sys.stderr.write(f"missing {label} binary: {p}\n")
                return 2

    series: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"base": [], "new": []}
    )

    for r in range(args.rounds):
        order = ("base", "new") if r % 2 == 0 else ("new", "base")
        dirs = {"base": args.base, "new": args.new}
        print(f"=== round {r + 1}/{args.rounds} order={order} ===", flush=True)
        for label in order:
            for b in BINARIES:
                res = run_one(dirs[label] / b, args.min_time, args.filter)
                for name, us in res.items():
                    series[name][label].append(us)
                print(f"  {label}/{b}: {len(res)} benches", flush=True)

    rows = []
    print("\nname | base_med_us | new_med_us | speedup")
    print("-" * 72)
    for name in sorted(series):
        b = series[name]["base"]
        n = series[name]["new"]
        if not b or not n:
            continue
        bm, nm = statistics.median(b), statistics.median(n)
        speed = bm / nm if nm else float("inf")
        rows.append(
            {
                "name": name,
                "base_us": bm,
                "new_us": nm,
                "speedup": speed,
                "base_all": b,
                "new_all": n,
            }
        )
        print(f"{name} | {bm:.1f} | {nm:.1f} | {speed:.2f}x")

    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
