#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Reads and summarises the per-call traces written by libgpuoffload.so.

Each trace directory holds the *.meta.json / *.trace files of one run. For every traced process
the report shows how much of the run was spent in interceptable math-library calls, how that time
splits by operation, precision and call size, what was offloaded to the GPU and why the rest
stayed on the CPU.
"""
import bisect
import glob
import json
import os

import numpy as np

REC = np.dtype([
    ("t_ns", "<u8"), ("dur_ns", "<u8"), ("callsite", "<u8"),
    ("buf_ptr", "<u8", 3), ("buf_bytes", "<u8", 3),
    ("flops", "<f8"), ("dims", "<i8", 4),
    ("tid", "<u4"), ("func", "<u2"), ("depth", "u1"), ("flags", "u1"),
    ("sig_pre", "<u4", 3), ("sig_post", "<u4", 3),
    ("buf_role", "u1", 3), ("device", "u1"), ("reason", "u1"), ("variant", "u1"), ("curve", "<u2"),
    ("pred_cpu_s", "<f4"), ("pred_gpu_s", "<f4"), ("gpu_in_s", "<f4"), ("gpu_out_s", "<f4"),
])
assert REC.itemsize == 168
MAGIC = b"GPUOFL1\0"
HEADER_BYTES = 16
ROLE_R, ROLE_W = 1, 2
FLAG_WSQUERY, FLAG_VERIFIED, FLAG_MISMATCH, FLAG_FAILED = 1, 2, 4, 8
DEV_GPU = 1
REASONS = ["", "predicted gain too small", "GPU start-up not yet worth it", "not enough free GPU memory",
           "GPU busy with other programs", "arguments the GPU path does not handle", "GPU error, fell back to CPU",
           "machine not calibrated", "GPU unavailable", "forced to CPU", "re-checking the CPU's speed"]
# Function-table operation name -> GPU-capable operation (same names as ofl_op_names in policy.c).
GPU_OPS = {"gemm": "gemm", "syrk": "syrk", "herk": "herk", "trsm": "trsm", "getrf": "getrf", "gesv": "gesv",
           "potrf": "potrf", "posv": "posv", "syev": "syevd", "syevd": "syevd", "heev": "syevd", "heevd": "syevd",
           "gesdd": "gesvd", "gesvd": "gesvd"}
PREC_NAME = {"s": "float32", "c": "float32", "d": "float64", "z": "float64"}
# Calls shorter than this are too short to repay a GPU kernel launch, whatever their size.
LARGE_CALL_NS = 1_000_000
DURATION_CLASSES = [(0, 10_000, "< 10 us"), (10_000, 1_000_000, "10 us - 1 ms"),
                    (1_000_000, 100_000_000, "1 ms - 100 ms"), (100_000_000, 1 << 62, ">= 100 ms")]
MAX_RESIDENCY_RECORDS = 5_000_000
# Cost of tracing one call, as measured by tests/run_tests.sh (it prints the current value).
TRACE_NS_PER_CALL = 150


def load_run(meta_path):
    """Loads one process: returns (meta dict, record array)."""
    with open(meta_path) as f:
        meta = json.load(f)
    if meta["record_size"] != REC.itemsize:
        raise SystemExit(f"{meta_path}: record size {meta['record_size']} does not match this script ({REC.itemsize})")
    trace_path = meta_path[: -len("meta.json")] + "trace"
    recs = np.zeros(0, REC)
    if os.path.exists(trace_path):
        with open(trace_path, "rb") as f:
            if f.read(8) != MAGIC:
                raise SystemExit(f"{trace_path}: not a gpu-offload trace")
        recs = np.fromfile(trace_path, dtype=REC, offset=HEADER_BYTES)
    return meta, recs


def find_runs(dirs):
    paths = []
    for d in dirs:
        paths += sorted(glob.glob(os.path.join(d, "*.meta.json")))
    return paths


def top_level_work(recs):
    """Calls made by the application itself, without workspace queries."""
    return recs[((recs["flags"] & FLAG_WSQUERY) == 0) & (recs["depth"] == 0)]


def transfer_bytes(recs):
    """Bytes read and written per call, the quantities the cost model charges for copies."""
    roles, sizes = recs["buf_role"], recs["buf_bytes"].astype(np.float64)
    return (sizes * ((roles & ROLE_R) != 0)).sum(axis=1), (sizes * ((roles & ROLE_W) != 0)).sum(axis=1)


def busy_seconds(recs):
    """Wall time during which at least one of the given calls was running (union of intervals)."""
    if len(recs) == 0:
        return 0.0
    order = np.argsort(recs["t_ns"], kind="stable")
    start = recs["t_ns"][order].astype(np.int64)
    end = start + recs["dur_ns"][order].astype(np.int64)
    reach = np.maximum.accumulate(end)
    prev_reach = np.concatenate(([start[0]], reach[:-1]))
    return float(np.sum(reach - np.maximum(start, prev_reach).clip(max=reach))) * 1e-9


def residency(recs):
    """Estimates host-to-GPU copy volume for copy-per-call versus keep-on-GPU-until-changed.

    A read counts as already resident when the same address was last seen with the same size and
    the same sampled content fingerprint. The fingerprint samples 8 words, so small in-place edits
    between calls can be missed: treat the result as an upper bound on what residency saves.
    """
    recs = recs[np.argsort(recs["t_ns"], kind="stable")][:MAX_RESIDENCY_RECORDS]
    resident = {}
    naive = moved = 0
    hits = reads = 0
    for ptrs, sizes, roles, pre, post in zip(recs["buf_ptr"].tolist(), recs["buf_bytes"].tolist(),
                                             recs["buf_role"].tolist(), recs["sig_pre"].tolist(),
                                             recs["sig_post"].tolist()):
        for i in range(3):
            role, size = roles[i], sizes[i]
            if not role or not size:
                continue
            if role & ROLE_R:
                reads += 1
                naive += size
                if resident.get(ptrs[i]) == (size, pre[i]) and pre[i]:
                    hits += 1
                else:
                    moved += size
                    resident[ptrs[i]] = (size, pre[i])
            if role & ROLE_W:
                resident[ptrs[i]] = (size, post[i])
    return dict(naive=naive, moved=moved, hits=hits, reads=reads)


def module_of(addr, maps):
    """Turns a call-site address into 'library+offset'."""
    starts = [m["start"] for m in maps]
    i = bisect.bisect_right(starts, addr) - 1
    if i >= 0 and addr < maps[i]["end"]:
        m = maps[i]
        return f"{os.path.basename(m['path'])}+0x{addr - m['start'] + m['off']:x}"
    return f"0x{addr:x}"


def gb(nbytes):
    return f"{nbytes / 1e9:.2f} GB" if nbytes >= 1e8 else f"{nbytes / 1e6:.1f} MB"


def pct(part, whole):
    return 100.0 * part / whole if whole > 0 else 0.0


def analyse(meta, recs):
    """Computes the numbers of one process report."""
    funcs = {f["id"]: f for f in meta["funcs"]}
    wall = meta["wall_s"]
    work = recs[(recs["flags"] & FLAG_WSQUERY) == 0]
    top = work[work["depth"] == 0]
    busy = busy_seconds(top)
    top_s = float(top["dur_ns"].sum()) * 1e-9

    by_op = {}
    for fid in np.unique(top["func"]):
        f = funcs[int(fid)]
        sel = top[top["func"] == fid]
        key = (f["fam"], f["op"], PREC_NAME[f["prec"]])
        agg = by_op.setdefault(key, dict(calls=0, gpu_calls=0, s=0.0, flops=0.0, dims=[], call_flops=[]))
        agg["calls"] += len(sel)
        agg["gpu_calls"] += int((sel["device"] == DEV_GPU).sum())
        agg["s"] += float(sel["dur_ns"].sum()) * 1e-9
        agg["flops"] += float(np.clip(sel["flops"], 0, None).sum())
        agg["dims"].append(sel["dims"])
        agg["call_flops"].append(sel["flops"])

    prec_s = {"float32": 0.0, "float64": 0.0}
    for (_, op, prec), agg in by_op.items():
        if op != "plan":
            prec_s[prec] += agg["s"]

    classes = []
    for lo, hi, label in DURATION_CLASSES:
        sel = top[(top["dur_ns"] >= lo) & (top["dur_ns"] < hi)]
        classes.append((label, len(sel), float(sel["dur_ns"].sum()) * 1e-9))
    large_s = busy_seconds(top[top["dur_ns"] >= LARGE_CALL_NS])

    sites = {}
    for addr in np.unique(top["callsite"]):
        sel = top[top["callsite"] == addr]
        sites[module_of(int(addr), meta["maps"])] = (len(sel), float(sel["dur_ns"].sum()) * 1e-9)

    # Offloading: what ran on the GPU, what the cost model expected, and why the rest stayed.
    gpu = top[top["device"] == DEV_GPU]
    predicted = gpu[gpu["pred_cpu_s"] > 0]
    offload = dict(
        gpu_calls=len(gpu), gpu_s=float(gpu["dur_ns"].sum()) * 1e-9,
        copy_s=float(gpu["gpu_in_s"].sum() + gpu["gpu_out_s"].sum()),
        cpu_would_s=float(predicted["pred_cpu_s"].sum()),
        gpu_predicted_s=float(predicted["pred_gpu_s"].sum()), gpu_actual_s=float(predicted["dur_ns"].sum()) * 1e-9,
        reasons={REASONS[i]: int((top["reason"] == i).sum()) for i in range(1, len(REASONS))},
        verified=int(((work["flags"] & FLAG_VERIFIED) != 0).sum()),
        mismatched=int(((work["flags"] & FLAG_MISMATCH) != 0).sum()),
    )

    return dict(wall=wall, cpu=meta["user_s"] + meta["sys_s"], busy=busy, top_s=top_s, top_calls=len(top),
                nested_calls=int((work["depth"] > 0).sum()), by_op=by_op, prec_s=prec_s, classes=classes,
                large_s=large_s, sites=sites, res=residency(top), dropped=meta["dropped"], offload=offload)


def print_process(meta, a):
    print("=" * 100)
    print(f"{meta['cmdline']}   (pid {meta['pid']})")
    print(f"  wall {a['wall']:.2f} s, CPU {a['cpu']:.2f} s on {meta['cpus']} threads, mode {meta.get('mode', '?')}")
    print(f"  in intercepted calls: {a['busy']:.2f} s = {pct(a['busy'], a['wall']):.1f}% of wall time "
          f"({a['top_calls']:,} calls from the application, {a['nested_calls']:,} nested inside them)")
    if a["dropped"]:
        print(f"  WARNING: {a['dropped']:,} calls were not traced (trace size cap); totals below are incomplete")
    overhead = (a["top_calls"] + a["nested_calls"]) * TRACE_NS_PER_CALL * 1e-9
    if overhead > 0.01 * a["wall"]:
        print(f"  note: tracing itself cost about {overhead:.2f} s of this run ({pct(overhead, a['wall']):.0f}% of wall)")
    if not a["top_calls"]:
        return
    if a["top_s"] > 1.05 * a["busy"]:
        print("  note: the application calls from several threads at once; the times below add up across threads")

    print("\n  By operation                         calls    on GPU     time s   % wall   GFLOP/s   typical size")
    for (fam, op, prec), agg in sorted(a["by_op"].items(), key=lambda kv: -kv[1]["s"])[:12]:
        # "Typical" = the call with the median amount of work.
        dims, call_flops = np.concatenate(agg["dims"]), np.concatenate(agg["call_flops"])
        med = dims[np.argsort(call_flops, kind="stable")[len(call_flops) // 2]]
        size = "x".join(str(v) for v in med if v > 0) or "-"
        rate = f"{agg['flops'] / agg['s'] / 1e9:8.1f}" if agg["s"] > 0 and op != "plan" else "       -"
        print(f"    {fam + ' ' + op:<18} {prec:<8} {agg['calls']:>12,} {agg['gpu_calls']:>9,} {agg['s']:>10.3f} "
              f"{pct(agg['s'], a['wall']):>7.1f}% {rate}   {size}")

    print("\n  By call duration                     calls     time s   % of intercepted time")
    for label, calls, secs in a["classes"]:
        print(f"    {label:<27} {calls:>12,} {secs:>10.3f} {pct(secs, a['top_s']):>7.1f}%")

    total = a["prec_s"]["float32"] + a["prec_s"]["float64"]
    print(f"\n  Precision: float64 {pct(a['prec_s']['float64'], total):.0f}%, "
          f"float32 {pct(a['prec_s']['float32'], total):.0f}% of intercepted time")

    o = a["offload"]
    if o["gpu_calls"] or any(o["reasons"].values()):
        print(f"  Offloading: {o['gpu_calls']:,} calls ran on the GPU in {o['gpu_s']:.3f} s "
              f"({o['copy_s']:.3f} s of that copying data)")
        if o["cpu_would_s"] > 0:
            print(f"    the CPU was predicted to need {o['cpu_would_s']:.3f} s for them; "
                  f"GPU time was predicted {o['gpu_predicted_s']:.3f} s, actual {o['gpu_actual_s']:.3f} s")
        for reason, count in o["reasons"].items():
            if count:
                print(f"    stayed on the CPU, {reason}: {count:,}")
        if o["verified"] or o["mismatched"]:
            print(f"    verify: {o['verified']:,} matched, {o['mismatched']:,} MISMATCHED")
    else:
        r = a["res"]
        print(f"  Host-to-GPU copies if every call were offloaded: {gb(r['naive'])} copying per call, "
              f"{gb(r['moved'])} keeping unchanged data on the GPU "
              f"({pct(r['naive'] - r['moved'], r['naive']):.0f}% avoided, {r['hits']:,} of {r['reads']:,} reads reused)")

    print("  Top call sites:")
    for site, (calls, secs) in sorted(a["sites"].items(), key=lambda kv: -kv[1][1])[:3]:
        print(f"    {site:<60} {calls:>10,} calls {secs:>9.3f} s")


def print_summary(rows):
    print("\n" + "=" * 100)
    print("SUMMARY")
    print(f"  {'program':<40} {'wall s':>8} {'intercepted':>12} {'in calls >= 1 ms':>17} {'float64':>8} "
          f"{'reuse':>6} {'on GPU':>8}")
    for label, a in rows:
        total = a["prec_s"]["float32"] + a["prec_s"]["float64"]
        r = a["res"]
        # Precision and reuse mean nothing for a program that barely touches the math libraries.
        negligible = a["busy"] < 0.001 * a["wall"]
        f64 = "-" if negligible else f"{pct(a['prec_s']['float64'], total):.0f}%"
        reuse = "-" if negligible else f"{pct(r['naive'] - r['moved'], r['naive']):.0f}%"
        print(f"  {label[:40]:<40} {a['wall']:>8.2f} {pct(a['busy'], a['wall']):>11.1f}% "
              f"{pct(a['large_s'], a['wall']):>16.1f}% {f64:>8} {reuse:>6} {a['offload']['gpu_calls']:>8,}")
    print("\n  intercepted      = share of wall time inside BLAS / LAPACK / FFTW calls made by the application")
    print("  in calls >= 1 ms = the part of that share in calls long enough to be worth offloading")
    print("  float64          = share of intercepted time in double precision")
    print("  reuse            = host-to-GPU copy volume a keep-data-on-GPU policy would avoid (upper bound)")
    print("  on GPU           = calls that were offloaded in this run")


def load_dirs(dirs):
    """Yields (label, meta, records) for the processes worth showing in each trace directory."""
    for d in dirs:
        procs = [load_run(path) for path in find_runs([d])]
        if not procs:
            continue
        # Helper processes (shells, launchers) make no calls; show them only if nothing else did.
        shown = [p for p in procs if len(p[1])] or [max(procs, key=lambda p: p[0]["wall_s"])]
        for meta, recs in shown:
            yield os.path.basename(os.path.abspath(d)) + ": " + meta["program"], meta, recs


def report(dirs):
    rows = []
    for label, meta, recs in load_dirs(dirs):
        a = analyse(meta, recs)
        print_process(meta, a)
        rows.append((label, a))
    if not rows:
        raise SystemExit("no traces found in " + ", ".join(dirs))
    print_summary(rows)
