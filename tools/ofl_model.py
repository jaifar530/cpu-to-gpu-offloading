#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""The machine profile: fitting it from calibration traces, and predicting with it.

The profile is what `gpu-offload calibrate` learns about one machine. For every GPU-capable
operation and precision it stores how fast the CPU library and the GPU run it as a function of
problem size, plus what copies between host and GPU cost. src/policy.c reads the same file and
uses the same formulas:

    T_cpu = flops / rate_cpu(flops)
    T_gpu = copy-in + flops / rate_gpu(flops) + copy-out
"""
import datetime
import math
import os

import numpy as np

import ofl_report as rep

# Required predicted speedup per aggressiveness level (index 1..10); mirrors set_level() in policy.c.
MIN_SPEEDUP = [None, 3.0, 2.5, 2.0, 1.6, 1.3, 1.2, 1.1, 1.05, 1.0, 0.9]
# Operations whose recorded read / write sizes are exactly what the GPU path copies.
EXACT_TRANSFER_OPS = {"gemm", "trsm", "getrf", "gesv", "potrf", "posv"}
VARIANT_LABEL = {("syevd", 0): "eigenvalues", ("syevd", 1): "eigenvectors",
                 ("gesvd", 0): "singular values", ("gesvd", 1): "SVD with vectors"}


def gpu_capable(meta, recs):
    """Top-level GPU-capable calls of one process, with their operation name and precision."""
    funcs = {f["id"]: f for f in meta["funcs"]}
    top = rep.top_level_work(recs)
    ops = np.array([rep.GPU_OPS.get(funcs[int(f)]["op"], "") for f in top["func"]], dtype=object)
    precs = np.array([funcs[int(f)]["prec"] for f in top["func"]], dtype=object)
    keep = ops != ""
    return top[keep], ops[keep], precs[keep]


# ---------------------------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------------------------

def _points(trace_dir, device):
    """{(op, variant, prec): {flops: [seconds, ...]}} of compute time on one device."""
    points = {}
    transfers = []  # (in bytes, in seconds, out bytes, out seconds) of GPU runs
    init_s = 0.0
    for _, meta, recs in rep.load_dirs([trace_dir]):
        init_s = max(init_s, meta.get("gpu_init_s", 0.0))
        calls, ops, precs = gpu_capable(meta, recs)
        calls_in, calls_out = rep.transfer_bytes(calls)
        on_device = (calls["device"] == rep.DEV_GPU) == (device == "gpu")
        succeeded = (calls["flags"] & rep.FLAG_FAILED) == 0
        for i in np.nonzero(on_device & succeeded & (calls["flops"] > 0))[0]:
            r = calls[i]
            seconds = r["dur_ns"] * 1e-9
            if device == "gpu":
                if ops[i] in EXACT_TRANSFER_OPS:
                    transfers.append((calls_in[i], r["gpu_in_s"], calls_out[i], r["gpu_out_s"]))
                seconds -= r["gpu_in_s"] + r["gpu_out_s"]
            key = (ops[i], int(r["variant"]), precs[i])
            points.setdefault(key, {}).setdefault(float(r["flops"]), []).append(max(seconds, 1e-7))
    return points, transfers, init_s


def _fit_line(x, y):
    """Least-squares y = fixed + x / rate through transfer measurements; returns (fixed, rate)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    slope, intercept = np.polyfit(x, y, 1) if len(x) >= 2 and x.max() > x.min() else (0.0, float(np.median(y)))
    if slope <= 0:  # degenerate data: fall back to the bandwidth of the largest transfer
        i = int(np.argmax(x))
        slope, intercept = y[i] / max(x[i], 1.0), 0.0
    return max(float(intercept), 0.0), 1.0 / float(slope)


def fit(cpu_dir, gpu_dir, info=None):
    """Builds a profile from a CPU-forced and a GPU-forced calibration trace."""
    cpu_points, _, _ = _points(cpu_dir, "cpu")
    gpu_points, transfers, init_s = _points(gpu_dir, "gpu")
    if not gpu_points or not transfers:
        raise SystemExit("calibration produced no GPU measurements: the GPU backend did not run")

    t = np.array(transfers)
    h2d_fixed, h2d_bps = _fit_line(t[:, 0], t[:, 1])
    d2h_fixed, d2h_bps = _fit_line(t[:, 2], t[:, 3])
    profile = dict(info=dict(info or {}), params=dict(init_s=init_s, h2d_fixed_s=h2d_fixed, h2d_bps=h2d_bps,
                                                      d2h_fixed_s=d2h_fixed, d2h_bps=d2h_bps), curves={})
    for device, points in (("cpu", cpu_points), ("gpu", gpu_points)):
        for key, by_size in points.items():
            # Slow outliers are one-off warm-up or another program taking the CPU (the calibration
            # workload repeats a point when that happens), so the lower quartile is the typical speed.
            curve = sorted((flops, flops / float(np.percentile(times, 25))) for flops, times in by_size.items())
            profile["curves"][key + (device,)] = curve
    return profile


def save(profile, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("gpu-offload-profile 1\n")
        for name, value in profile["info"].items():
            f.write(f"# {name}: {value}\n")
        for name, value in profile["params"].items():
            f.write(f"param {name} {value:.9g}\n")
        for (op, variant, prec, device), curve in sorted(profile["curves"].items()):
            pts = " ".join(f"{flops:.6g}:{rate:.6g}" for flops, rate in curve)
            f.write(f"curve {op} {variant} {prec} {device} {len(curve)} {pts}\n")


def load(path):
    profile = dict(info={}, params={}, curves={}, path=path)
    with open(path) as f:
        for line in f:
            words = line.split()
            if not words:
                continue
            if words[0] == "#" and ":" in line:
                name, value = line[1:].split(":", 1)
                profile["info"][name.strip()] = value.strip()
            elif words[0] == "param":
                profile["params"][words[1]] = float(words[2])
            elif words[0] == "curve":
                pts = [tuple(float(v) for v in w.split(":")) for w in words[6:6 + int(words[5])]]
                profile["curves"][(words[1], int(words[2]), words[3], words[4])] = pts
    return profile


# ---------------------------------------------------------------------------------------------
# Prediction (same formulas as src/policy.c)
# ---------------------------------------------------------------------------------------------

def curve_seconds(curve, flops):
    """Piecewise-linear in log-log space, flat outside the calibrated range."""
    if flops <= 0 or not curve:
        return 0.0
    xs = [math.log(f) for f, _ in curve]
    ys = [math.log(r) for _, r in curve]
    return flops / math.exp(float(np.interp(math.log(flops), xs, ys)))


def predict(profile, op, variant, prec, flops, in_bytes, out_bytes):
    """Returns (cpu seconds, gpu seconds), or None if the operation was not calibrated."""
    cpu = profile["curves"].get((op, variant, prec, "cpu"))
    gpu = profile["curves"].get((op, variant, prec, "gpu"))
    if not cpu or not gpu:
        return None
    p = profile["params"]
    copies = p["h2d_fixed_s"] + in_bytes / p["h2d_bps"] + p["d2h_fixed_s"] + out_bytes / p["d2h_bps"]
    return curve_seconds(cpu, flops), copies + curve_seconds(gpu, flops)


# ---------------------------------------------------------------------------------------------
# Human-readable views
# ---------------------------------------------------------------------------------------------

def _square_size(op, variant, prec, flops):
    """Matrix size n of the square calibration problem that has this many flops."""
    per_n3 = {"gemm": 2.0, "syrk": 1.0, "herk": 1.0, "trsm": 1.0, "getrf": 2 / 3, "gesv": 2 / 3, "potrf": 1 / 3,
              "posv": 1 / 3, "syevd": 9.0 if variant else 4 / 3, "gesvd": 12.0 if variant else 4.0}[op]
    if prec in "cz":
        per_n3 *= 4
    # Calibration sizes are powers of two; lower-order terms in the flop count are rounded away.
    return 2 ** round(math.log2((flops / per_n3) ** (1 / 3)))


def describe(profile):
    """Prints, per operation and precision, where the GPU starts to win on this machine."""
    p = profile["params"]
    for name, value in profile["info"].items():
        print(f"  {name}: {value}")
    print(f"  GPU start-up {p['init_s']:.2f} s, host-to-GPU {p['h2d_bps'] / 1e9:.1f} GB/s, "
          f"GPU-to-host {p['d2h_bps'] / 1e9:.1f} GB/s")
    print("\n  GPU speedup over the CPU for square problems, copies included (- = not measured)")
    sizes = [256, 512, 1024, 2048, 4096, 8192]
    print(f"  {'operation':<18} {'precision':<10}" + "".join(f"{'n=' + str(n):>9}" for n in sizes) + "   GPU wins from")
    keys = sorted({k[:3] for k in profile["curves"]})
    for op, variant, prec in keys:
        cpu = dict(profile["curves"].get((op, variant, prec, "cpu"), []))
        gpu = dict(profile["curves"].get((op, variant, prec, "gpu"), []))
        esz = {"s": 4, "d": 8, "c": 8, "z": 16}[prec]
        cells, first_win = [], None
        by_n = {_square_size(op, variant, prec, fl): fl for fl in sorted(set(cpu) & set(gpu))}
        for n in sizes:
            flops = by_n.get(n)
            if flops is None:
                cells.append(f"{'-':>9}")
                continue
            # Copies for a square problem: about two matrices in and one out.
            t_cpu, t_gpu = predict(profile, op, variant, prec, flops, 2 * n * n * esz, n * n * esz)
            speedup = t_cpu / t_gpu
            cells.append(f"{speedup:>8.1f}x")
            if speedup >= MIN_SPEEDUP[5] and first_win is None:
                first_win = n
        label = VARIANT_LABEL.get((op, variant), op)
        precision = {"s": "float32", "d": "float64", "c": "complex64", "z": "complex128"}[prec]
        print(f"  {label:<18} {precision:<10}" + "".join(cells) + f"   {'n >= ' + str(first_win) if first_win else 'never'}")


def estimate(dirs, profile):
    """Predicts, from measure-mode traces, what offloading would save at each aggressiveness level."""
    p = profile["params"]
    for label, meta, recs in rep.load_dirs(dirs):
        calls, ops, precs = gpu_capable(meta, recs)
        calls_in, calls_out = rep.transfer_bytes(calls)
        cpu_s = calls["dur_ns"] * 1e-9
        gpu_s = np.full(len(calls), np.inf)
        for i in range(len(calls)):
            pred = predict(profile, ops[i], int(calls["variant"][i]), precs[i], float(calls["flops"][i]),
                           calls_in[i], calls_out[i])
            if pred:
                gpu_s[i] = pred[1]
        wall = meta["wall_s"]
        print("=" * 100)
        print(f"{label}: wall {wall:.2f} s, {len(calls):,} GPU-capable calls taking {cpu_s.sum():.2f} s on the CPU")
        print(f"  {'level':>5} {'offloaded calls':>16} {'CPU time replaced':>18} {'GPU time':>9} {'new wall':>9} {'speedup':>8}")
        for level in range(1, 11):
            chosen = cpu_s >= MIN_SPEEDUP[level] * gpu_s
            saved = float((cpu_s[chosen] - gpu_s[chosen]).sum()) - (p["init_s"] if chosen.any() else 0.0)
            if saved <= 0:  # the start-up cost is never recovered: the policy stays on the CPU
                chosen, saved = np.zeros(len(calls), bool), 0.0
            print(f"  {level:>5} {int(chosen.sum()):>16,} {cpu_s[chosen].sum():>16.2f} s {gpu_s[chosen].sum():>7.2f} s "
                  f"{wall - saved:>7.2f} s {wall / (wall - saved):>7.2f}x")
    print("\n  Estimates use the CPU times measured in the trace and GPU times predicted by this machine's profile.")
    print("  They include the GPU start-up cost and ignore GPU memory limits and other programs using the GPU.")


def machine_info():
    info = {"calibrated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}
    try:
        with open("/proc/cpuinfo") as f:
            models = [line.split(":", 1)[1].strip() for line in f if line.startswith("model name")]
        info["cpu"] = f"{models[0]} ({len(models)} threads)"
    except (OSError, IndexError):
        pass
    threads = os.environ.get("OPENBLAS_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS")
    info["blas threads"] = threads if threads else "library default"
    return info
