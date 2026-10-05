#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Runs every workload plain and under `gpu-offload run`, and prints the times side by side.

usage: bench.py [--reps N] [--levels 5,9] [--out DIR] [--only NAME,...]

Each workload runs N times plain and N times at each level, interleaved so that drift in machine
load affects all variants alike; the table shows medians. The output of every offloaded run is
compared with the plain run (numbers to 3 significant digits). Afterwards one more run per
workload is traced, and the per-call report is written to DIR/report.txt.

Optional: a virtualenv at ~/venvs/pypi with NumPy / SciPy / scikit-learn wheels from PyPI, which
bundle their own OpenBLAS, unlike the distribution's python3-numpy. Create it with:
  python3 -m venv ~/venvs/pypi && ~/venvs/pypi/bin/pip install numpy scipy scikit-learn pandas
"""
import argparse
import math
import os
import re
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.environ.get("GPU_OFFLOAD", os.path.expanduser("~/build/gpu-offload/gpu-offload"))
PYPI = os.path.expanduser("~/venvs/pypi/bin/python")


def workloads():
    w = lambda name: os.path.join(HERE, name)  # noqa: E731
    items = [
        ("np_linalg_f64", ["python3", w("np_linalg.py"), "3000", "float64"]),
        ("np_linalg_f32", ["python3", w("np_linalg.py"), "3000", "float32"]),
        ("np_linalg_f64_large", ["python3", w("np_linalg.py"), "6000", "float64"]),
        ("np_linalg_f32_large", ["python3", w("np_linalg.py"), "6000", "float32"]),
        ("np_mlp_f32", ["python3", w("np_mlp_f32.py")]),
        ("np_signal", ["python3", w("np_signal.py")]),
        ("sklearn_pipeline", ["python3", w("sklearn_pipeline.py")]),
        ("pandas_etl", ["python3", w("pandas_etl.py")]),
        ("octave_signal", ["octave", "--no-gui", "--quiet", w("octave_signal.m")]),
        ("r_regression", ["Rscript", w("r_regression.R")]),
    ]
    if os.access(PYPI, os.X_OK):
        items += [
            ("pypi_np_linalg_f64", [PYPI, w("np_linalg.py"), "6000", "float64"]),
            ("pypi_np_linalg_f32", [PYPI, w("np_linalg.py"), "6000", "float32"]),
            ("pypi_sklearn", [PYPI, w("sklearn_pipeline.py")]),
        ]
    return items


def timed(cmd):
    t0 = time.perf_counter()
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return time.perf_counter() - t0, result


def same_output(a, b):
    """Same text, with numbers equal to 3 significant digits."""
    ta, tb = re.split(r"(\s+)", a), re.split(r"(\s+)", b)

    def close(x, y):
        try:
            return math.isclose(float(x.strip(",")), float(y.strip(",")), rel_tol=2e-3, abs_tol=1e-9)
        except ValueError:
            return x == y
    return len(ta) == len(tb) and all(close(x, y) for x, y in zip(ta, tb))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--levels", default="", help="comma-separated levels (default: the configured level)")
    parser.add_argument("--out", default=os.path.expanduser(time.strftime("~/gpu-offload-results/%Y%m%d-%H%M%S")))
    parser.add_argument("--only", default="", help="comma-separated workload names")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # OpenBLAS defaults to one thread per logical CPU. On a 16-core / 32-thread machine inside WSL
    # that default ran these workloads 15-25x slower than one thread per physical core. Compare
    # the GPU against the well-configured CPU, not against that.
    if "OPENBLAS_NUM_THREADS" not in os.environ:
        cores = {line for line in subprocess.run(["lscpu", "-p=core,socket"], capture_output=True, text=True)
                 .stdout.splitlines() if line and not line.startswith("#")}
        os.environ["OPENBLAS_NUM_THREADS"] = str(len(cores))
    configured = subprocess.run([TOOL, "level"], capture_output=True, text=True).stdout.split()[1]
    levels = [int(v) for v in args.levels.split(",") if v] or [int(configured)]
    print(f"OPENBLAS_NUM_THREADS={os.environ['OPENBLAS_NUM_THREADS']}, {args.reps} run(s) each, median times\n")

    header = f"{'workload':<22} {'plain s':>8}" + "".join(f"{'level ' + str(v) + ' s':>12} {'speedup':>8}" for v in levels)
    print(header + "  output")
    traced = []
    for name, cmd in workloads():
        if args.only and name not in args.only.split(","):
            continue
        plain, offloaded, verdict, reference = [], {v: [] for v in levels}, "same", None
        for _ in range(args.reps):
            seconds, result = timed(cmd)
            plain.append(seconds)
            reference = result.stdout
            if result.returncode:
                verdict = "PLAIN RUN FAILED"
            for v in levels:
                seconds, result = timed([TOOL, "run", "-q", "-l", str(v)] + cmd)
                offloaded[v].append(seconds)
                if result.returncode:
                    verdict = "OFFLOADED RUN FAILED"
                elif verdict == "same" and not same_output(reference, result.stdout):
                    verdict = "DIFFERS"
        base = statistics.median(plain)
        row = f"{name:<22} {base:>8.2f}"
        for v in levels:
            t = statistics.median(offloaded[v])
            row += f"{t:>12.2f} {base / t:>7.2f}x"
        print(row + "  " + verdict, flush=True)

        # One traced run at the last level for the per-call report (not part of the timings).
        trace_dir = os.path.join(args.out, name)
        result = subprocess.run([TOOL, "run", "-l", str(levels[-1]), "--trace", trace_dir] + cmd,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        with open(trace_dir + ".summary.txt", "w") as f:
            f.write(result.stderr)
        traced.append(trace_dir)

    report = subprocess.run([TOOL, "report"] + traced, capture_output=True, text=True).stdout
    with open(os.path.join(args.out, "report.txt"), "w") as f:
        f.write(report)
    print("\n" + report[report.find("SUMMARY"):])
    print(f"full report: {args.out}/report.txt   (exit summary of each traced run: {args.out}/*.summary.txt)")


if __name__ == "__main__":
    sys.exit(main())
