#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Checks traces produced by the test programs.

usage: check_trace.py measure TRACE_DIR     trace of test_calls in measure mode
       check_trace.py gpu TRACE_DIR         trace of test_gpu_ops with GPU_OFFLOAD_FORCE=gpu
       check_trace.py verify TRACE_DIR      trace of test_gpu_ops with GPU_OFFLOAD_VERIFY=1
       check_trace.py nogpu TRACE_DIR       trace of test_gpu_ops with the backend missing
       check_trace.py policy TRACE_DIR on|off   trace of test_calls under a synthetic profile
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import ofl_model  # noqa: E402
import ofl_report as rep  # noqa: E402

failures = 0
REASON = {name: i for i, name in enumerate(rep.REASONS)}


def expect(cond, what):
    global failures
    if not cond:
        failures += 1
        print(f"FAIL: {what}")


def load(trace_dir):
    paths = rep.find_runs([trace_dir])
    expect(len(paths) == 1, f"one traced process expected, found {len(paths)}")
    meta, recs = rep.load_run(paths[0])
    expect(meta["dropped"] == 0 and meta["records"] == len(recs), "record count matches the meta file")
    return meta, recs


def check_measure(trace_dir):
    """The trace must contain exactly the calls tests/test_calls.c makes."""
    meta, recs = load(trace_dir)
    funcs = {f["id"]: f for f in meta["funcs"]}
    expect(meta["wall_s"] > 0 and len(meta["maps"]) > 0, "meta has wall time and module map")

    def select(name, top_only=True, queries=False):
        ids = [i for i, f in funcs.items() if f["name"] == name]
        sel = recs[[int(f) in ids for f in recs["func"]]] if len(recs) else recs
        if top_only:
            sel = sel[sel["depth"] == 0]
        if not queries:
            sel = sel[(sel["flags"] & rep.FLAG_WSQUERY) == 0]
        return sel

    # (symbol, calls made by the application, dims of the first call)
    expected = [
        ("cblas_dgemm", 4, (64, 32, 16)), ("dgemm_", 1, (64, 32, 16)), ("cblas_sgemm", 1, (48, 48, 48)),
        ("zgemm_", 1, (48, 48, 48)), ("cblas_dsyrk", 2, (48, 16)), ("cblas_dgemv", 1, (64, 16)),
        ("cblas_ddot", 1, (16,)), ("cblas_sdot", 1, (48,)), ("cblas_dnrm2", 1, (16,)), ("cblas_daxpy", 1, (16,)),
        ("cblas_dscal", 1, (16,)), ("dgesv_", 1, (48, 2)), ("dpotrf_", 1, (48,)), ("dsyev_", 1, (48,)),
        ("dgesdd_", 1, (48, 48)), ("fftw_execute*", 3, (1024, 0, 0, 1)), ("fftwf_execute*", 1, (32, 64, 0, 1)),
    ]
    for name, calls, dims in expected:
        sel = select(name)
        expect(len(sel) == calls, f"{name}: {len(sel)} top-level calls, expected {calls}")
        if len(sel):
            got = tuple(int(v) for v in sel["dims"][0][: len(dims)])
            expect(got == dims, f"{name}: dims {got}, expected {dims}")
            expect(bool((sel["dur_ns"] > 0).all()), f"{name}: durations recorded")
    expect(int((recs["device"] == rep.DEV_GPU).sum()) == 0, "measure mode never uses the GPU")

    # LAPACKE asks for the workspace size first; those queries must be flagged, not counted as work.
    expect(len(select("dsyev_", queries=True)) == 2, "dsyev_: workspace query recorded and flagged")
    expect(int(select("dsyev_")["variant"][0]) == 1, "dsyev_ with jobz=V is marked as computing vectors")

    # Buffer roles: C is write-only when beta == 0 and read-write otherwise.
    expect(select("cblas_dgemm")["buf_role"][0].tolist() == [1, 1, 2], "cblas_dgemm roles with beta == 0")
    expect(select("dgemm_")["buf_role"][0].tolist() == [1, 1, 3], "dgemm_ roles with beta != 0")
    expect(int(select("cblas_dgemm")["buf_bytes"][0][0]) == 64 * 16 * 8, "cblas_dgemm buffer size")
    expect(int(select("zgemm_")["buf_bytes"][0][0]) == 48 * 48 * 16, "zgemm_ buffer size (complex double)")

    # Residency: A and B are reused by calls 2 and 3; call 4 has an edited A (miss) and the same B (hit).
    res = rep.residency(select("cblas_dgemm"))
    expect(res["reads"] == 8 and res["hits"] == 5, f"cblas_dgemm reuse: {res['hits']} of {res['reads']}, expected 5 of 8")

    # FFTW geometry comes from the plan.
    many = select("fftw_execute*")[-1]
    expect(tuple(int(v) for v in many["dims"]) == (256, 0, 0, 8), "fftw_plan_many_dft geometry")
    expect(int(many["buf_bytes"][0]) == 256 * 8 * 16, "fftw many input size")
    r2c = select("fftwf_execute*")[0]
    expect(int(r2c["buf_bytes"][0]) == 32 * 64 * 4 and int(r2c["buf_bytes"][1]) == 32 * 33 * 8, "fftwf r2c sizes")
    expect(len(select("fftw_plan_*", top_only=False)) >= 2 and len(select("fftwf_plan_*", top_only=False)) >= 1,
           "FFTW planning calls recorded")

    a = rep.analyse(meta, recs)
    expect(0 < a["busy"] <= a["wall"], "busy time is positive and within wall time")
    print(f"{len(recs)} records ({int((recs['depth'] > 0).sum())} nested), {len(funcs)} distinct functions")


def capable(trace_dir):
    meta, recs = load(trace_dir)
    calls, ops, precs = ofl_model.gpu_capable(meta, recs)
    return meta, calls, ops, precs


def check_gpu(trace_dir):
    """Forced to the GPU, every GPU-capable call of test_gpu_ops must run there, except the wide SVD."""
    meta, calls, ops, precs = capable(trace_dir)
    on_gpu = calls["device"] == rep.DEV_GPU
    wide_svd = (ops == "gesvd") & (calls["dims"][:, 0] < calls["dims"][:, 1])
    expect(len(calls) >= 4 * 20, f"expected at least 80 GPU-capable calls, found {len(calls)}")
    expect(bool(on_gpu[~wide_svd].all()), f"{int((~on_gpu & ~wide_svd).sum())} supported calls did not run on the GPU: "
           + ", ".join(sorted({f"{precs[i]}{ops[i]}:{rep.REASONS[calls['reason'][i]]}"
                               for i in range(len(calls)) if not on_gpu[i] and not wide_svd[i]})))
    expect(int(wide_svd.sum()) == 4 and not on_gpu[wide_svd].any()
           and bool((calls["reason"][wide_svd] == REASON["arguments the GPU path does not handle"]).all()),
           "the wide SVD falls back to the CPU as an unsupported argument combination")
    expect(set(precs[on_gpu]) == set("sdcz"), "all four precisions ran on the GPU")
    expect(set(ops[on_gpu]) == set(rep.GPU_OPS.values()), f"every GPU operation was exercised: {sorted(set(ops[on_gpu]))}")
    expect(bool((calls["gpu_in_s"][on_gpu] > 0).all()), "GPU runs record their copy-in time")
    expect(meta["gpu"] != "" and meta["gpu_init_s"] > 0, "meta names the GPU and its start-up time")
    print(f"{int(on_gpu.sum())} calls on the GPU ({meta['gpu']}, start-up {meta['gpu_init_s']:.2f} s), "
          f"{int((~on_gpu).sum())} on the CPU")


def check_verify(trace_dir):
    """In verify mode the CPU result is kept and every supported call is compared with the GPU's."""
    meta, calls, ops, precs = capable(trace_dir)
    verified = (calls["flags"] & rep.FLAG_VERIFIED) != 0
    mismatched = (calls["flags"] & rep.FLAG_MISMATCH) != 0
    expect(int((calls["device"] == rep.DEV_GPU).sum()) == 0, "verify mode returns CPU results")
    expect(int(mismatched.sum()) == 0, f"{int(mismatched.sum())} GPU results differ from the CPU")
    expect(int(verified.sum()) >= 4 * 18, f"only {int(verified.sum())} calls were verified")
    print(f"{int(verified.sum())} calls verified against the CPU, {int(mismatched.sum())} mismatches")


def check_nogpu(trace_dir):
    meta, calls, ops, precs = capable(trace_dir)
    expect(int((calls["device"] == rep.DEV_GPU).sum()) == 0, "no call runs on a GPU that is not there")
    expect(bool((calls["reason"] == REASON["GPU unavailable"]).all()), "every call reports the GPU as unavailable")


def check_policy(trace_dir, expectation):
    """A synthetic profile that makes the GPU look fast (on) or slow (off) must steer the decision."""
    meta, calls, ops, precs = capable(trace_dir)
    in_profile = (ops == "gemm") & (precs == "d")  # the synthetic profile only knows double-precision gemm
    gemm = calls[in_profile]
    on_gpu = gemm["device"] == rep.DEV_GPU
    expect(bool((gemm["pred_cpu_s"] > 0).all()), "predictions are recorded for calibrated operations")
    if expectation == "on":
        # One call is deliberately sent to the CPU to check the profile's CPU speed against reality.
        probes = gemm["reason"] == REASON["re-checking the CPU's speed"]
        expect(bool((on_gpu | probes).all()) and int(probes.sum()) == 1,
               f"{int(on_gpu.sum())} of {len(gemm)} gemm calls on the GPU, {int(probes.sum())} CPU probes")
    else:
        expect(not on_gpu.any(), "gemm calls went to the GPU although the profile says it is slower")
        expect(bool((gemm["reason"] == REASON["predicted gain too small"]).all()), "the reason is the predicted gain")
    other = calls[~in_profile]
    expect(not (other["device"] == rep.DEV_GPU).any()
           and bool((other["reason"] == REASON["machine not calibrated"]).all()),
           "operations missing from the profile stay on the CPU")


def main():
    mode, trace_dir = sys.argv[1], sys.argv[2]
    {"measure": check_measure, "gpu": check_gpu, "verify": check_verify, "nogpu": check_nogpu}.get(
        mode, lambda d: check_policy(d, sys.argv[3]))(trace_dir)
    print(f"CHECK_{mode.upper()}_FAILED" if failures else f"CHECK_{mode.upper()}_OK")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
