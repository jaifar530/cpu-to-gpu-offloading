#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
# usage: run_tests.sh BUILD_DIR
#
# 1. Measure mode: test_calls gives the same results under the library, and the trace holds
#    exactly the calls it makes.
# 2. GPU path: test_gpu_ops checks every GPU operation in four precisions against references it
#    computes itself; forced to the GPU, all of them must pass and must really run on the GPU.
# 3. Verify mode: every GPU result matches the CPU result.
# 4. Fallback: with the GPU backend missing, everything still passes on the CPU.
# 5. Policy: a synthetic profile steers the decision both ways.
# 6. Prints the per-call overhead of the library.
set -euo pipefail

build="$1"
src="$(cd "$(dirname "$0")/.." && pwd)"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
lib="$build/libgpuoffload.so"
check() { python3 "$src/tests/check_trace.py" "$@"; }
# Starts from a clean configuration so the user's own level and profile cannot change the result.
under() { env -u GPU_OFFLOAD_LEVEL -u GPU_OFFLOAD_FORCE -u GPU_OFFLOAD_VERIFY HOME="$out/home" LD_PRELOAD="$lib" "$@"; }

echo "== 1. measure mode =="
"$build/test_calls"
under env GPU_OFFLOAD_MODE=measure GPU_OFFLOAD_TRACE="$out/measure" "$build/test_calls"
check measure "$out/measure"

echo "== 2. GPU path =="
"$build/test_gpu_ops"
if [ -f "$build/libgpuoffload-cuda.so" ] && nvidia-smi >/dev/null 2>&1; then
    under env GPU_OFFLOAD_FORCE=gpu GPU_OFFLOAD_TRACE="$out/gpu" "$build/test_gpu_ops"
    check gpu "$out/gpu"

    echo "== 3. verify mode =="
    under env GPU_OFFLOAD_VERIFY=1 GPU_OFFLOAD_SUMMARY=1 GPU_OFFLOAD_TRACE="$out/verify" "$build/test_gpu_ops"
    check verify "$out/verify"

    echo "== 5. policy =="
    # gemm in double precision only: 1e6 flops/s on one device, 1e12 on the other, free start-up.
    profile() {
        printf 'gpu-offload-profile 1\nparam init_s 0\nparam h2d_bps 1e12\nparam d2h_bps 1e12\n' >"$out/profile.$1"
        printf 'curve gemm 0 d cpu 2 1:%s 1e15:%s\ncurve gemm 0 d gpu 2 1:%s 1e15:%s\n' "$2" "$2" "$3" "$3" >>"$out/profile.$1"
    }
    profile on 1e6 1e12
    profile off 1e12 1e6
    for want in on off; do
        under env GPU_OFFLOAD_PROFILE="$out/profile.$want" GPU_OFFLOAD_MIN_FLOPS=0 GPU_OFFLOAD_TRACE="$out/policy-$want" \
            "$build/test_calls"
        check policy "$out/policy-$want" "$want"
    done
else
    echo "SKIPPED: no GPU backend or no GPU on this machine"
fi

echo "== 4. fallback without a GPU backend =="
under env GPU_OFFLOAD_FORCE=gpu GPU_OFFLOAD_BACKEND=/nonexistent GPU_OFFLOAD_TRACE="$out/nogpu" "$build/test_gpu_ops"
check nogpu "$out/nogpu"

echo "== 6. overhead on a tiny (4x4) dgemm =="
export OPENBLAS_NUM_THREADS=1
plain=$("$build/bench_overhead")
idle=$(under env GPU_OFFLOAD_PROFILE="$out/profile.off" "$build/bench_overhead")
traced=$(under env GPU_OFFLOAD_MODE=measure GPU_OFFLOAD_TRACE="$out/bench" "$build/bench_overhead")
echo "per call: ${plain} ns plain, ${idle} ns with offloading enabled (call stays on the CPU), ${traced} ns when tracing"
