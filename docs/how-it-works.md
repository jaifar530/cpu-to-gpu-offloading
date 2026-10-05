---
title: How it works
nav_order: 3
description: >-
  The design of gpu-offload: LD_PRELOAD interception of BLAS and LAPACK, a per-machine cost model
  fitted by calibration, the 1 to 10 aggressiveness level, run-time learning, CPU fallback and
  the cuBLAS / cuSOLVER backend.
---

# How gpu-offload works

## The idea

A GPU cannot run a CPU program. What it can do is run specific, well-defined, data-parallel
operations much faster than a CPU, when the operation is large enough to repay copying its data
across the PCIe bus. Programs written in NumPy, R, Octave, Julia, Fortran or C++ already hand
exactly such operations to shared math libraries: BLAS and LAPACK.

gpu-offload places itself at that library boundary. It does not translate machine code and it does
not guess what a loop means; it sees a call such as "multiply these two 6000 x 6000 matrices" and
decides which device should do it.

```
 your program  (unmodified)
       |  dgemm_, dsyevd_, cblas_sgemm, scipy_dgesv_64_, ...
       v
 libgpuoffload.so                      (LD_PRELOAD)
       |   describe the call: operation, precision, sizes, buffers, work in flops
       |   predict T_cpu and T_gpu from the machine profile
       |   decide
       +-------------------------------+
       v                               v
 the program's own CPU library   libgpuoffload-cuda.so   (loaded on first use)
 (OpenBLAS, reference LAPACK,    copy in -> cuBLAS / cuSOLVER -> copy out
  NumPy's bundled OpenBLAS)      any failure: return to the CPU path
```

## 1. Interception

`libgpuoffload.so` is loaded into the process with `LD_PRELOAD` and exports the same symbols as the
math libraries. The dynamic linker resolves the program's calls to these wrappers first.

- **468 wrappers are generated** from one table of operations in `tools/gen_wrappers.py`: every
  operation in four precisions (`s`, `d`, `c`, `z`), in the Fortran and CBLAS interfaces, and in
  three symbol naming schemes: the standard one (`dgemm_`, `cblas_dgemm`) and the two used by the
  OpenBLAS bundled in PyPI's NumPy and SciPy wheels (`scipy_dgemm_`, `scipy_dgemm_64_` with 64-bit
  integers).
- **Finding the real function.** A wrapper must call the library the program would have called.
  The usual `dlsym(RTLD_NEXT)` is not enough: NumPy loads its math library in a private scope where
  `RTLD_NEXT` cannot see it. The wrapper therefore first looks in the calling module and its
  dependencies, then in the global scope, then in every loaded object.
- **Nested calls.** A CPU LAPACK routine calls BLAS internally. A per-thread depth counter marks
  those nested calls so that reports count only what the application asked for.
- **Workspace queries** (`lwork = -1`) are recognised and always passed to the CPU library.
- **Fortran hidden string lengths** are forwarded, so the wrappers are correct for callers compiled
  by Fortran compilers and by C.

FFTW calls are intercepted for measurement only (plans are tracked so that each transform's size
is known). They are not offloaded yet.

## 2. The machine profile

`gpu-offload calibrate` runs a calibration workload twice under the library: once with every call
forced to the CPU and once forced to the GPU. The workload is an ordinary BLAS / LAPACK client, so
the GPU timings include everything a real call pays. From the two traces it fits:

| Quantity | How it is obtained |
| --- | --- |
| CPU speed curve per operation and precision | flops per second as a function of problem size (flops), from the CPU pass |
| GPU compute speed curve per operation and precision | the same from the GPU pass, with copy time removed |
| Host-to-GPU and GPU-to-host copy cost | a straight-line fit (fixed cost + bytes / bandwidth) over all GPU calls whose copies are known exactly |
| GPU start-up cost | time to load the CUDA libraries and create the context, measured once |

Each curve is a short list of (work, speed) points. The profile is a plain text file you can read.

**Protecting calibration from a noisy machine.** A multithreaded CPU library can run hundreds of
times slower for seconds when other programs or virtual machines take its cores (measured: a
complex LU of size 512 took 0.83 to 1.28 s during such an episode and a few milliseconds
normally). The
calibration workload therefore times a small fixed CPU job, a canary, before and after every
measurement point. A point starts only when the canary runs at its normal speed, and is measured
again if the canary slowed down meanwhile. The fit uses the lower quartile of the repetitions.
Failed LAPACK calls (`info != 0`) are flagged and excluded, because a call that fails returns
early and would otherwise look fast.

## 3. The decision

For every GPU-capable call with enough work to matter (at least 1e6 flops, roughly an 80 x 80
matrix product; smaller calls skip the decision and cost about 15 ns extra):

```
T_cpu = flops / cpu_speed(flops)                      * cpu_correction
T_gpu = ( copy_in_fixed  + bytes_in  / host_to_gpu_bandwidth
        + flops / gpu_speed(flops)
        + copy_out_fixed + bytes_out / gpu_to_host_bandwidth ) * gpu_correction

offload if   T_cpu >= required_speedup(level) * T_gpu
        and  the GPU has started, or starting it is now justified
        and  no other program is loading the GPU beyond the level's limit
        and  enough GPU memory is free beyond the level's reserve
```

Speed curves are interpolated in log-log space and held flat outside the calibrated range.

### The level

One number from 1 to 10 sets how much predicted gain is required and how considerate the tool is
towards other users of the GPU. It never trades away correctness.

| Level | Required predicted speedup | Back off when other programs load the GPU above | GPU memory always left free | Start the GPU once missed savings reach |
| --- | --- | --- | --- | --- |
| 1 | 3.0x | 30% | 50% | 3.0x its start-up cost |
| 2 | 2.5x | 37% | 45% | 2.75x |
| 3 | 2.0x | 44% | 40% | 2.5x |
| 4 | 1.6x | 52% | 35% | 2.25x |
| 5 (default) | 1.3x | 59% | 30% | 2.0x |
| 6 | 1.2x | 66% | 25% | 1.75x |
| 7 | 1.1x | 73% | 20% | 1.5x |
| 8 | 1.05x | 81% | 15% | 1.25x |
| 9 | 1.0x | 88% | 10% | 1.0x |
| 10 | 0.9x (accepts a small loss to free the CPU) | 95% | 5% | 0.75x |

### Not paying the GPU start-up cost for nothing

Loading the CUDA libraries and creating a GPU context takes a fraction of a second to more than a
second. A short program should not pay that. While the GPU is not started, every call that would
have been offloaded adds its predicted saving to a running total, and the call runs on the CPU. The
GPU is started only when that total reaches the start-up cost times the level's factor. This is the
ski-rental rule: the process never loses more than a bounded multiple of the start-up cost, and a
process that does little heavy math never touches CUDA at all.

### Sharing the GPU

- **Memory.** Before offloading, free GPU memory is checked against the call's needs plus the
  level's reserve. If an allocation still fails, the call returns to the CPU.
- **Other programs.** GPU utilisation is read through NVML. Because the figure includes this
  process's own work, it is consulted only when this process has not used the GPU for a second:
  it answers "is somebody else using the GPU?", and if so beyond the level's limit, calls stay on
  the CPU.
- **Device memory is not hoarded.** Scratch buffers are released when they add up to more than
  1 GB.

## 4. Learning while the program runs

Calibration is one measurement on one day. The program's own CPU library may differ from the one
calibrated (NumPy wheels bundle their own OpenBLAS), and the machine's load changes.

- Every call is timed on whichever device runs it. The ratio of observed to predicted time
  becomes a correction factor per operation, precision and device. Each new observation moves the
  factor halfway (in ratio terms) towards the latest ratio.
- **CPU probes.** When an operation's CPU speed has not been observed for 16 offloadable calls
  (and once right after the GPU starts, if it has not been observed yet), the next call runs on
  the CPU on purpose. Without this, a CPU that is faster than the profile claims would never be
  noticed, because it would never be given work.

## 5. The GPU backend

`libgpuoffload-cuda.so` is a separate shared object, loaded only when the first call is about to
be offloaded, so the preload library itself has no CUDA dependency.

| Step | Detail |
| --- | --- |
| Copy in | Operands go to device buffers with `cudaMemcpy` (or `cublasSetMatrix` when the leading dimension differs from the row count). |
| Compute | cuBLAS for `gemm`, `syrk`, `herk`, `trsm`; cuSOLVER for `getrf` / `getrs`, `potrf` / `potrs`, `syevd` / `heevd`, `gesvd`. |
| Copy out | Results are written to the program's buffers only after the GPU work has succeeded. |
| Semantics | LAPACK conventions are preserved: pivots, `info`, in-place factors, the untouched triangle of symmetric outputs. Row-major CBLAS calls are mapped to column-major ones. 64-bit-integer callers get 64-bit pivots. |
| Serialisation | One GPU call at a time per process. |
| Recursion guard | While the backend runs, the wrappers pass straight through, so a CUDA library calling host BLAS cannot re-enter the dispatcher. |
| `fork()` | A CUDA context does not survive `fork()`; a child of a process that had started the GPU stays on the CPU. |

Every failure before the copy-out step returns a reason code, and the CPU library runs the call
as if nothing had happened.

## 6. Verify mode

`gpu-offload run --verify` runs each GPU-capable call on both devices. The GPU writes into private
copies of the outputs; the program receives the CPU result; the two are compared (largest
difference relative to the largest value, with tolerances of 2e-3 for single and 1e-8 for double
precision). Eigenvectors and singular vectors are only defined up to sign, so for those operations
the values are compared. LU factors are compared only when both devices chose the same pivots.

## 7. Measurement and traces

With `--trace DIR` or in `measure` mode, each call becomes a 168-byte record: time, duration,
operation, sizes, estimated flops, operand addresses and sizes, call site, device, reason, the two
predictions, and copy times. Each buffer also gets a cheap content fingerprint (8 sampled words),
which lets the report estimate how much copying could be avoided if unchanged data stayed on the
GPU. The record layout is defined in `src/ofl.h` and mirrored in `tools/ofl_report.py`.

## Source map

| File | Role |
| --- | --- |
| `tools/gen_wrappers.py` | The table of operations and the wrapper generator. |
| `src/core.c` | Symbol resolution, tracing, process start, fork and exit. |
| `src/policy.c` | Profile loading, prediction, the level, dispatch, run-time learning, verify mode, summary. |
| `src/cuda_backend.c` | The cuBLAS / cuSOLVER implementations. |
| `src/calibrate.c` | The calibration workload with the quiet-machine check. |
| `src/fftw.c`, `src/fftw.inc` | FFTW measurement wrappers and the plan table. |
| `tools/ofl_model.py` | Profile fitting, the Python copy of the cost model, `estimate`. |
| `tools/ofl_report.py` | Trace reader and report. |
| `tools/gpu-offload` | The command-line tool. |
