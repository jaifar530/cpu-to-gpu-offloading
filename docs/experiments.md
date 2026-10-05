---
title: Experiments and benchmarks
nav_order: 5
description: >-
  Every experiment made while building gpu-offload: CPU versus GPU matrix-product benchmarks,
  the measurement study of NumPy, scikit-learn, pandas, R and Octave workloads, the full
  calibration table of an RTX 5080 against a Ryzen 9 9950X3D, correctness tests, overheads and
  end-to-end speedups, including the results that went wrong and why.
---

# Experiments and benchmarks

This page records every measurement made while building gpu-offload v0.1, in the order they were
made, including the ones that turned out to be wrong and what was learned from them. All numbers
come from one machine. They describe that machine; yours will differ, which is why the tool
calibrates.

## Summary of findings

| # | Finding | Consequence for the design |
| --- | --- | --- |
| 1 | On an RTX 5080, float32 matrix products are 3 to 6 times faster than on a 16-core Ryzen; float64 products are **slower** than on that CPU. | Precision is a first-class input of the cost model. |
| 2 | OpenBLAS's default thread count (32, one per logical CPU) made NumPy workloads 15 to 25 times slower than 16 threads. | Calibration measures the CPU as configured, and warns when one thread per core is faster. |
| 3 | Numeric programs spend 15% to 60% of their run time in interceptable library calls; data pipelines and NumPy FFT spend 0%. | `gpu-offload measure` exists so users can check before expecting anything. |
| 4 | scikit-learn's k-means makes 467,664 tiny matrix products; no per-call offloader can help. | Calls below 1e6 flops skip the decision. |
| 5 | NumPy's `linalg` solvers compute in float64 even for float32 input. | For NumPy users the GPU's float64 speed is what matters for solvers. |
| 6 | The CPU math library ran up to 300 times slower for seconds at a time while other virtual machines were busy. | Calibration waits for a quiet machine and repeats disturbed points; the run time re-checks the CPU's speed. |
| 7 | The cost model predicted a 6000 x 6000 eigendecomposition at 2.23 s on the GPU; it took 2.27 s. | The model is accurate where it matters. |
| 8 | Whole-program speedup on large NumPy linear algebra: 1.53x to 1.69x (median of 3). Workloads without large dense math: unchanged. | The gain is real, bounded, and specific to dense linear algebra. |
| 9 | 168 result checks pass on the GPU in four precisions; 128 GPU results compared with CPU results, 0 mismatches. | The GPU path is correct on the tested operations. |

## Test environment

| Item | Value |
| --- | --- |
| CPU | AMD Ryzen 9 9950X3D, 16 cores / 32 threads |
| Memory | 125.7 GB |
| GPU | NVIDIA GeForce RTX 5080, 16 GB, driver 616.92 |
| Host | Windows 11 Pro (build 26200), WSL 2.7.13, kernel 6.18.33.2-microsoft-standard-WSL2 |
| Guest | Ubuntu 24.04.5 LTS |
| CUDA | Toolkit 13.3 (cuBLAS 13.6, cuSOLVER 12.2) |
| CPU math library | OpenBLAS 0.3.26 (Ubuntu's pthread build) as system BLAS / LAPACK; `OPENBLAS_NUM_THREADS=16` unless stated |
| Compilers and tools | GCC 13.3, CMake 3.28.3 |
| System Python stack | NumPy 1.26.4, SciPy 1.11.4, scikit-learn 1.4.1, pandas 2.1.4 |
| PyPI Python stack (virtualenv) | NumPy 2.5.3, SciPy 1.18.1, scikit-learn 1.9.1, pandas 3.0.6 (these wheels bundle their own OpenBLAS) |
| Other | GNU Octave 8.4.0, R 4.3.3 |
| Dates | 4 and 5 October 2026 |

**Important caveat.** The machine was not dedicated. The Windows host also ran several Hyper-V
virtual machines used as CI runners, plus Docker containers, and the display used the same GPU.
Their load came and went. This is the source of the timing noise described below, and it is why
medians and repeated measurements are used. It also makes the environment realistic for a
workstation.

## Experiment 1: CPU versus GPU on a matrix product

**Question.** Before building anything: does the GPU beat the CPU on the most basic operation,
with copies included?

**Method.** A C program multiplies two n x n random matrices with OpenBLAS (`cblas_?gemm`) and with
cuBLAS (`cublas?gemm`), timing the GPU's copy-in, compute and copy-out separately after a warm-up
call, and compares the results.

**First run** (OpenBLAS default threads, single run):

| Operation | CPU | GPU copy in | GPU compute | GPU copy out | GPU total | GPU vs CPU |
| --- | --- | --- | --- | --- | --- | --- |
| float64, 4096 x 4096 | 197.1 ms | 19.4 ms | 182.0 ms | 9.4 ms | 210.8 ms | about equal |
| float64, 8192 x 8192 | 1083.3 ms | 78.7 ms | 1398.4 ms | 41.5 ms | 1518.5 ms | 1.4x slower |
| float32, 4096 x 4096 | 78.5 ms | 10.7 ms | 3.9 ms | 5.1 ms | 19.7 ms | 4.0x faster |
| float32, 8192 x 8192 | 540.6 ms | 42.1 ms | 30.1 ms | 20.3 ms | 92.4 ms | 5.9x faster |

**Repeated with 16 threads, best of 3:**

| Operation | CPU | GPU including copies |
| --- | --- | --- |
| float64, 4096 x 4096 | 162.6 ms | 224.9 ms |
| float64, 8192 x 8192 | 1425.8 ms | 1601.7 ms |
| float32, 4096 x 4096 | 70.1 ms | 22.4 ms |
| float32, 8192 x 8192 | 557.3 ms | 94.5 ms |

CPU and GPU results agreed to a relative difference of 5.5e-15 to 7.9e-15 in float64 and 1.2e-06
to 4.3e-06 in float32.

**What it showed.**

- In float32 the GPU's compute time is tiny (3.9 ms against 78.5 ms); **the copies dominate** the
  GPU's total (15.8 of 19.7 ms).
- In float64 this consumer GPU is no faster than this CPU. Most scientific tools default to
  float64.

## Experiment 2: the CPU baseline depends on the thread count

**Trigger.** In the first measurement run two workloads took 37 to 100 s, and repeated runs of the
same script varied by a factor of two.

**Method.** Two NumPy workloads timed with `OPENBLAS_NUM_THREADS` set to 32 (the default: one per
logical CPU), 16 (one per physical core) and 8.

| Workload | 32 threads | 16 threads | 8 threads |
| --- | --- | --- | --- |
| NumPy linear algebra (n = 3000) | 37.20 s | 2.47 s | 2.55 s |
| NumPy neural-net training (float32) | 56.59 s | 2.36 s | 2.19 s |

**What it showed.** The default configuration was 15 to 25 times slower. Any GPU comparison made
against the default would have reported enormous, meaningless speedups. All later experiments use
16 threads, and `gpu-offload calibrate` now tests this and tells the user.

## Experiment 3: how much of real programs is offloadable?

**Question.** The gate for the whole project: do real programs spend enough time in calls that
could be offloaded?

**Method.** The library in measure mode (no offloading) recorded every BLAS, LAPACK and FFTW call
of ten runs: seven programs, three of them also run with the PyPI NumPy / SciPy stack.

### First run: wrong (default thread count)

| Program | Wall time | In interceptable calls |
| --- | --- | --- |
| NumPy linear algebra (system NumPy) | 103.5 s | 98.4% |
| NumPy neural-net training (system) | 54.9 s | 96.2% |
| R regression script | 17.8 s | 90.5% |
| Octave FFT + linear algebra | 8.3 s | 80.5% |

These shares were inflated by the thread problem of experiment 2: the CPU library was thrashing,
so nearly all the time was inside it. They are kept here as a warning.

### Corrected run (16 threads)

| Program | Wall time | In interceptable calls | In calls of 1 ms or longer | Precision of that time | Copy volume a keep-on-GPU scheme could avoid |
| --- | --- | --- | --- | --- | --- |
| NumPy linear algebra (system NumPy) | 4.59 s | 60.3% | 57.3% | float64 | 66% |
| NumPy linear algebra (PyPI NumPy) | 3.94 s | 57.9% | 50.4% | float64 | 67% |
| scikit-learn pipeline (system) | 8.29 s | 52.9% | 39.5% | float64 | 67% |
| scikit-learn pipeline (PyPI) | 11.82 s | 46.9% | 36.9% | float64 | 67% |
| NumPy neural-net training (PyPI) | 2.59 s | 46.3% | 11.1% | float32 | 33% |
| NumPy neural-net training (system) | 3.28 s | 43.2% | 17.7% | float32 | 33% |
| R regression script | 2.93 s | 33.5% | 14.2% | float64 | 73% |
| Octave FFT + linear algebra | 2.23 s | 15.5% | 7.4% | float64 | 27% |
| pandas data pipeline | 9.07 s | 0.0% | 0.0% | n/a | n/a |
| NumPy FFT signal processing | 0.72 s | 0.0% | 0.0% | n/a | n/a |

The last column is an upper bound: it compares address, size and an 8-word content sample of each
operand between calls.

### Where the time went

| Program | Largest contributors (share of wall time) |
| --- | --- |
| NumPy linear algebra (system) | linear solve `gesv` 19.1% (6 calls), eigenvalues `syevd` 14.9% (1 call), matrix product `gemm` 10.5% (68 calls), SVD `gesdd` 6.8%, QR `geqrf` 4.2% (60 calls) |
| NumPy neural-net training | matrix product `gemm` 43.2%: 5,617 calls, typical size 256 x 256 x 512, most under 1 ms |
| scikit-learn pipeline | 467,664 `gemm` calls of size 16 x 256 x 30 from k-means (made from several threads), and one SVD of a 300,000 x 120 matrix taking 2.9 s |
| R regression | `dot` 13.8% (1,952 calls on 400,000-element vectors) and `axpy` 6.6% from `lm.fit`, eigenvalues `syevr` 3.6%, SVD 2.9% |
| Octave | FFTW transforms 8.8% (612 calls of 65,536 points), the rest spread thinly |

**What it showed.**

- A real but bounded opportunity: even the best case (60% interceptable) caps the whole-program
  speedup at 2.5x with an infinitely fast GPU.
- Almost all of it is float64, where experiment 1 says this GPU does not win on products. The
  solvers (eigenvalues, SVD, linear solve) had not been measured on the GPU yet; they turned out to
  be the part that wins (experiment 5).
- Three kinds of work are out of reach of a library-level tool: very many tiny calls
  (scikit-learn), level-1 operations on long vectors (R's `lm`), and code compiled into the
  application (NumPy's FFT).

## Experiment 4: overhead of the library

**Method.** 500,000 calls of a 4 x 4 `dgemm`, the worst case for overhead, timed plain, with
offloading enabled, and with tracing.

| Configuration | Time per call |
| --- | --- |
| Plain | 13 to 14 ns |
| Offloading enabled, call stays on the CPU (final design) | 28 to 30 ns |
| Offloading enabled, before the small-call fast path was added | 120 ns |
| Tracing every call to disk | 131 to 213 ns |

The fast path (calls below 1e6 flops skip the cost model) was added because of the 120 ns result.
For the scikit-learn workload this means 467,664 calls x about 15 ns = 7 ms of overhead.

## Experiment 5: calibration of the machine

**Method.** `gpu-offload calibrate --full`: every GPU-capable operation, four precisions, square
problems of size 128 to 8192, at least 3 and up to 12 repetitions per point, once forced to the
CPU and once forced to the GPU. Three complete calibrations were run because the first two exposed
problems.

| Run | Duration | Outcome |
| --- | --- | --- |
| 1 | 2061 s | Usable but with impossible outliers (for example a complex LU "30x faster on the GPU" at n = 512), and one bogus point caused by a failed CPU call |
| 2 | 1837 s (CPU pass 1231 s, GPU pass 724 s) | Different outliers (a complex128 LU "198x faster" at n = 512) |
| 3 | 1969 s | Clean; this is the profile used for all results below |

### What went wrong in runs 1 and 2

**A failed call looked fast.** LAPACKE's `cheevd` at n = 4096 failed on the CPU with `info = -11`
("parameter 10 had an illegal value"): its workspace size is returned through a single-precision
number, which cannot represent a value that large exactly. The call returned at once, and its tiny
duration was fitted as if the CPU were extremely fast. **Fix:** calls that return `info != 0` are
flagged in the trace and excluded from the fit.

**The CPU was sometimes hundreds of times slower.** Raw repetitions from the CPU pass of run 2:

| Point | Time of each repetition |
| --- | --- |
| complex128 LU, n = 256 | 134.4, 33.2, 29.2, 2.1, 22.0, 11.3, 15.9, 11.5, 506.9 ms |
| complex128 LU, n = 512 | 831.7, 1105.7, 1281.1 ms |
| complex128 LU, n = 1024 | 1251.6, 1263.7, 1005.1 ms |
| complex128 `herk`, n = 1024 | 94.0, 129.7, 141.6 ms |
| float64 LU, n = 512 (a quiet moment, same pass) | 4.2, 3.5, 2.8, 2.9, 2.6, 2.1, 1.9, 2.5, 1.9, 1.9, 2.3, 2.2 ms |

A separate re-measurement of the same kind of points in a quiet moment gave about 1 ms for a
complex64 LU at n = 256 and 2 to 3 ms at n = 512. The slow episodes lasted tens of seconds and
coincided with load from the other virtual machines; a spinning multithreaded library suffers
badly when its cores are taken away. **Fixes:**

1. The calibration workload times a small fixed CPU job (a "canary") before and after every point,
   waits until it runs at normal speed, and repeats a point if the canary slowed down meanwhile.
2. The fit uses the lower quartile of the repetitions instead of the median.
3. At run time, calls are periodically sent to the CPU on purpose to re-measure its speed, and
   corrections move quickly (halfway in ratio terms per observation).

**GPU timings, by contrast, were stable** to within a few percent, apart from a one-off warm-up
on the first call of each routine:

| Point | Time of each repetition (GPU compute) |
| --- | --- |
| complex64 LU, n = 256 | 83.40, 0.51, 0.51, 0.50, 0.51, 0.52, 0.51, 0.50, 0.53, 0.52, 0.49, 0.51 ms |
| float64 LU, n = 1024 | 261.52, 5.24, 5.24, 5.23, 5.24, 5.23 ms |
| float64 linear solve, n = 2048 | 18.47, 18.57, 19.12, 18.47, 18.44, 22.59, 18.44, 18.45, 19.14, 18.54, 18.47, 18.50 ms |

**A CPU-library quirk worth knowing.** OpenBLAS's `dgesv` at n = 2048 took 64 to 80 ms, while its
`dgetrf` (the factorisation that makes up nearly all of a solve) typically took 17 to 23 ms at the
same size. The GPU needs 18.5 ms. So the profile correctly says the GPU wins `gesv` at n = 2048 and not
`getrf`. A hand-written rule would not know this.

### Final profile (run 3)

Fitted parameters: GPU start-up 0.36 s, host-to-GPU copies 9.7 GB/s, GPU-to-host copies 7.8 GB/s.
(Start-up measured between 0.33 and 1.86 s across all runs, depending on whether the CUDA libraries
were already in the file cache.)

GPU speedup over the CPU for square problems, copies included. Above 1 the GPU is faster. "GPU
wins from" is the smallest calibrated size with at least the 1.3x that level 5 requires. A dash
means the point was not measured (a single call exceeded the time limit, or the CPU call failed).

| Operation | Precision | n=256 | n=512 | n=1024 | n=2048 | n=4096 | n=8192 | GPU wins from |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Matrix product (`gemm`) | float32 | 0.3x | 0.5x | 0.8x | 1.4x | 2.7x | 4.7x | n >= 2048 |
| Matrix product (`gemm`) | float64 | 0.3x | 0.4x | 0.4x | 0.5x | 0.6x | 0.8x | never |
| Matrix product (`gemm`) | complex64 | 0.5x | 0.9x | 2.3x | 3.1x | 3.6x | 5.5x | n >= 1024 |
| Matrix product (`gemm`) | complex128 | 0.3x | 0.5x | 0.5x | 0.6x | 0.7x | 0.8x | never |
| Symmetric rank-k update (`syrk`) | float32 | 0.2x | 0.4x | 0.6x | 0.9x | 1.5x | 2.8x | n >= 4096 |
| Symmetric rank-k update (`syrk`) | float64 | 0.3x | 0.3x | 0.3x | 0.4x | 0.5x | 0.7x | never |
| Symmetric rank-k update (`syrk`) | complex64 | 0.3x | 0.5x | 1.6x | 1.6x | 2.7x | 4.5x | n >= 1024 |
| Symmetric rank-k update (`syrk`) | complex128 | 0.2x | 0.2x | 0.4x | 0.5x | 0.6x | 0.8x | never |
| Hermitian rank-k update (`herk`) | complex64 | 0.3x | 0.6x | 0.8x | 1.6x | 3.0x | 5.1x | n >= 2048 |
| Hermitian rank-k update (`herk`) | complex128 | 0.2x | 0.2x | 0.3x | 0.5x | 0.6x | 0.9x | never |
| Triangular solve (`trsm`) | float32 | 0.3x | 0.4x | 0.6x | 0.9x | 1.5x | 2.9x | n >= 4096 |
| Triangular solve (`trsm`) | float64 | 0.1x | 0.2x | 0.3x | 0.4x | 0.4x | 0.6x | never |
| Triangular solve (`trsm`) | complex64 | 0.4x | 0.5x | 0.9x | 1.4x | 2.2x | 3.9x | n >= 2048 |
| Triangular solve (`trsm`) | complex128 | 0.1x | 0.2x | 0.4x | 0.4x | 0.6x | 0.9x | never |
| LU factorisation (`getrf`) | float32 | 1.1x | 1.2x | 1.1x | 1.1x | 1.3x | 2.4x | n >= 4096 |
| LU factorisation (`getrf`) | float64 | 1.1x | 2.0x | 0.9x | 0.8x | 0.9x | 0.8x | n >= 512 |
| LU factorisation (`getrf`) | complex64 | 3.5x | 2.9x | 1.9x | 1.4x | 2.1x | 3.0x | n >= 256 |
| LU factorisation (`getrf`) | complex128 | 0.7x | 0.5x | 0.4x | 0.6x | 0.7x | 0.8x | never |
| Linear solve (`gesv`) | float32 | 0.4x | 0.7x | 1.7x | 3.2x | 6.4x | 11.9x | n >= 1024 |
| Linear solve (`gesv`) | float64 | 0.3x | 0.6x | 1.1x | 2.1x | 0.9x | 1.4x | n >= 2048 |
| Linear solve (`gesv`) | complex64 | 0.7x | 1.6x | 3.2x | 6.2x | 11.9x | 19.1x | n >= 512 |
| Linear solve (`gesv`) | complex128 | 0.4x | 0.8x | 1.5x | 2.7x | 0.7x | 0.8x | n >= 1024 |
| Cholesky factorisation (`potrf`) | float32 | 0.6x | 0.4x | 0.6x | 0.6x | 0.7x | 1.2x | never |
| Cholesky factorisation (`potrf`) | float64 | 0.2x | 0.3x | 0.9x | 2.1x | 0.9x | 1.4x | n >= 2048 |
| Cholesky factorisation (`potrf`) | complex64 | 0.7x | 0.7x | 0.6x | 0.7x | 1.1x | 1.8x | n >= 8192 |
| Cholesky factorisation (`potrf`) | complex128 | 0.2x | 0.2x | 0.2x | 0.3x | 0.5x | 0.7x | never |
| Positive-definite solve (`posv`) | float32 | 0.3x | 0.4x | 0.6x | 0.7x | 0.9x | 1.3x | never |
| Positive-definite solve (`posv`) | float64 | 0.2x | 1.4x | 1.0x | 9.2x | 1.2x | 1.7x | n >= 512 |
| Positive-definite solve (`posv`) | complex64 | 0.4x | 0.5x | 0.6x | 0.8x | 1.4x | 2.7x | n >= 4096 |
| Positive-definite solve (`posv`) | complex128 | 0.2x | 0.3x | 0.3x | 0.4x | 0.5x | 0.7x | never |
| Eigenvalues only | float32 | 1.0x | 1.7x | 2.5x | 2.7x | 3.5x | 5.6x | n >= 512 |
| Eigenvalues only | float64 | 0.3x | 1.3x | 1.4x | 6.7x | 4.0x | 6.1x | n >= 512 |
| Eigenvalues only | complex64 | 2.4x | 5.8x | 20.0x | 11.1x | 4.7x | 8.9x | n >= 256 |
| Eigenvalues only | complex128 | 0.6x | 1.0x | 1.4x | 1.7x | 3.4x | 6.8x | n >= 1024 |
| Eigenvalues and eigenvectors | float32 | 1.8x | 2.9x | 5.4x | 7.2x | 8.9x | 10.2x | n >= 256 |
| Eigenvalues and eigenvectors | float64 | 0.7x | 1.3x | 2.3x | 3.3x | 3.7x | 6.2x | n >= 512 |
| Eigenvalues and eigenvectors | complex64 | 2.0x | 3.7x | 8.3x | 11.7x | - | - | n >= 256 |
| Eigenvalues and eigenvectors | complex128 | 0.8x | 1.1x | 1.7x | 4.5x | 19.1x | - | n >= 1024 |
| Singular values only | float32 | 0.3x | 0.4x | 0.8x | 1.4x | 5.1x | 16.8x | n >= 2048 |
| Singular values only | float64 | 0.3x | 0.5x | 0.9x | 1.9x | 6.9x | 21.1x | n >= 2048 |
| Singular values only | complex64 | 0.3x | 0.6x | 1.3x | 2.6x | 12.4x | 21.9x | n >= 1024 |
| Singular values only | complex128 | 0.4x | 0.6x | 0.8x | 2.4x | 9.3x | - | n >= 2048 |
| SVD with vectors | float32 | 1.9x | 0.9x | 1.1x | 3.2x | 3.0x | 6.2x | n >= 256 |
| SVD with vectors | float64 | 0.3x | 0.5x | 0.9x | 0.8x | 2.7x | - | n >= 4096 |
| SVD with vectors | complex64 | 0.3x | 0.6x | 1.1x | 1.9x | 3.4x | 5.5x | n >= 2048 |
| SVD with vectors | complex128 | 0.2x | 0.3x | 0.5x | 0.7x | 2.0x | - | n >= 4096 |

**Reading the table.**

- The broad picture is robust: single precision and complex-single win from moderate sizes; double
  precision wins for eigenproblems and singular values, and loses for products, triangular solves
  and (mostly) factorisations.
- A few isolated cells still look odd even in run 3 (float64 `posv` at n = 2048: 9.2x; complex64
  eigenvalues at n = 1024: 20.0x; float64 `potrf` at n = 2048: 2.1x). These are most likely
  remaining CPU slow-downs during those points. The run-time CPU probes exist to correct exactly
  this.
- Compare experiment 1: float32 `gemm` at n = 8192 is 4.7x here and 5.9x there. The two used
  different programs and different days; that is the size of the measurement uncertainty.

## Experiment 6: correctness

**Method.** `tests/test_gpu_ops.c` calls every offloaded operation in all four precisions, with
transposed and conjugated operands, non-contiguous leading dimensions, row-major CBLAS calls,
`beta != 0`, both triangles, unit and non-unit diagonals, rectangular LU, and SVD in three job
modes. Each result is checked against a reference computed in the test itself with plain loops in
double-complex arithmetic: products, residuals `A x - b`, reconstructions `P L U = A`,
`U^H U = A`, `A v = lambda v`, `U S V^T = A`, and that the triangle a routine must not touch is
untouched.

| Run | Result |
| --- | --- |
| Plain (CPU libraries only) | 168 checks, 0 failed |
| Forced to the GPU | 168 checks, 0 failed; 108 calls ran on the GPU, 4 fell back to the CPU as designed (SVD of a wide matrix, which the GPU path does not handle) |
| Verify mode (both devices, results compared) | 128 comparisons including nested calls, 0 mismatches, largest relative difference 1.23e-06 |
| GPU backend missing | 168 checks, 0 failed; every call reported "GPU unavailable" and ran on the CPU |
| Synthetic profile that makes the GPU look fast, then slow | Calls go to the GPU, then stay on the CPU, with the expected reason codes |
| Measure mode on a second test program | The trace contains exactly the calls the program makes, with correct sizes, buffer roles, workspace-query flags and FFTW geometry |

The whole suite passed on its first GPU run and on every run since. It runs with `./build.sh test`.

**Not covered by tests:** behaviour when another program is loading the GPU, behaviour when GPU
memory runs out, multi-threaded callers offloading at the same time, and `fork()` after GPU start.
The code paths exist; they have not been exercised by an experiment.

## Experiment 7: end-to-end speedups

**Method.** Each workload was run plain and under `gpu-offload run`, using the run-3 profile. The
tool compared the output of both runs (text equal, numbers equal to 3 significant digits).

### Single runs at level 5, all thirteen workloads

Offloaded runs here also traced every call to disk.

| Workload | Plain | Offloaded | Speedup | Calls on the GPU | Output |
| --- | --- | --- | --- | --- | --- |
| NumPy linear algebra, n = 3000, float64 | 7.51 s | 6.23 s | 1.21x | 1 | same |
| NumPy linear algebra, n = 3000, float32 input | 6.11 s | 4.45 s | 1.37x | 1 | same |
| NumPy linear algebra, n = 6000, float64 | 32.82 s | 16.56 s | 1.98x | 1 | same |
| NumPy linear algebra, n = 6000, float32 input | 22.45 s | 11.18 s | 2.01x | 1 | same |
| NumPy neural-net training, float32 | 2.55 s | 2.60 s | 0.98x | 0 | same |
| NumPy FFT signal processing | 0.71 s | 0.72 s | 0.99x | 0 | same |
| scikit-learn pipeline | 6.43 s | 8.66 s | 0.74x | 0 | same |
| pandas data pipeline | 9.94 s | 7.96 s | 1.25x | 0 | same |
| Octave FFT + linear algebra | 2.38 s | 2.27 s | 1.05x | 0 | same |
| R regression script | 2.66 s | 2.53 s | 1.05x | 0 | same |
| PyPI NumPy linear algebra, n = 6000, float64 | 19.42 s | 11.75 s | 1.65x | 1 | same |
| PyPI NumPy linear algebra, n = 6000, float32 input | 19.09 s | 11.18 s | 1.71x | 1 | same |
| PyPI scikit-learn pipeline | 11.05 s | 9.40 s | 1.18x | 0 | same |

**The noise is visible in this table.** The pandas (1.25x) and scikit-learn (0.74x, 1.18x) rows
offloaded nothing, so their differences are pure run-to-run variation of a shared machine: about
plus or minus 25%. Single runs cannot be trusted, hence the next table.

### Medians of 3 runs at levels 5 and 9

Runs were interleaved (plain, level 5, level 9, repeated three times) so that drift in machine
load affects all variants alike. No tracing during timed runs.

| Workload | Plain | Level 5 | Level 9 | Output |
| --- | --- | --- | --- | --- |
| NumPy linear algebra, n = 3000, float64 | 4.18 s | 3.83 s (1.09x) | 3.90 s (1.07x) | same |
| NumPy linear algebra, n = 3000, float32 input | 3.92 s | 3.89 s (1.01x) | 3.88 s (1.01x) | same |
| NumPy linear algebra, n = 6000, float64 | 20.76 s | 13.53 s (1.53x) | 12.30 s (1.69x) | same |
| NumPy linear algebra, n = 6000, float32 input | 19.06 s | 11.90 s (1.60x) | 11.50 s (1.66x) | same |
| NumPy neural-net training, float32 | 5.72 s | 5.46 s (1.05x) | 4.55 s (1.26x) | same |
| NumPy FFT signal processing | 0.66 s | 0.72 s (0.92x) | 0.86 s (0.77x) | same |
| scikit-learn pipeline | 10.83 s | 8.21 s (1.32x) | 9.07 s (1.19x) | same |

**This benchmark is incomplete.** It was stopped after seven of the thirteen workloads because the
Windows host ran low on memory (3.6 GB free of 125.7 GB; the Hyper-V runner VMs held about 91 GB
at that moment). The remaining six workloads (pandas, Octave, R and the three PyPI variants) have
single-run results only, in the table above. Contributions of complete, repeated runs on quiet
machines are very welcome.

**Notes on individual rows.**

- **n = 6000 rows.** The gain comes from one call, `numpy.linalg.eigh` (LAPACK `dsyevd`). Details
  below.
- **"float32 input" rows behave like float64 rows** because NumPy's `linalg` functions convert
  float32 input to float64, call the double-precision LAPACK routine, and convert the result back.
  The trace shows `dsyevd`, `dgesv` and `dgesdd` for a float32 script. Only `@` (matrix product)
  stays in float32.
- **NumPy FFT, 0.92x and 0.77x.** Nothing is intercepted in this 0.7 s program. The difference was
  start-up time of the `gpu-offload` command itself, which at that point imported NumPy just to
  check that a profile existed. That import has since been removed from `gpu-offload run`.
- **scikit-learn and neural-net rows.** No call was offloaded at level 5; the differences are
  noise. Whether the 1.26x at level 9 on the neural-net workload is a real gain from offloading
  borderline products was not investigated.

### Anatomy of the n = 6000 result

From the traced run of `np_linalg.py 6000 float64` at level 5:

| Quantity | Value |
| --- | --- |
| GPU-capable calls made by the application | 77 |
| Sent to the GPU | 1 (`dsyevd`, n = 6000, with eigenvectors) |
| Stayed on the CPU, predicted gain too small | 75 to 76 (the linear solves at 6000 x 8, the 4000 x 2000 SVD with vectors, and the products) |
| CPU time predicted for the offloaded call | 11.10 s |
| GPU time predicted, copies included | 2.23 s |
| GPU time measured | 2.27 to 2.30 s across four runs, of which 0.06 to 0.07 s copying |
| Exit summary of that run, nested calls included | 6 calls on the GPU taking 2.30 s; estimated saving 8.46 s |

The cost model's GPU prediction was within 2 to 3 percent of the measured time. At n = 3000 it
predicted 0.397 s and the call took 0.450 s and 0.590 s in two runs.

**Why the solves stayed on the CPU.** At 6000 x 6000 the profile's float64 `gesv` speedup is
between 0.9x (n = 4096) and 1.4x (n = 8192), below the 1.3x that level 5 requires. That is the
cost model doing its job, not a missing feature.

## Experiment 8: packaging and installation

| Check | Result |
| --- | --- |
| `./build.sh package` | Produces `gpu-offload_0.1.1_amd64.deb`, installed size 2.4 MB |
| `apt install ./gpu-offload_0.1.1_amd64.deb` in the Ubuntu 24.04 distro | Installs; `gpu-offload` on the PATH |
| `gpu-offload status` from the installed copy | Finds the library, the backend, the GPU and the profile |
| Installed library, test program forced to the GPU | 108 calls on the GPU, 4 on the CPU, all checks pass |
| `gpu-offload run` on a 3000 x 3000 `eigh` script | 1.95 s wall; 1 call on the GPU (0.60 s), 1 on the CPU; estimated saving 0.34 s |

The package has been installed on the development machine only.

## What has not been measured

Listed so that nobody mistakes absence of evidence for a result:

- Any other machine, GPU model, GPU vendor, CPU, or a native (non-WSL) Linux installation.
- MKL, BLIS or other CPU math libraries; only OpenBLAS (system and PyPI-bundled).
- Energy use.
- Behaviour with a second program loading the GPU, or with GPU memory nearly full.
- Programs written in Julia, MATLAB, C++ (Eigen, Armadillo) or Fortran applications.
- Levels other than 5 and 9 on real workloads.
- Long-running services; the longest run was about 35 seconds.

## Reproducing

```bash
./build.sh test                              # correctness (experiment 6) and overhead (experiment 4)
gpu-offload calibrate --full --keep          # experiment 5; --keep preserves the raw traces
gpu-offload status                           # prints your version of the calibration table
python3 workloads/bench.py --reps 3 --levels 5,9    # experiment 7
gpu-offload measure python3 workloads/np_linalg.py  # experiment 3, for one program
```

The workload sources are in [`workloads/`](https://github.com/jaifar530/cpu-to-gpu-offloading/tree/main/workloads).
The raw outputs behind this page (the machine profile, the per-call reports and the timing tables)
are in [`experiments/raw/`](https://github.com/jaifar530/cpu-to-gpu-offloading/tree/main/experiments).
If you run these on your machine, please share the results: see
[Contributing](https://github.com/jaifar530/cpu-to-gpu-offloading/blob/main/CONTRIBUTING.md).
