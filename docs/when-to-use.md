---
title: When to use it
nav_order: 4
description: >-
  Where gpu-offload gives a real speedup, where it does nothing, and where it should not be used:
  workload types, matrix sizes, precision, hardware, and a checklist to decide in five minutes.
---

# When gpu-offload is excellent, when it does nothing, and when not to use it

Short version: **it pays when a program spends most of its time in a few large dense
linear-algebra calls, and it does nothing for everything else.** It is built to do nothing rather
than make things slower, but there are cases where you should not use it at all.

## Decide in five minutes

```bash
gpu-offload measure  your-command     # how much of the run time is offloadable math?
gpu-offload calibrate                 # once: where does this machine's GPU win?
gpu-offload estimate TRACE_DIR        # predicted saving at every level
gpu-offload run --verify your-command # do GPU and CPU results agree on your data?
```

If `measure` shows less than about a third of the run time in calls of 1 ms or longer, stop here:
no offloading tool can help much, because the rest of the program is not offloadable.

## Excellent fit

| Situation | Why it works | Evidence from the experiments |
| --- | --- | --- |
| Large symmetric eigenproblems (`numpy.linalg.eigh`, `eigvalsh`, PCA via eigendecomposition), n from about 1000 up | The GPU solver is several times faster even in float64, and the copy is small compared with the work | eigenvectors, float64: 2.3x at n=1024, 3.7x at n=4096, 6.2x at n=8192; `eigh` at n=6000 took 2.27 s on the GPU against a predicted 11.1 s on the CPU |
| Large singular value problems (values only) | Same reason | float64: 6.9x at n=4096, 21.1x at n=8192 |
| Single-precision or complex-single dense algebra, n from about 2000 up | Consumer GPUs are strong in float32 | float32 matrix product 2.7x at n=4096 and 4.7x at n=8192; float32 linear solve 6.4x at n=4096; complex64 linear solve 11.9x at n=4096 |
| Programs you cannot or do not want to modify | No source, no build system, a vendor binary, a script you only run | Works on unmodified NumPy, R, Octave programs; no code was changed in any workload |
| Machines where the CPU math library is weak or badly configured | The profile reflects the CPU as it really behaves | With OpenBLAS's default thread count the same workloads ran 15 to 25 times slower on the CPU |
| Mixed workloads on a shared workstation | Small calls stay on the CPU, the GPU is only started when it pays, and it backs off when other programs use the GPU | Programs with nothing to offload ran unchanged |

Expect whole-program gains of the order measured here: **1.5x to 1.7x** on linear-algebra-heavy
scripts at n=6000 (median of 3 runs). The gain is capped by the share of the program that is
offloadable math.

## Works, but with little or no gain

| Situation | Why | What you will see |
| --- | --- | --- |
| Double-precision matrix products, triangular solves, Cholesky on a consumer GPU with a strong CPU | Consumer GPUs are slow in float64; a modern 16-core CPU matches or beats them | "stayed on the CPU, predicted gain too small"; float64 `gemm` was never faster on the RTX 5080 (0.8x at n=8192) |
| Small and medium problems (roughly n below 1000) | Copying data and launching GPU kernels costs more than the work | Calls stay on the CPU |
| Many tiny calls (scikit-learn k-means made 467,664 products of size 16 x 256 x 30; small neural-network layers) | No single call is large enough; the work is in the number of calls | Nothing offloaded |
| Short programs | The GPU start-up cost (0.3 to 1.9 s measured) is never repaid | "GPU start-up not yet worth it" |
| Data pipelines (pandas), FFT with NumPy, text processing, I/O-bound work | They do not call BLAS / LAPACK | 0% of run time interceptable in the pandas and NumPy-FFT workloads |
| R's `lm()`, element-wise array math | Level-1 BLAS on long vectors and plain loops: too little work per byte moved | Not offloaded |

In these cases gpu-offload costs almost nothing (about 15 ns per intercepted call that stays on
the CPU) and changes nothing. A higher level will offload more of the borderline calls for a small
gain or loss.

## Do not use it

| Situation | Reason |
| --- | --- |
| You need bit-identical results to a CPU run | GPU floating-point operations run in a different order. Results agree to rounding (largest relative difference 1.2e-06 in float32 in the tests), not bit for bit. Eigenvectors and singular vectors may differ in sign. |
| Safety-critical, regulated or otherwise high-assurance computation | It is a v0.1 prototype tested on one machine. |
| Production services | No system-wide daemon, no persistence of what it learns between runs, opt-in per command. |
| Programs you can easily port to a GPU library | CuPy, PyTorch, RAPIDS or JAX keep data on the GPU between operations. gpu-offload copies data in and out on every call, so a native GPU program is faster. |
| Programs that `fork()` workers after doing GPU work | The children stay on the CPU (a CUDA context does not survive `fork()`). Harmless, but there is no gain in the workers. |
| Security-sensitive processes, set-uid programs, anti-cheat or DRM-protected software | It injects a library with `LD_PRELOAD`. Do not put it into processes that must not be instrumented. |
| No NVIDIA GPU, or macOS / native Windows | Only a CUDA backend exists, on Linux (WSL2 included). |
| Problems larger than GPU memory | They fall back to the CPU; there is no out-of-core path. |

## Hardware changes the answer

The table that matters is the one `gpu-offload status` prints on **your** machine. Two examples of
how it shifts:

- **A data-centre GPU with full-speed float64** (instead of a consumer card) would move the
  double-precision rows from "never" to a clear win.
- **A laptop CPU** (instead of a 16-core desktop CPU) lowers every CPU curve, so the GPU wins at
  smaller sizes.

Neither was measured here. Calibrating is how you find out.

## Known rough edges

- Calibration on a busy machine is noisy. The calibrator waits for quiet moments and repeats
  disturbed points, and the run-time CPU probes correct what remains, but a profile taken while
  other heavy jobs run will be less accurate. Calibrate when the machine is otherwise idle.
- Timings of whole programs varied by 20 to 30 percent between runs on the development machine,
  which was shared with other virtual machines. Compare medians of several runs.
- The first GPU call of each kind includes a one-off warm-up inside the CUDA libraries (83 to
  262 ms was observed). Long-running programs do not notice; very short ones do.
