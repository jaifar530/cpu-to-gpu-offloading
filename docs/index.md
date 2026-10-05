---
title: Transparent CPU-to-GPU offloading
nav_order: 1
description: >-
  gpu-offload transparently runs the BLAS and LAPACK calls of unmodified programs (NumPy, R,
  Octave, C, Fortran) on an NVIDIA GPU when a per-machine calibration predicts a speedup, and
  falls back to the CPU otherwise. Open source under the AGPL v3, with a commercial licence available.
permalink: /
---

# gpu-offload: transparent CPU-to-GPU offloading

**gpu-offload runs the heavy math of unmodified programs on your GPU, but only when your GPU is
actually faster for that piece of work.** You do not change, recompile or even restart your code
differently; you put `gpu-offload run` in front of the command.

```bash
gpu-offload calibrate                  # once per machine: learn where the GPU beats the CPU
gpu-offload run python3 my_script.py   # any program that uses BLAS / LAPACK
```

It is an `LD_PRELOAD` library for Linux that sits between a program and its math libraries
(OpenBLAS, the reference BLAS / LAPACK, the OpenBLAS bundled in NumPy and SciPy wheels). For each
heavy call it predicts the CPU time and the GPU time on *this* machine, including the cost of
copying data, and sends the call to cuBLAS / cuSOLVER only when the prediction shows a gain. If the
GPU path cannot handle a call or fails, the CPU library runs it.

## In one minute

| Question | Answer |
| --- | --- |
| What does it accelerate? | Dense linear algebra: matrix products, triangular solves, LU, Cholesky, linear solves, symmetric eigenproblems, SVD, in float32, float64, complex64 and complex128. |
| What do I have to change in my program? | Nothing. No source changes, no recompilation, no different imports. |
| How does it know when the GPU is faster? | `gpu-offload calibrate` measures every operation on your CPU and your GPU over a range of sizes and stores a machine profile. Decisions are made per call from that profile and corrected at run time. |
| What if the GPU would be slower? | The call stays on the CPU. On the development machine a float64 matrix product never goes to the GPU, because it is never faster there. |
| How much faster do programs get? | Measured: 1.5x to 1.7x whole-program on large NumPy linear algebra (n = 6000), nothing on workloads without large dense math. See [Experiments and benchmarks](experiments.md). |
| What does it need? | Linux x86-64 (WSL2 works), an NVIDIA GPU, the CUDA 13 runtime libraries, Python 3 with NumPy. |
| Is it production ready? | No. It is a working, tested prototype (v0.1). See [When to use it](when-to-use.md). |

## Why the decision must be learned per machine

Measured on the development machine (AMD Ryzen 9 9950X3D, NVIDIA GeForce RTX 5080), GPU speedup
over the CPU with copies included:

| Operation | Precision | n = 1024 | n = 4096 | n = 8192 |
| --- | --- | --- | --- | --- |
| Matrix product | float32 | 0.8x | 2.7x | 4.7x |
| Matrix product | float64 | 0.4x | 0.6x | 0.8x |
| Eigenvalues and eigenvectors | float32 | 5.4x | 8.9x | 10.2x |
| Eigenvalues and eigenvectors | float64 | 2.3x | 3.7x | 6.2x |
| Singular values | float64 | 0.9x | 6.9x | 21.1x |

The same GPU is nearly five times faster, or slower, than the same CPU depending on the operation,
the precision and the size. A fixed rule such as "offload matrices larger than 500" is wrong for
half of this table. gpu-offload measures instead of assuming.

## Where to go next

- [Install and use](usage.md): installation, every command, every setting, troubleshooting.
- [How it works](how-it-works.md): interception, the cost model, the 1 to 10 level, run-time learning.
- [When to use it](when-to-use.md): where it is excellent, where it does nothing, where it hurts.
- [Experiments and benchmarks](experiments.md): every measurement made while building it, including the ones that went wrong.
- [Comparison](comparison.md): NVBLAS, SCILIB-Accel, CuPy, cudf.pandas and others.
- [FAQ](faq.md): short answers to common questions.
- [Contributing](https://github.com/jaifar530/cpu-to-gpu-offloading/blob/main/CONTRIBUTING.md): the project is open to contributions.
