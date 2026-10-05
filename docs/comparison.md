---
title: Comparison
nav_order: 6
description: >-
  How gpu-offload compares with NVBLAS, SCILIB-Accel, CuPy, cudf.pandas, cuml.accel, RAPIDS,
  OpenACC and other ways of using a GPU from existing code, and which one to choose.
---

# gpu-offload compared with other ways to use a GPU

There are many ways to get existing code onto a GPU. They differ in three things: how much you
must change, what they cover, and who decides when the GPU is used.

## At a glance

| Approach | Change needed in your program | What it covers | Who decides CPU or GPU | Data stays on the GPU between operations |
| --- | --- | --- | --- | --- |
| **gpu-offload** | None | Dense BLAS level 3 and LAPACK calls of any dynamically linked program | A cost model fitted per machine by calibration, corrected at run time, with one 1-to-10 setting | No (copies per call) |
| NVBLAS (NVIDIA) | None (`LD_PRELOAD` and a configuration file) | Level-3 BLAS | The library, per call, from the call's characteristics and its configuration | No |
| SCILIB-Accel (TACC) | None (`LD_PRELOAD`) | Level-3 BLAS, including statically linked symbols | A matrix-size threshold | Yes, on unified-memory systems it was designed for |
| CuPy | Replace `numpy` with `cupy` in your code | Most of the NumPy and SciPy array API | You, in the code | Yes |
| cudf.pandas, cuml.accel (RAPIDS) | One flag or import line | pandas; scikit-learn, UMAP, HDBSCAN | The library: GPU where an operation is supported, CPU otherwise | Yes, within the library |
| PyTorch, JAX, TensorFlow | Write the program for the framework | Whatever you write | You (device placement) | Yes |
| OpenACC, OpenMP target, `-stdpar` | Annotate or structure loops, then recompile | Loops the compiler can offload | Mostly you, through directives | Depends on the directives |
| CUDA, SYCL, OpenCL | Rewrite the hot code | Anything | You | Yes |

Entries for other projects summarise their public documentation as understood at the time of
writing (October 2026); check the projects themselves for current details.

## What is different about gpu-offload

1. **The decision is learned, not configured.** Thresholds in other drop-in tools are constants
   someone chose. Here `gpu-offload calibrate` measures each operation, precision and size on the
   actual CPU and GPU. On the development machine that correctly keeps every float64 matrix product
   on the CPU while sending float64 eigenproblems to the GPU, a distinction no size threshold
   expresses.
2. **Copies are part of the prediction.** The comparison is CPU time against GPU time *including*
   moving the data.
3. **It covers LAPACK solvers, not only BLAS.** Eigenvalues, SVD and linear solves were where the
   measured gains came from; on this hardware level-3 BLAS alone would have gained little in
   double precision.
4. **It keeps learning.** Observed timings correct the predictions, and the CPU is re-measured
   periodically.
5. **It avoids costs it cannot repay.** The CUDA libraries are not even loaded until the missed
   savings exceed the GPU start-up cost.
6. **One user-facing setting**, a level from 1 to 10, like a swappiness knob for the GPU.

## What the others do better

- **Anything that keeps data on the GPU** (CuPy, PyTorch, RAPIDS, hand-written CUDA) avoids the
  per-call copies that limit gpu-offload. If you can port your program, port it: it will be faster.
- **cudf.pandas and cuml.accel** accelerate whole library operations (a group-by, a k-means fit).
  gpu-offload sees only the BLAS calls inside them, which for k-means are hundreds of thousands of
  tiny products that cannot be offloaded one by one.
- **SCILIB-Accel** intercepts statically linked BLAS symbols through binary instrumentation;
  gpu-offload only sees calls that cross a shared-library boundary.
- **Vendor-supported products** are tested on many systems. gpu-offload v0.1 has been tested on
  one.

## Which should I use?

| Your situation | Suggestion |
| --- | --- |
| You can change the code and it is NumPy-style array code | CuPy |
| The program is pandas or scikit-learn | cudf.pandas / cuml.accel |
| It is a deep-learning workload | The framework's own GPU support |
| You cannot or do not want to change the program, and it does large dense linear algebra | gpu-offload |
| You want to know whether a GPU could help an existing program at all | `gpu-offload measure`, which works without a GPU |
| You need a supported, production-grade tool | A vendor product |
