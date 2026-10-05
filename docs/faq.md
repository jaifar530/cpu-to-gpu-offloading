---
title: FAQ
nav_order: 7
description: >-
  Frequently asked questions about gpu-offload: running CPU code on a GPU without code changes,
  NumPy, R and Octave support, accuracy, safety, performance, WSL2 and how it differs from NVBLAS
  and CuPy.
---

# Frequently asked questions

## What is gpu-offload?

gpu-offload is an open-source Linux tool that runs the heavy linear-algebra calls of unmodified
programs on an NVIDIA GPU when a per-machine calibration predicts that the GPU is faster, and on
the CPU otherwise. It is an `LD_PRELOAD` library plus a command-line tool. It is released under
the GNU AGPL v3, and a commercial licence is available.

## What licence is it under? Can I use it commercially?

gpu-offload is dual-licensed. Under the GNU Affero General Public License v3 you may use, modify
and share it free of charge, including in a company, as long as you follow the AGPL's conditions
(in particular, publishing source code when you distribute it or offer a modified version over a
network). If those conditions do not work for you, for example because you want to ship it inside
a closed-source product, a commercial licence is available. See
[LICENSING.md](https://github.com/jaifar530/cpu-to-gpu-offloading/blob/main/LICENSING.md).

## Can I run CPU code on a GPU without changing the code?

Partly. No tool can move an arbitrary CPU program onto a GPU. What can be moved transparently are
the calls a program makes to standard math libraries (BLAS and LAPACK). gpu-offload intercepts
those calls and runs the large ones on the GPU. Everything else in the program stays on the CPU.

## Do I need to modify or recompile my program?

No. Run it as `gpu-offload run your-command`. The program must use dynamically linked BLAS /
LAPACK, which NumPy, SciPy, R, GNU Octave and most compiled scientific codes do.

## Does it work with NumPy and SciPy?

Yes, with both the distribution packages and the wheels from PyPI. The PyPI wheels bundle their own
OpenBLAS under renamed symbols (`scipy_dgemm_64_` and similar); gpu-offload intercepts those too.
Note that NumPy's `linalg` solvers compute in float64 even when given float32 arrays.

## Does it work with R, Octave, Julia, MATLAB?

R and GNU Octave were tested: their calls are intercepted, and the test scripts ran unchanged.
Julia and MATLAB were not tested. Anything that calls the standard BLAS / LAPACK symbols through a
shared library should work; anything that ships a statically linked or privately named math
library will not be seen.

## How much faster will my program be?

It depends on how much of its time is large dense linear algebra. Measured on the development
machine: 1.5x to 1.7x whole-program for a NumPy script doing eigendecompositions, solves and SVD
at n = 6000, and no change for pandas, FFT, scikit-learn k-means and small neural-network
workloads. Run `gpu-offload measure your-command` to see your program's offloadable share before
anything else. See [Experiments and benchmarks](experiments.md).

## When is the GPU slower than the CPU?

For small problems (copying data costs more than the work), and on consumer GPUs for many
double-precision operations. On an RTX 5080 against a 16-core Ryzen 9, a float64 matrix product
was never faster on the GPU (0.8x at n = 8192), while a float32 one was 4.7x faster. gpu-offload
keeps such calls on the CPU.

## What does calibration do, and how long does it take?

`gpu-offload calibrate` times every supported operation on your CPU and your GPU, in four
precisions and at several sizes, and stores speed curves for both, the cost of copies and the GPU
start-up time. The most thorough setting (`--full`, sizes up to 8192) took 31 to 34 minutes on the
development machine; the default and `--quick` settings take much less. You run it once per
machine, and again after hardware or math-library changes.

## What is the level from 1 to 10?

It is the one setting that controls how eagerly work goes to the GPU. Level 1 requires a predicted
3x speedup and leaves half of the GPU memory free; level 5, the default, requires 1.3x; level 10
offloads even at a predicted small loss to free the CPU. It never affects correctness.

## Are the results the same as on the CPU?

They agree to rounding error, not bit for bit, because the GPU performs floating-point operations
in a different order. In the tests the largest relative difference was 1.2e-06 in single
precision. Eigenvectors and singular vectors can differ in sign, which is mathematically
equivalent. Use `gpu-offload run --verify your-command` to compare GPU and CPU results on your own
data.

## What happens if the GPU fails or runs out of memory?

The call runs on the CPU. Results are written into the program's memory only after the GPU work
has succeeded, so a failed GPU attempt leaves the inputs untouched for the CPU.

## Does it slow down programs that have nothing to offload?

Barely. A call that stays on the CPU costs about 15 nanoseconds extra, and the CUDA libraries are
never loaded if the program does too little heavy math to repay the GPU start-up cost.

## Does it work on Windows or macOS?

It works on Windows through WSL2, which is how it was developed. There is no native Windows or
macOS version.

## Does it support AMD or Intel GPUs?

Not yet. The only backend uses NVIDIA's cuBLAS and cuSOLVER. The backend is a separate library
behind a small interface, so another one can be added; contributions are welcome.

## How is it different from NVBLAS?

Both are drop-in libraries loaded with `LD_PRELOAD`. NVBLAS covers level-3 BLAS. gpu-offload also
covers LAPACK solvers (linear solves, eigenproblems, SVD), decides from a cost model measured on
your machine instead of fixed settings, includes copy costs in the decision, and adapts while the
program runs. See [Comparison](comparison.md).

## How is it different from CuPy?

CuPy is a GPU array library: you change your code to use it, and data stays on the GPU, which is
faster. gpu-offload needs no code change and works on programs you cannot modify, but copies data
to and from the GPU on every offloaded call.

## Is it safe to use in production?

Not yet. It is version 0.1, tested on one machine. It is suitable for experiments, for workstation
use on your own computations, and as a base for further development.

## Is `LD_PRELOAD` a security risk?

`LD_PRELOAD` loads a library into the processes you start with it, which is exactly what this tool
needs and also what makes it unsuitable for security-sensitive or instrumentation-averse programs.
gpu-offload is opt-in per command or per shell and does not install itself system-wide.

## Why does it copy data on every call? Would keeping it on the GPU not be faster?

Yes, it would. The measurement tool estimates that 27% to 73% of the copy volume in the test
workloads could be avoided. Doing it safely is hard: the tool must know when the CPU modifies or
reads data that currently lives on the GPU. It is the first item on the roadmap.

## Can I contribute?

Yes, please. Code, calibration results from other machines, bug reports and documentation are all
welcome. See [CONTRIBUTING.md](https://github.com/jaifar530/cpu-to-gpu-offloading/blob/main/CONTRIBUTING.md).

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    {"@type": "Question", "name": "What is gpu-offload?",
     "acceptedAnswer": {"@type": "Answer", "text": "gpu-offload is an open-source Linux tool that runs the heavy linear-algebra calls (BLAS and LAPACK) of unmodified programs on an NVIDIA GPU when a per-machine calibration predicts that the GPU is faster, and on the CPU otherwise. It is an LD_PRELOAD library plus a command-line tool, released under the GNU AGPL v3, with a commercial licence available."}},
    {"@type": "Question", "name": "Can I run CPU code on a GPU without changing the code?",
     "acceptedAnswer": {"@type": "Answer", "text": "Partly. No tool can move an arbitrary CPU program onto a GPU, but the calls a program makes to standard math libraries (BLAS and LAPACK) can be intercepted and run on the GPU transparently. gpu-offload does that for large calls and leaves everything else on the CPU."}},
    {"@type": "Question", "name": "Does gpu-offload work with NumPy and SciPy?",
     "acceptedAnswer": {"@type": "Answer", "text": "Yes, with both distribution packages and PyPI wheels, including the renamed OpenBLAS symbols bundled in the wheels. No code changes are needed: run gpu-offload run python3 script.py."}},
    {"@type": "Question", "name": "How much faster does gpu-offload make a program?",
     "acceptedAnswer": {"@type": "Answer", "text": "It depends on the share of large dense linear algebra. Measured on a Ryzen 9 9950X3D with an RTX 5080: 1.5x to 1.7x whole-program for NumPy eigendecompositions, solves and SVD at n = 6000, and no change for pandas, FFT, k-means and small neural-network workloads."}},
    {"@type": "Question", "name": "When is a GPU slower than a CPU for linear algebra?",
     "acceptedAnswer": {"@type": "Answer", "text": "For small problems, where copying data costs more than the work, and on consumer GPUs for many double-precision operations. On an RTX 5080 against a 16-core Ryzen 9 9950X3D a float64 matrix product was never faster on the GPU, while a float32 one was 4.7x faster at n = 8192."}},
    {"@type": "Question", "name": "Are gpu-offload results identical to CPU results?",
     "acceptedAnswer": {"@type": "Answer", "text": "They agree to rounding error, not bit for bit. The largest relative difference in the tests was 1.2e-06 in single precision. A verify mode runs each call on both devices and compares the results."}},
    {"@type": "Question", "name": "How is gpu-offload different from NVBLAS?",
     "acceptedAnswer": {"@type": "Answer", "text": "Both are LD_PRELOAD drop-in libraries. NVBLAS covers level-3 BLAS. gpu-offload also covers LAPACK solvers, decides from a cost model calibrated on the machine instead of fixed settings, includes copy costs in the decision, and adapts at run time."}},
    {"@type": "Question", "name": "Does gpu-offload work on Windows?",
     "acceptedAnswer": {"@type": "Answer", "text": "It works on Windows through WSL2 with an NVIDIA GPU. There is no native Windows or macOS version."}}
  ]
}
</script>
