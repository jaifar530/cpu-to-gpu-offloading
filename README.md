# gpu-offload: transparent CPU-to-GPU offloading for unmodified programs

[![CI](https://github.com/jaifar530/cpu-to-gpu-offloading/actions/workflows/ci.yml/badge.svg)](https://github.com/jaifar530/cpu-to-gpu-offloading/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/jaifar530/cpu-to-gpu-offloading)](https://github.com/jaifar530/cpu-to-gpu-offloading/releases)
[![License: AGPL v3 or commercial](https://img.shields.io/badge/license-AGPL%20v3%20%7C%20commercial-blue.svg)](LICENSING.md)
[![Contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Docs](https://img.shields.io/badge/docs-website-informational)](https://jaifar530.github.io/cpu-to-gpu-offloading/)

**Run the heavy math of unmodified programs on your GPU, but only when your GPU is actually
faster for that piece of work.**

`gpu-offload` is an open-source `LD_PRELOAD` library and command-line tool for Linux. It sits
between a program (NumPy, SciPy, R, GNU Octave, C, C++, Fortran) and its math libraries (BLAS and
LAPACK). For every heavy call it predicts how long the CPU and the GPU would take **on this
machine**, including the cost of copying data, runs the call on an NVIDIA GPU through cuBLAS /
cuSOLVER when that is faster, and falls back to the CPU otherwise. No source changes, no
recompilation, no different imports.

```bash
gpu-offload calibrate                  # once per machine: learn where the GPU beats the CPU
gpu-offload run python3 my_script.py   # any program that uses BLAS / LAPACK
```

```
gpu-offload: level 5 on NVIDIA GeForce RTX 5080 (GPU start-up 0.47 s)
  GPU-capable calls: 1 on the GPU (0.60 s), 1 on the CPU
  estimated time saved by offloading: 0.34 s
  stayed on the CPU, predicted gain too small: 1
```

**Status:** v0.1, a working and tested prototype, measured on one machine so far.
**Documentation:** https://jaifar530.github.io/cpu-to-gpu-offloading/
**Contributions are welcome:** see [CONTRIBUTING.md](CONTRIBUTING.md).

## Contents

- [Why it exists](#why-it-exists)
- [Key results](#key-results)
- [When to use it, and when not](#when-to-use-it-and-when-not)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [What is offloaded](#what-is-offloaded)
- [Commands](#commands)
- [Experiments and benchmarks](#experiments-and-benchmarks)
- [Correctness](#correctness)
- [Limitations](#limitations)
- [Comparison with other tools](#comparison-with-other-tools)
- [FAQ](#faq)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [Repository layout](#repository-layout)
- [Citing](#citing)
- [Licence](#licence)

## Why it exists

"GPU acceleration" normally means rewriting code for CUDA or switching to a GPU-specific library.
Existing drop-in tools each cover one library and decide with a fixed rule, such as a matrix-size
threshold.

In practice the right decision depends on the machine, the operation, the precision and the size.
Measured on the development machine (AMD Ryzen 9 9950X3D, NVIDIA GeForce RTX 5080), GPU speedup
over the CPU with copies included:

| Operation | Precision | n = 1024 | n = 4096 | n = 8192 |
| --- | --- | --- | --- | --- |
| Matrix product | float32 | 0.8x | 2.7x | 4.7x |
| Matrix product | float64 | 0.4x | 0.6x | 0.8x |
| Linear solve | float32 | 1.7x | 6.4x | 11.9x |
| Eigenvalues and eigenvectors | float64 | 2.3x | 3.7x | 6.2x |
| Singular values | float64 | 0.9x | 6.9x | 21.1x |

The same GPU is several times faster, or slower, than the same CPU. A fixed rule cannot capture
this table, and the table is different on every machine. So `gpu-offload` **measures each machine
once** (`gpu-offload calibrate`), decides per call from the measurements, keeps correcting itself
while programs run, and exposes one setting: a level from 1 to 10 for how eager it should be.

## Key results

All from one machine: Ryzen 9 9950X3D (16 cores), RTX 5080 16 GB, Ubuntu 24.04 in WSL2, CUDA 13.3,
OpenBLAS with 16 threads. Details, raw numbers and caveats are in
[docs/experiments.md](docs/experiments.md).

| Result | Value |
| --- | --- |
| Whole-program speedup, NumPy linear algebra at n = 6000 (median of 3 runs) | **1.53x** at level 5, **1.69x** at level 9 |
| The same with float32 input | 1.60x and 1.66x |
| One `numpy.linalg.eigh` call at n = 6000 | 2.27 s on the GPU; the CPU was predicted to need 11.1 s |
| Accuracy of the cost model on that call | predicted 2.23 s, measured 2.27 s |
| Workloads without large dense math (pandas, NumPy FFT, k-means, small neural nets) | no calls offloaded, run time unchanged within noise |
| Correctness | 168 result checks pass on the GPU in four precisions; 128 GPU results compared with CPU results, 0 mismatches |
| Overhead on a call that stays on the CPU | about 15 ns |
| Share of run time that is interceptable at all | 0% to 60% depending on the program |

## When to use it, and when not

| | Situation |
| --- | --- |
| **Excellent** | Large symmetric eigenproblems and singular value problems (n from about 1000), even in float64 |
| **Excellent** | Single-precision and complex-single dense algebra (products, solves) from n of about 2000 |
| **Excellent** | Programs you cannot or do not want to modify: a script, a vendor binary, legacy Fortran |
| **Good** | Finding out whether a GPU could help a program at all: `gpu-offload measure` works without a GPU |
| **No gain** | Double-precision matrix products, triangular solves and Cholesky on a consumer GPU with a strong CPU |
| **No gain** | Small problems (n below roughly 1000), and programs made of many tiny calls (k-means, small neural-net layers) |
| **No gain** | pandas pipelines, NumPy FFT, text or I/O-bound work: they do not call BLAS / LAPACK |
| **Do not use** | When you need results bit-identical to a CPU run |
| **Do not use** | Production services, safety-critical or regulated computation: this is a v0.1 prototype |
| **Do not use** | When you can port the program to CuPy, PyTorch or RAPIDS: keeping data on the GPU is faster |
| **Do not use** | Security-sensitive or set-uid processes: it works by injecting a library with `LD_PRELOAD` |

In the "no gain" cases the tool does nothing rather than harm: such calls stay on the CPU.
The full guide, with the evidence for each row, is [docs/when-to-use.md](docs/when-to-use.md).

## Quick start

**Requirements:** Linux x86-64 (WSL2 works), an NVIDIA GPU with its driver, the CUDA 13 runtime
libraries (cuBLAS, cuSOLVER), Python 3 with NumPy, and a system BLAS / LAPACK such as OpenBLAS.

Install the release package:

```bash
# from https://github.com/jaifar530/cpu-to-gpu-offloading/releases
sudo apt install ./gpu-offload_0.1.1_amd64.deb
```

or build from source (Ubuntu 24.04):

```bash
sudo apt install build-essential cmake ninja-build python3-numpy libopenblas-dev liblapacke-dev libfftw3-dev
# plus NVIDIA's CUDA toolkit 13.x (cuda-toolkit-13-3 from NVIDIA's apt repository)
git clone https://github.com/jaifar530/cpu-to-gpu-offloading.git && cd cpu-to-gpu-offloading
./build.sh test          # builds into ~/build/gpu-offload and runs the test suite
./build.sh package       # builds the .deb
```

Then:

```bash
gpu-offload calibrate                      # learn this machine (once)
gpu-offload status                         # see what it learned
gpu-offload measure python3 my_script.py   # how much of my program is offloadable?
gpu-offload run python3 my_script.py       # run with automatic offloading
gpu-offload run --verify python3 my_script.py   # compare GPU and CPU results on my data
gpu-offload level 7                        # be more eager (1 = cautious ... 10 = eager)
```

Without the CUDA toolkit the build still produces the measurement part. Full instructions,
settings and troubleshooting: [docs/usage.md](docs/usage.md).

## How it works

```
 your program  (NumPy, R, Octave, C, Fortran ... unmodified)
       |  calls dgemm_, dsyevd_, cblas_sgemm, ...
       v
 libgpuoffload.so          loaded with LD_PRELOAD, sits in front of BLAS / LAPACK
       |
       |  1. predict:  T_cpu = work / cpu_speed(operation, precision, size)
       |               T_gpu = copy in + work / gpu_speed(...) + copy out
       |  2. decide:   GPU if T_cpu >= required_speedup(level) x T_gpu,
       |               GPU memory is free, and no other program is using the GPU
       v
   CPU library  <-- otherwise, and on any GPU error -->  libgpuoffload-cuda.so (cuBLAS, cuSOLVER)
```

1. **Calibrate (the "learn this machine" step).** `gpu-offload calibrate` runs every supported
   operation on the CPU and on the GPU, in four precisions, over a range of sizes, and writes a
   machine profile: speed curves for both devices, the cost of copies, and the GPU start-up time.
   It waits for a quiet machine before each measurement and repeats measurements that were
   disturbed.
2. **Decide per call.** The preload library looks each call up in the profile. The comparison
   includes the copies, so small problems stay on the CPU.
3. **Keep learning while running.** Every call is timed. If the CPU or the GPU turns out slower
   than the profile said (a different BLAS, a busy GPU), later decisions shift. The CPU is
   re-measured on purpose at least every 16 offloadable calls of an operation.
4. **Never pay for nothing.** The CUDA libraries are loaded only once the work seen so far would
   have repaid the GPU start-up cost. A program that does little heavy math never touches the GPU.
5. **Fall back.** Unsupported arguments, GPU out of memory, or any GPU error: the CPU runs the
   call. Results are written to the program's memory only after the GPU work has succeeded.

### The level (1 to 10)

`gpu-offload level 7` saves the level; `gpu-offload run -l 7 ...` sets it for one run.
The level never relaxes correctness; it only changes how much predicted gain is required and how
considerate the tool is towards other users of the GPU.

| Level | Offload when predicted speedup is at least | Back off when other programs load the GPU above | GPU memory always left free | Start the GPU once missed savings reach |
| --- | --- | --- | --- | --- |
| 1 | 3.0x | 30% | 50% | 3.0x its start-up cost |
| 3 | 2.0x | 44% | 40% | 2.5x |
| 5 (default) | 1.3x | 59% | 30% | 2.0x |
| 7 | 1.1x | 73% | 20% | 1.5x |
| 10 | 0.9x (accepts a small loss to free the CPU) | 95% | 5% | 0.75x |

The design in detail: [docs/how-it-works.md](docs/how-it-works.md).

## What is offloaded

| Family | Operations | Precisions |
| --- | --- | --- |
| BLAS level 3 (cuBLAS) | `gemm`, `syrk`, `herk`, `trsm`; Fortran and CBLAS interfaces, row- and column-major | float32, float64, complex64, complex128 |
| LAPACK (cuSOLVER) | `getrf`, `gesv`, `potrf`, `posv`, `syev` / `syevd` / `heev` / `heevd`, `gesdd` / `gesvd` (tall or square matrices) | same |

Many more calls are *measured* but not offloaded: BLAS levels 1 and 2, `symm`, `trmm`, `syr2k`,
`getrs`, `potrs`, `getri`, `geqrf`, `gelsd`, `geev`, `syevr`, and FFTW transforms. Both the
standard symbol names and those of the OpenBLAS bundled in PyPI's NumPy and SciPy wheels
(`scipy_dgemm_`, `scipy_dgemm_64_`) are covered: 468 wrappers, generated from one table.

## Commands

| Command | What it does |
| --- | --- |
| `gpu-offload calibrate [--quick\|--full]` | Learn this machine. Default sizes up to 4096; `--full` up to 8192 (31 to 34 minutes on the development machine); `--quick` up to 2048. |
| `gpu-offload run [-l LEVEL] CMD...` | Run a program with automatic offloading; prints a summary at exit (`-q` to silence). |
| `gpu-offload run --verify CMD...` | Run every GPU-capable call on both devices, compare the results, keep the CPU result. |
| `gpu-offload run --trace DIR CMD...` | Also record every call; read it with `gpu-offload report DIR`. |
| `gpu-offload level [N]` | Show or set the level. |
| `gpu-offload status` | Show installation, GPU, level and the calibration table. |
| `gpu-offload measure CMD...` | Only record what math a program does (no offloading) and report it. |
| `gpu-offload estimate DIR...` | From a `measure` trace, predict the saving at each level. |
| `gpu-offload env` | Print the shell setting that offloads everything started from that shell. |

## Experiments and benchmarks

Everything measured while building the tool is written up in
**[docs/experiments.md](docs/experiments.md)**: eight experiments, the full 46-row calibration
table, raw repetition timings, and the results that went wrong and why. A summary:

| # | Experiment | Main result |
| --- | --- | --- |
| 1 | CPU vs GPU on a matrix product | float32: GPU 4.0x to 5.9x faster; float64: GPU equal or 1.4x slower. Copies dominate the GPU's time in float32. |
| 2 | Effect of the BLAS thread count | OpenBLAS's default (32 threads) was 15 to 25 times slower than 16 threads on NumPy workloads. |
| 3 | How much of real programs is offloadable | 60% for NumPy linear algebra, 43 to 53% for neural-net and scikit-learn scripts, 34% for R, 16% for Octave, 0% for pandas and NumPy FFT. |
| 4 | Overhead of the library | 13 ns plain, 28 to 30 ns with offloading enabled, 131 to 213 ns when tracing, per tiny call. |
| 5 | Calibration | Three full runs (31 to 34 minutes each); the CPU library ran up to 300x slower in bursts caused by other virtual machines, which led to the quiet-machine check. |
| 6 | Correctness | 168 checks pass on CPU and GPU; verify mode: 128 comparisons, 0 mismatches, largest relative difference 1.23e-06. |
| 7 | End-to-end speedup | 1.53x to 1.69x on NumPy linear algebra at n = 6000 (median of 3); unchanged elsewhere. |
| 8 | Packaging | The `.deb` installs and runs end to end on Ubuntu 24.04. |

Whole-program wall time, plain versus `gpu-offload run`, median of 3 interleaved runs, outputs
compared and identical in every case:

| Workload | Plain | Level 5 | Level 9 | What happened |
| --- | --- | --- | --- | --- |
| NumPy linear algebra, n = 6000, float64 | 20.8 s | 13.5 s (1.53x) | 12.3 s (1.69x) | the eigendecomposition moved to the GPU |
| NumPy linear algebra, n = 6000, float32 input | 19.1 s | 11.9 s (1.60x) | 11.5 s (1.66x) | same (NumPy's solvers compute in float64 internally) |
| NumPy linear algebra, n = 3000, float64 | 4.2 s | 3.8 s (1.09x) | 3.9 s (1.07x) | one call offloaded |
| NumPy linear algebra, n = 3000, float32 input | 3.9 s | 3.9 s (1.01x) | 3.9 s (1.01x) | one call offloaded |
| NumPy neural-net training, float32 | 5.7 s | 5.5 s | 4.6 s | products too small to offload at level 5 |
| scikit-learn pipeline | 10.8 s | 8.2 s | 9.1 s | nothing offloaded (467,664 tiny products) |
| NumPy FFT signal processing | 0.7 s | 0.7 s | 0.9 s | nothing to offload (NumPy's FFT is built in) |

Read these with care:

- **The machine was shared** with other virtual machines, so timings vary by 20 to 30% between
  runs. The scikit-learn row shows that noise: nothing was offloaded there, yet the times differ.
- **This benchmark is incomplete.** It stopped after seven of thirteen workloads when the host ran
  low on memory. The other six (pandas, Octave, R, and three runs with PyPI NumPy) have single-run
  results only; they are in [docs/experiments.md](docs/experiments.md).
- **Speedups are bounded** by how much of a program is heavy math. `gpu-offload measure CMD` tells
  you that share for your own program before you expect anything.
- **Not measured at all:** other machines, other GPUs, native Linux, MKL, energy use. Reports from
  other machines are very welcome.

## Correctness

- The test suite calls every offloaded operation in all four precisions and checks each result
  against references computed in plain loops (products, residuals, reconstructions): 168 checks,
  run once on the CPU libraries and once forced onto the GPU.
- In verify mode every GPU result is compared with the CPU result of the same call. Across the
  test suite: 128 calls compared, 0 mismatches, largest relative difference 1.2e-06 (float32).
- GPU results are not bit-identical to CPU results: floating-point operations run in a different
  order. Eigenvectors and singular vectors may differ in sign, which is equally valid.
- Not yet exercised by any test: another program loading the GPU, GPU memory running out,
  several threads offloading at once, and `fork()` after the GPU has started.

## Limitations

- **Every offloaded call copies its data to the GPU and back.** Data does not stay on the GPU
  between calls. The measurement tool reports how much copying a residency scheme could avoid
  (27% to 73% in the test workloads); building one safely is the main open problem.
- **Opt-in per command or per shell**, not a system-wide service. There is no background daemon.
- Linux only; built against CUDA 13; one GPU. No AMD or Intel GPU backend.
- A process created with `fork()` after the GPU was started keeps running on the CPU.
- Calibration assumes the CPU library and thread count your programs will use. With OpenBLAS,
  one thread per *physical* core can be much faster than the default; `calibrate` checks and says
  so.
- Only math that crosses a shared-library boundary can be seen. Code compiled into a program
  (NumPy's FFT, statically linked BLAS) is invisible.
- Programs dominated by many tiny calls (scikit-learn's k-means, small neural-net layers) gain
  nothing.
- What the tool learns while a program runs is not saved between runs.

## Comparison with other tools

| Approach | Change needed in your program | Covers | Who decides CPU or GPU |
| --- | --- | --- | --- |
| **gpu-offload** | None | BLAS level 3 and LAPACK calls of any dynamically linked program | A cost model calibrated per machine, corrected at run time |
| NVBLAS | None | Level-3 BLAS | The library and its configuration |
| SCILIB-Accel | None | Level-3 BLAS | A matrix-size threshold |
| CuPy | Replace `numpy` with `cupy` | NumPy-style arrays | You, in the code |
| cudf.pandas / cuml.accel | One flag or import | pandas / scikit-learn | The library (GPU where supported) |
| CUDA, SYCL, OpenACC | Rewrite or annotate, recompile | Anything you write | You |

Tools that keep data on the GPU (CuPy, PyTorch, RAPIDS) are faster than gpu-offload when you can
use them. gpu-offload is for programs you do not want to change. More in
[docs/comparison.md](docs/comparison.md).

## FAQ

**Can I run CPU code on a GPU without changing the code?**
Partly. No tool can move an arbitrary CPU program onto a GPU. The calls a program makes to
standard math libraries can be intercepted and run on the GPU transparently; gpu-offload does that
for the large ones and leaves everything else on the CPU.

**Does it work with NumPy, SciPy, R and Octave?**
Yes. NumPy and SciPy from the distribution and from PyPI, R and GNU Octave were tested unmodified.

**How much faster will my program be?**
It depends on its share of large dense linear algebra. Run `gpu-offload measure your-command`
first. Measured here: 1.5x to 1.7x on a linear-algebra-heavy script, nothing on pandas or FFT.

**Are the results the same?**
To rounding error, not bit for bit. `gpu-offload run --verify` compares both on your data.

**Does it work on Windows?**
Through WSL2, yes; that is how it was developed. There is no native Windows or macOS version.

More questions and answers: [docs/faq.md](docs/faq.md).

## Contributing

The project is open to contributions of every size, and you do not need a GPU for many of them.

- **No coding:** run `gpu-offload calibrate` and share `gpu-offload status` from your machine
  through a "Machine report" issue, or run the benchmark on a quiet machine.
- **Code:** add an operation (QR, `syevr`), an FFT path through cuFFT, another GPU backend, or
  work on keeping data on the GPU between calls.
- **Docs and bug reports** are just as valuable.

Start with [CONTRIBUTING.md](CONTRIBUTING.md), the
[good first issues](https://github.com/jaifar530/cpu-to-gpu-offloading/labels/good%20first%20issue),
and the [Code of Conduct](CODE_OF_CONDUCT.md). Security problems: see [SECURITY.md](SECURITY.md).

## Roadmap

1. Data residency: keep unchanged operands on the GPU between calls.
2. More operations: QR (`geqrf` / `orgqr`), `syevr`, least squares, FFT through cuFFT.
3. A system-wide policy service so several programs share one GPU fairly.
4. Per-application learned profiles that persist between runs.
5. Other GPU vendors through a portable backend; a Windows port.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/core.c` | Finding the real library functions, tracing, start-up and exit. |
| `src/policy.c` | The profile, the cost model, the level, the decision, verify mode. |
| `src/cuda_backend.c` | cuBLAS / cuSOLVER implementations of the offloaded calls. |
| `src/fftw.c`, `src/fftw.inc` | FFTW measurement wrappers. |
| `src/calibrate.c` | The calibration workload. |
| `tools/gen_wrappers.py` | Generates the BLAS / LAPACK wrappers from one table of operations. |
| `tools/gpu-offload`, `tools/ofl_*.py` | The command-line tool, profile fitting, reports. |
| `tests/` | Test programs and trace checks (`./build.sh test`). |
| `workloads/` | Example programs and `bench.py`, which produced the results above. |
| `docs/` | The documentation website. |
| `experiments/raw/` | Raw outputs of the experiments: the machine profile, per-call reports, timing tables. |

## Citing

If you use gpu-offload in your work, please cite it using the metadata in
[CITATION.cff](CITATION.cff) (GitHub's "Cite this repository" button).

## Licence

Copyright 2026 Jaifar Alshizawi. gpu-offload is dual-licensed:

- **Open source:** the [GNU Affero General Public License v3](LICENSE). You may use, modify and
  share it free of charge under the AGPL's conditions.
- **Commercial:** if those conditions do not work for you (for example you want to ship it inside
  a closed-source product, offer a modified version as a service without publishing your changes,
  or your organisation does not permit AGPL software), a commercial licence is available.

Details and how to enquire: [LICENSING.md](LICENSING.md). Contributions are accepted under the
[Contributor Licence Agreement](CLA.md).

---

Keywords: transparent GPU offloading, automatic CPU to GPU offload, run CPU code on GPU without
code changes, drop-in GPU acceleration, LD_PRELOAD BLAS LAPACK interception, cuBLAS, cuSOLVER,
NVIDIA CUDA, NumPy GPU acceleration without code changes, R and Octave GPU acceleration,
heterogeneous computing, per-machine calibration, auto-tuning, cost model, NVBLAS alternative.
