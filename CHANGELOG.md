# Changelog

All notable changes to gpu-offload are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.1] - 2026-10-05

First public release: a working prototype, tested on one machine.

### Licence
- Dual-licensed under the GNU AGPL v3 or a commercial licence (see `LICENSING.md`). Contributions
  require agreement to the Contributor Licence Agreement (`CLA.md`). A first snapshot, 0.1.0, was
  briefly published under the Apache License 2.0.

### Added
- `libgpuoffload.so`, an `LD_PRELOAD` library with 468 generated wrappers for BLAS levels 1 to 3
  and LAPACK in four precisions, the Fortran and CBLAS interfaces, and the symbol names of the
  OpenBLAS bundled in PyPI's NumPy and SciPy wheels; FFTW wrappers for measurement.
- GPU offloading through cuBLAS and cuSOLVER for `gemm`, `syrk`, `herk`, `trsm`, `getrf`, `gesv`,
  `potrf`, `posv`, `syev` / `syevd` / `heev` / `heevd`, `gesdd` / `gesvd`, with CPU fallback.
- `gpu-offload calibrate`: per-machine profile of CPU and GPU speed per operation, precision and
  size, copy costs and GPU start-up cost, with a quiet-machine check.
- A per-call cost-model decision with an aggressiveness level from 1 to 10, deferred GPU start-up,
  GPU-memory and GPU-busy checks, run-time correction and periodic CPU probes.
- `gpu-offload run`, `measure`, `report`, `estimate`, `level`, `status`, `env`; verify mode.
- Test suite (168 result checks per run), example workloads, a benchmark driver, `.deb` packaging.
- Documentation site in `docs/`: overview, usage, design, when to use it, the full record of
  experiments and benchmarks, comparison with other tools, FAQ.
- `CONTRIBUTING.md`, code of conduct, security policy, issue and pull-request templates, citation
  file; continuous integration that builds without a GPU and runs the CPU parts of the test suite.
