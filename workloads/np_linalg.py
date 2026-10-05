"""Scientific-computing style NumPy script: subspace iteration, direct solves, eigenvalues and an
SVD, with ordinary array code around them.

usage: np_linalg.py [N] [float64|float32]      (default: 3000 float64)
"""
import sys

import numpy as np

n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
dtype = np.dtype(sys.argv[2] if len(sys.argv) > 2 else "float64")
rng = np.random.default_rng(0)


def randn(*shape):
    return rng.standard_normal(shape, dtype=dtype)


# Build a symmetric positive definite operator.
g = randn(n, n)
a = g @ g.T / dtype.type(n) + np.eye(n, dtype=dtype)

# Subspace iteration: the same operator applied again and again.
x = randn(n, 64)
for _ in range(60):
    x = a @ x
    x, _ = np.linalg.qr(x)
ritz = np.sort(np.diag(x.T @ a @ x))[::-1]

# Direct solves with fresh right-hand sides.
for _ in range(6):
    b = randn(n, 8)
    sol = np.linalg.solve(a, b)
    resid = np.abs(a @ sol - b).max()

eigs, vecs = np.linalg.eigh(a)
u, s, vt = np.linalg.svd(randn(2 * n // 3, n // 3), full_matrices=False)

# Plain array work that no math library call covers.
field = randn(4000, 4000)
for _ in range(5):
    field = np.tanh(field) * dtype.type(0.9) + np.roll(field, 1, axis=0) * dtype.type(0.1)
hist = np.histogram(field, bins=200)[0]

small = resid < (1e-8 if dtype == np.float64 else 1e-2)
print(f"largest Ritz value {ritz[0]:.3f}, largest eigenvalue {eigs[-1]:.3f}, residual small: {small}, "
      f"largest singular value {s[0]:.2f}, histogram peak {hist.max()}")
