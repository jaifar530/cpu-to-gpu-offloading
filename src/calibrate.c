// SPDX-License-Identifier: AGPL-3.0-only
// Calibration workload: calls every GPU-capable operation over a range of sizes and precisions.
//
// It is an ordinary BLAS / LAPACK client. `gpu-offload calibrate` runs it twice under the preload
// library, once forced to the CPU and once forced to the GPU, and fits the machine profile from
// the two traces. Symbols are looked up by name so one code path serves all four precisions.
//
// usage: gpu-offload-calibrate [--max-n N] [--min-n N] [--precs sdcz] [--ops gemm,...] [--point-seconds S]
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define COL_MAJOR 102  // LAPACK_COL_MAJOR
#define NRHS 4

static char g_prec;
static size_t g_es, g_rs;  // bytes per element, bytes per real number
static int g_complex;
static double g_point_seconds = 20;

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *lookup(const char *fmt, const char *base) {
    char name[64];
    snprintf(name, sizeof name, fmt, g_prec, base);
    void *p = dlsym(RTLD_DEFAULT, name);
    if (!p) {
        fprintf(stderr, "calibrate: %s not found\n", name);
        exit(1);
    }
    return p;
}
#define BLAS(base) lookup("%c%s_", base)
#define LAPACKE(base) lookup("LAPACKE_%c%s", base)

// ---- waiting for a quiet machine ----
//
// A multithreaded CPU math library can run hundreds of times slower for seconds at a stretch when
// other programs (or other virtual machines) take its cores. A calibration point measured during
// such an episode would make the CPU look far slower than it is. A small fixed CPU job, the
// canary, is timed before and after every point: a point starts only once the canary runs at its
// normal speed, and is measured again if the canary has slowed down by the time it ends.

#define CANARY_N 384
static double g_canary_best = 1e30;
static double *g_canary[3];

static double canary(void) {
    typedef void (*symm_fn)(const char *, const char *, const int *, const int *, const double *, const double *,
                            const int *, const double *, const int *, const double *, double *, const int *, size_t,
                            size_t);
    static symm_fn dsymm;
    int n = CANARY_N;
    double one = 1, zero = 0, best = 1e30;
    if (!dsymm) {
        dsymm = (symm_fn)dlsym(RTLD_DEFAULT, "dsymm_");  // BLAS level 3, threaded, and never offloaded
        for (int i = 0; i < 3; i++) g_canary[i] = calloc((size_t)n * n, sizeof(double));
        for (int i = 0; i < n * n; i++) g_canary[0][i] = g_canary[1][i] = 1.0 / (i + 1);
    }
    for (int rep = 0; rep < 3; rep++) {
        double t0 = now();
        dsymm("L", "U", &n, &n, &one, g_canary[0], &n, g_canary[1], &n, &zero, g_canary[2], &n, 1, 1);
        double dt = now() - t0;
        if (dt < best) best = dt;
    }
    if (best < g_canary_best) g_canary_best = best;
    return best;
}

static int quiet(void) {
    return canary() <= 2.5 * g_canary_best;
}

static void wait_for_quiet(void) {
    for (int tries = 0; tries < 150; tries++) {  // up to about 30 s
        if (quiet()) return;
        usleep(200000);
    }
    fprintf(stderr, "calibrate: the machine stays busy; measuring anyway\n");
}

// ---- test data ----

static uint64_t g_seed = 0x2545F4914F6CDD1Dull;
static double rnd(void) {  // uniform in [-0.5, 0.5)
    g_seed ^= g_seed << 13;
    g_seed ^= g_seed >> 7;
    g_seed ^= g_seed << 17;
    return (double)(g_seed >> 11) / 9007199254740992.0 - 0.5;
}

static void set(void *m, size_t idx, double re, double im) {
    if (g_rs == 8) {
        ((double *)m)[idx * (g_es / 8)] = re;
        if (g_complex) ((double *)m)[idx * 2 + 1] = im;
    } else {
        ((float *)m)[idx * (g_es / 4)] = (float)re;
        if (g_complex) ((float *)m)[idx * 2 + 1] = (float)im;
    }
}

static void *matrix(int64_t rows, int64_t cols) {
    void *m = malloc((size_t)(rows * cols) * g_es);
    if (!m) {
        fprintf(stderr, "calibrate: out of memory\n");
        exit(1);
    }
    for (int64_t i = 0; i < rows * cols; i++) set(m, (size_t)i, rnd(), rnd());
    return m;
}

// Well-conditioned n x n matrix; Hermitian positive definite if `hermitian`.
static void *conditioned(int64_t n, int hermitian) {
    void *m = matrix(n, n);
    for (int64_t j = 0; j < n; j++) {
        if (hermitian)
            for (int64_t i = 0; i < j; i++) {
                double re = rnd(), im = rnd();
                set(m, (size_t)(i + j * n), re, im);
                set(m, (size_t)(j + i * n), re, -im);
            }
        set(m, (size_t)(j + j * n), (double)n, 0);
    }
    return m;
}

// ---- operations ----

enum { GEMM, SYRK, HERK, TRSM, GETRF, GESV, POTRF, POSV, EIG_VALUES, EIG_VECTORS, SVD_VALUES, SVD_VECTORS, NOPS };
static const char *const op_label[NOPS] = {"gemm",  "syrk", "herk",        "trsm",         "getrf",       "gesv",
                                           "potrf", "posv", "eigenvalues", "eigenvectors", "svd values", "svd vectors"};

typedef struct {
    int64_t n;
    void *a, *b, *a0, *b0, *c, *w, *u, *vt;
    int *ipiv;
} work;

static void prepare(int op, work *k, int64_t n) {
    memset(k, 0, sizeof *k);
    k->n = n;
    size_t nn = (size_t)(n * n) * g_es;
    switch (op) {
    case GEMM:
        k->a = matrix(n, n), k->b = matrix(n, n), k->c = malloc(nn);
        break;
    case SYRK:
    case HERK:
        k->a = matrix(n, n), k->c = malloc(nn);
        memset(k->c, 0, nn);
        break;
    case TRSM:
        k->a = conditioned(n, 0), k->b0 = matrix(n, n), k->b = malloc(nn);
        break;
    case GETRF:
        k->a0 = conditioned(n, 0), k->a = malloc(nn), k->ipiv = malloc(sizeof(int) * (size_t)n);
        break;
    case GESV:
        k->a0 = conditioned(n, 0), k->a = malloc(nn), k->ipiv = malloc(sizeof(int) * (size_t)n);
        k->b0 = matrix(n, NRHS), k->b = malloc((size_t)(n * NRHS) * g_es);
        break;
    case POTRF:
        k->a0 = conditioned(n, 1), k->a = malloc(nn);
        break;
    case POSV:
        k->a0 = conditioned(n, 1), k->a = malloc(nn);
        k->b0 = matrix(n, NRHS), k->b = malloc((size_t)(n * NRHS) * g_es);
        break;
    case EIG_VALUES:
    case EIG_VECTORS:
        k->a0 = conditioned(n, 1), k->a = malloc(nn), k->w = malloc((size_t)n * g_rs);
        break;
    case SVD_VALUES:
    case SVD_VECTORS:
        k->a0 = matrix(n, n), k->a = malloc(nn), k->w = malloc((size_t)n * g_rs);
        if (op == SVD_VECTORS) k->u = malloc(nn), k->vt = malloc(nn);
        break;
    }
}

static void release(work *k) {
    free(k->a), free(k->b), free(k->a0), free(k->b0), free(k->c), free(k->w), free(k->u), free(k->vt), free(k->ipiv);
}

// Runs the operation once and returns the seconds it took (0 if it reported a failure).
static double run(int op, work *k) {
    int n = (int)k->n, nrhs = NRHS, info = 0;
    size_t nn = (size_t)k->n * (size_t)k->n * g_es;
    float f_one[2] = {1, 0}, f_zero[2] = {0, 0};
    double d_one[2] = {1, 0}, d_zero[2] = {0, 0};
    const void *one = g_rs == 8 ? (void *)d_one : (void *)f_one, *zero = g_rs == 8 ? (void *)d_zero : (void *)f_zero;

    if (k->a0) memcpy(k->a, k->a0, nn);  // in-place operations get fresh input every time
    if (k->b0) memcpy(k->b, k->b0, (op == TRSM ? nn : (size_t)(k->n * NRHS) * g_es));

    double t0 = now();
    switch (op) {
    case GEMM:
        ((void (*)(const char *, const char *, const int *, const int *, const int *, const void *, const void *,
                   const int *, const void *, const int *, const void *, void *, const int *, size_t, size_t))BLAS(
            "gemm"))("N", "N", &n, &n, &n, one, k->a, &n, k->b, &n, zero, k->c, &n, 1, 1);
        break;
    case SYRK:
    case HERK:
        ((void (*)(const char *, const char *, const int *, const int *, const void *, const void *, const int *,
                   const void *, void *, const int *, size_t, size_t))BLAS(op == HERK ? "herk" : "syrk"))(
            "U", "N", &n, &n, one, k->a, &n, zero, k->c, &n, 1, 1);
        break;
    case TRSM:
        ((void (*)(const char *, const char *, const char *, const char *, const int *, const int *, const void *,
                   const void *, const int *, void *, const int *, size_t, size_t, size_t, size_t))BLAS("trsm"))(
            "L", "U", "N", "N", &n, &n, one, k->a, &n, k->b, &n, 1, 1, 1, 1);
        break;
    case GETRF:
        info = ((int (*)(int, int, int, void *, int, int *))LAPACKE("getrf"))(COL_MAJOR, n, n, k->a, n, k->ipiv);
        break;
    case GESV:
        info = ((int (*)(int, int, int, void *, int, int *, void *, int))LAPACKE("gesv"))(COL_MAJOR, n, nrhs, k->a, n,
                                                                                          k->ipiv, k->b, n);
        break;
    case POTRF:
        info = ((int (*)(int, char, int, void *, int))LAPACKE("potrf"))(COL_MAJOR, 'U', n, k->a, n);
        break;
    case POSV:
        info = ((int (*)(int, char, int, int, void *, int, void *, int))LAPACKE("posv"))(COL_MAJOR, 'U', n, nrhs, k->a,
                                                                                         n, k->b, n);
        break;
    case EIG_VALUES:
    case EIG_VECTORS:
        info = ((int (*)(int, char, char, int, void *, int, void *))LAPACKE(g_complex ? "heevd" : "syevd"))(
            COL_MAJOR, op == EIG_VECTORS ? 'V' : 'N', 'U', n, k->a, n, k->w);
        break;
    case SVD_VALUES:
    case SVD_VECTORS:
        info = ((int (*)(int, char, int, int, void *, int, void *, void *, int, void *, int))LAPACKE("gesdd"))(
            COL_MAJOR, op == SVD_VECTORS ? 'S' : 'N', n, n, k->a, n, k->w, k->u, n, k->vt, n);
        break;
    }
    double dt = now() - t0;
    if (info != 0) {
        fprintf(stderr, "calibrate: %c %s n=%d reported info=%d\n", g_prec, op_label[op], n, info);
        return 0;
    }
    return dt;
}

int main(int argc, char **argv) {
    int64_t max_n = 4096, min_n = 128;
    const char *precs = "sdcz", *ops = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--max-n") && i + 1 < argc) max_n = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--min-n") && i + 1 < argc) min_n = atoll(argv[++i]);
        else if (!strcmp(argv[i], "--precs") && i + 1 < argc) precs = argv[++i];
        else if (!strcmp(argv[i], "--ops") && i + 1 < argc) ops = argv[++i];
        else if (!strcmp(argv[i], "--point-seconds") && i + 1 < argc) g_point_seconds = atof(argv[++i]);
        else {
            fprintf(stderr, "usage: %s [--max-n N] [--min-n N] [--precs sdcz] [--ops gemm,...] [--point-seconds S]\n",
                    argv[0]);
            return 2;
        }
    }

    for (const char *p = precs; *p; p++) {
        g_prec = *p;
        g_complex = g_prec == 'c' || g_prec == 'z';
        g_rs = (g_prec == 's' || g_prec == 'c') ? 4 : 8;
        g_es = g_rs * (g_complex ? 2 : 1);
        for (int op = 0; op < NOPS; op++) {
            if (op == HERK && !g_complex) continue;
            if (ops && !strstr(ops, op_label[op])) continue;
            for (int64_t n = min_n; n <= max_n; n *= 2) {
                work k;
                prepare(op, &k, n);
                double last = 0;
                for (int attempt = 0; attempt < 3; attempt++) {
                    wait_for_quiet();
                    // At least 3 runs, more for fast points, until about 0.3 s has been measured.
                    double total = 0;
                    int reps = 0;
                    while (reps < 3 || (total < 0.3 && reps < 12)) {
                        last = run(op, &k);
                        total += last;
                        reps++;
                        if (last == 0 || last > g_point_seconds) break;
                    }
                    if (last == 0 || quiet()) break;  // otherwise the machine got busy meanwhile: once more
                }
                release(&k);
                fprintf(stderr, "  %c %-12s n=%-5lld %9.4f s\n", g_prec, op_label[op], (long long)n, last);
                if (last == 0 || last > g_point_seconds) break;  // larger sizes would take too long
            }
        }
    }
    return 0;
}
