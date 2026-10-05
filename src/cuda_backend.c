// SPDX-License-Identifier: AGPL-3.0-only
// CUDA backend: runs GPU-capable calls with cuBLAS and cuSOLVER.
//
// Each call copies its operands to the GPU, computes, and copies the results back (no data stays
// on the GPU between calls). Results are written to the caller's buffers only after the GPU work
// has succeeded, so any failure before that point lets the CPU run the call instead.
#define _GNU_SOURCE
#include "backend.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static cublasHandle_t g_blas;
static cusolverDnHandle_t g_solver;
static char g_name[256];

static const size_t ESZ[4] = {4, 8, 8, 16}, RSZ[4] = {4, 8, 4, 8};

static int pidx(char p) {
    return p == 's' ? 0 : p == 'd' ? 1 : p == 'c' ? 2 : 3;
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// ---------------------------------------------------------------------------------------------
// Device memory: a few grow-only scratch buffers, released when they add up to more than
// POOL_KEEP_BYTES so that an idle process does not sit on GPU memory other programs could use.
// ---------------------------------------------------------------------------------------------

#define NSLOT 8
#define POOL_KEEP_BYTES (1ull << 30)
static struct {
    void *p;
    size_t cap;
} g_pool[NSLOT];

static void pool_release(void) {
    for (int i = 0; i < NSLOT; i++) {
        if (g_pool[i].p) cudaFree(g_pool[i].p);
        g_pool[i].p = NULL;
        g_pool[i].cap = 0;
    }
}

static void pool_trim(void) {
    size_t total = 0;
    for (int i = 0; i < NSLOT; i++) total += g_pool[i].cap;
    if (total > POOL_KEEP_BYTES) pool_release();
}

static void *dev(int slot, size_t bytes) {
    if (bytes < 16) bytes = 16;
    if (g_pool[slot].cap >= bytes) return g_pool[slot].p;
    if (g_pool[slot].p) cudaFree(g_pool[slot].p);
    g_pool[slot].p = NULL;
    g_pool[slot].cap = 0;
    void *p = NULL;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        cudaGetLastError();  // clear the sticky error so later calls can still succeed
        return NULL;
    }
    g_pool[slot].p = p;
    g_pool[slot].cap = bytes;
    return p;
}

#define DEV(var, slot, bytes)                \
    void *var = dev(slot, (size_t)(bytes));  \
    if (!var) return OFL_WHY_VRAM
#define TRY(expr)                            \
    do {                                     \
        if (expr) return OFL_WHY_ERROR;      \
    } while (0)

// ---------------------------------------------------------------------------------------------
// Transfers
// ---------------------------------------------------------------------------------------------

// Host matrix (rows x cols, leading dimension ld) -> compact device matrix (leading dimension rows).
static int up(void *d, const void *h, int64_t rows, int64_t cols, int64_t ld, size_t es) {
    if (ld == rows || cols == 1) return cudaMemcpy(d, h, (size_t)(rows * cols) * es, cudaMemcpyHostToDevice) != cudaSuccess;
    return cublasSetMatrix((int)rows, (int)cols, (int)es, h, (int)ld, d, (int)rows) != CUBLAS_STATUS_SUCCESS;
}

// Once results start to land in the caller's buffers there is no way back to the CPU path.
static void down(void *h, const void *d, int64_t rows, int64_t cols, int64_t ld, size_t es) {
    int failed;
    if (!h) return;
    if (ld == rows || cols == 1) failed = cudaMemcpy(h, d, (size_t)(rows * cols) * es, cudaMemcpyDeviceToHost) != cudaSuccess;
    else failed = cublasGetMatrix((int)rows, (int)cols, (int)es, d, (int)rows, h, (int)ld) != CUBLAS_STATUS_SUCCESS;
    if (failed) {
        fprintf(stderr, "gpu-offload: fatal: copying a result back from the GPU failed\n");
        abort();
    }
}

static int fetch_info(const void *dinfo, int *info) {
    return cudaMemcpy(info, dinfo, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess;
}

static void store_int(void *p, int bytes, int64_t v) {
    if (!p) return;
    if (bytes == 8) *(int64_t *)p = v;
    else *(int32_t *)p = (int32_t)v;
}

// Pivots come back as 32-bit integers; the caller may use 64-bit ones.
static void down_pivots(void *host, const void *d, int64_t n, int int_bytes) {
    if (!host) return;
    if (int_bytes == 4) {
        down(host, d, n, 1, n, 4);
        return;
    }
    int32_t *tmp = malloc((size_t)n * 4);
    if (!tmp) abort();
    down(tmp, d, n, 1, n, 4);
    for (int64_t i = 0; i < n; i++) ((int64_t *)host)[i] = tmp[i];
    free(tmp);
}

// ---------------------------------------------------------------------------------------------
// Per-precision function tables. All arguments are pointers or integers, so one generic
// signature per routine covers the four precisions.
// ---------------------------------------------------------------------------------------------

typedef cublasStatus_t (*gemm_fn)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void *,
                                  const void *, int, const void *, int, const void *, void *, int);
typedef cublasStatus_t (*syrk_fn)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const void *,
                                  const void *, int, const void *, void *, int);
typedef cublasStatus_t (*trsm_fn)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                                  cublasDiagType_t, int, int, const void *, const void *, int, void *, int);
typedef cusolverStatus_t (*getrf_bs_fn)(cusolverDnHandle_t, int, int, void *, int, int *);
typedef cusolverStatus_t (*getrf_fn)(cusolverDnHandle_t, int, int, void *, int, void *, int *, int *);
typedef cusolverStatus_t (*getrs_fn)(cusolverDnHandle_t, cublasOperation_t, int, int, const void *, int, const int *,
                                     void *, int, int *);
typedef cusolverStatus_t (*potrf_bs_fn)(cusolverDnHandle_t, cublasFillMode_t, int, void *, int, int *);
typedef cusolverStatus_t (*potrf_fn)(cusolverDnHandle_t, cublasFillMode_t, int, void *, int, void *, int, int *);
typedef cusolverStatus_t (*potrs_fn)(cusolverDnHandle_t, cublasFillMode_t, int, int, const void *, int, void *, int,
                                     int *);
typedef cusolverStatus_t (*evd_bs_fn)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, const void *, int,
                                      const void *, int *);
typedef cusolverStatus_t (*evd_fn)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t, int, void *, int, void *,
                                   void *, int, int *);
typedef cusolverStatus_t (*svd_bs_fn)(cusolverDnHandle_t, int, int, int *);
typedef cusolverStatus_t (*svd_fn)(cusolverDnHandle_t, signed char, signed char, int, int, void *, int, void *, void *,
                                   int, void *, int, void *, int, void *, int *);

#define TABLE(type, name, s, d, c, z) static const type name[4] = {(type)s, (type)d, (type)c, (type)z}
TABLE(gemm_fn, GEMM, cublasSgemm, cublasDgemm, cublasCgemm, cublasZgemm);
TABLE(syrk_fn, SYRK, cublasSsyrk, cublasDsyrk, cublasCsyrk, cublasZsyrk);
TABLE(syrk_fn, HERK, cublasSsyrk, cublasDsyrk, cublasCherk, cublasZherk);
TABLE(trsm_fn, TRSM, cublasStrsm, cublasDtrsm, cublasCtrsm, cublasZtrsm);
TABLE(getrf_bs_fn, GETRF_BS, cusolverDnSgetrf_bufferSize, cusolverDnDgetrf_bufferSize, cusolverDnCgetrf_bufferSize,
      cusolverDnZgetrf_bufferSize);
TABLE(getrf_fn, GETRF, cusolverDnSgetrf, cusolverDnDgetrf, cusolverDnCgetrf, cusolverDnZgetrf);
TABLE(getrs_fn, GETRS, cusolverDnSgetrs, cusolverDnDgetrs, cusolverDnCgetrs, cusolverDnZgetrs);
TABLE(potrf_bs_fn, POTRF_BS, cusolverDnSpotrf_bufferSize, cusolverDnDpotrf_bufferSize, cusolverDnCpotrf_bufferSize,
      cusolverDnZpotrf_bufferSize);
TABLE(potrf_fn, POTRF, cusolverDnSpotrf, cusolverDnDpotrf, cusolverDnCpotrf, cusolverDnZpotrf);
TABLE(potrs_fn, POTRS, cusolverDnSpotrs, cusolverDnDpotrs, cusolverDnCpotrs, cusolverDnZpotrs);
TABLE(evd_bs_fn, EVD_BS, cusolverDnSsyevd_bufferSize, cusolverDnDsyevd_bufferSize, cusolverDnCheevd_bufferSize,
      cusolverDnZheevd_bufferSize);
TABLE(evd_fn, EVD, cusolverDnSsyevd, cusolverDnDsyevd, cusolverDnCheevd, cusolverDnZheevd);
TABLE(svd_bs_fn, SVD_BS, cusolverDnSgesvd_bufferSize, cusolverDnDgesvd_bufferSize, cusolverDnCgesvd_bufferSize,
      cusolverDnZgesvd_bufferSize);
TABLE(svd_fn, SVD, cusolverDnSgesvd, cusolverDnDgesvd, cusolverDnCgesvd, cusolverDnZgesvd);

static int valid(char c, const char *allowed) {
    return c && strchr(allowed, c) != NULL;
}
static cublasOperation_t op_of(char t) {
    return t == 'N' ? CUBLAS_OP_N : t == 'T' ? CUBLAS_OP_T : CUBLAS_OP_C;
}
static cublasFillMode_t fill_of(char u) {
    return u == 'U' ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
}
static int is_zero(const void *scalar, int p, int real_scalar) {
    size_t comps = real_scalar ? 1 : ESZ[p] / RSZ[p];
    for (size_t i = 0; i < comps; i++)
        if ((RSZ[p] == 8 ? ((const double *)scalar)[i] : ((const float *)scalar)[i]) != 0) return 0;
    return 1;
}

#define SWAP(type, x, y)  \
    do {                  \
        type tmp_ = x;    \
        x = y;            \
        y = tmp_;         \
    } while (0)

// ---------------------------------------------------------------------------------------------
// BLAS level 3
// ---------------------------------------------------------------------------------------------

static int do_gemm(const ofl_call *c, ofl_gpu_times *t) {
    int p = pidx(c->prec);
    size_t es = ESZ[p];
    const void *a = c->a, *b = c->b;
    int64_t lda = c->lda, ldb = c->ldb, m = c->m, n = c->n, k = c->k;
    char ta = c->trans_a, tb = c->trans_b;
    if (c->row_major) {  // C = op(A) op(B) stored by rows is C' = op(B') op(A') stored by columns
        SWAP(const void *, a, b);
        SWAP(int64_t, lda, ldb);
        SWAP(int64_t, m, n);
        SWAP(char, ta, tb);
    }
    if (k <= 0 || !valid(ta, "NTC") || !valid(tb, "NTC")) return OFL_WHY_ARGS;
    int64_t ra = ta == 'N' ? m : k, ca = ta == 'N' ? k : m;  // stored shape of A
    int64_t rb = tb == 'N' ? k : n, cb = tb == 'N' ? n : k;
    if (lda < ra || ldb < rb || c->ldc < m) return OFL_WHY_ARGS;
    int c_is_input = !is_zero(c->beta, p, 0);

    double t0 = now();
    DEV(da, 0, ra * ca * es);
    DEV(db, 1, rb * cb * es);
    DEV(dc, 2, m * n * es);
    TRY(up(da, a, ra, ca, lda, es));
    TRY(up(db, b, rb, cb, ldb, es));
    if (c_is_input) TRY(up(dc, c->c, m, n, c->ldc, es));
    double t1 = now();
    TRY(GEMM[p](g_blas, op_of(ta), op_of(tb), (int)m, (int)n, (int)k, c->alpha, da, (int)ra, db, (int)rb, c->beta, dc,
                (int)m));
    TRY(cudaDeviceSynchronize());
    double t2 = now();
    down(c->c_out, dc, m, n, c->ldc, es);
    t->in_s = t1 - t0, t->compute_s = t2 - t1, t->out_s = now() - t2;
    return 0;
}

static int do_syrk(const ofl_call *c, ofl_gpu_times *t, int herk) {
    int p = pidx(c->prec);
    size_t es = ESZ[p];
    int complex = es != RSZ[p];
    char uplo = c->uplo, trans = c->trans_a;
    if (!complex && trans == 'C') trans = 'T';
    if (!valid(uplo, "UL") || !valid(trans, herk && complex ? "NC" : "NT")) return OFL_WHY_ARGS;
    if (c->row_major) {
        uplo = uplo == 'U' ? 'L' : 'U';
        trans = trans == 'N' ? (herk && complex ? 'C' : 'T') : 'N';
    }
    int64_t n = c->n, k = c->k;
    int64_t ra = trans == 'N' ? n : k, ca = trans == 'N' ? k : n;
    if (k <= 0 || c->lda < ra || c->ldc < n) return OFL_WHY_ARGS;

    double t0 = now();
    DEV(da, 0, ra * ca * es);
    DEV(dc, 2, n * n * es);
    TRY(up(da, c->a, ra, ca, c->lda, es));
    TRY(up(dc, c->c, n, n, c->ldc, es));  // always: only one triangle is computed, the other must survive
    double t1 = now();
    TRY((herk ? HERK : SYRK)[p](g_blas, fill_of(uplo), op_of(trans), (int)n, (int)k, c->alpha, da, (int)ra, c->beta, dc,
                                (int)n));
    TRY(cudaDeviceSynchronize());
    double t2 = now();
    down(c->c_out, dc, n, n, c->ldc, es);
    t->in_s = t1 - t0, t->compute_s = t2 - t1, t->out_s = now() - t2;
    return 0;
}

static int do_trsm(const ofl_call *c, ofl_gpu_times *t) {
    int p = pidx(c->prec);
    size_t es = ESZ[p];
    char side = c->side, uplo = c->uplo;
    int64_t m = c->m, n = c->n;
    if (!valid(side, "LR") || !valid(uplo, "UL") || !valid(c->trans_a, "NTC") || !valid(c->diag, "NU")) return OFL_WHY_ARGS;
    if (c->row_major) {
        side = side == 'L' ? 'R' : 'L';
        uplo = uplo == 'U' ? 'L' : 'U';
        SWAP(int64_t, m, n);
    }
    int64_t ka = side == 'L' ? m : n;
    if (c->lda < ka || c->ldb < m) return OFL_WHY_ARGS;

    double t0 = now();
    DEV(da, 0, ka * ka * es);
    DEV(db, 1, m * n * es);
    TRY(up(da, c->a, ka, ka, c->lda, es));
    TRY(up(db, c->b, m, n, c->ldb, es));
    double t1 = now();
    TRY(TRSM[p](g_blas, side == 'L' ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT, fill_of(uplo), op_of(c->trans_a),
                c->diag == 'U' ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, (int)m, (int)n, c->alpha, da, (int)ka, db,
                (int)m));
    TRY(cudaDeviceSynchronize());
    double t2 = now();
    down(c->b_out, db, m, n, c->ldb, es);
    t->in_s = t1 - t0, t->compute_s = t2 - t1, t->out_s = now() - t2;
    return 0;
}

// ---------------------------------------------------------------------------------------------
// LAPACK
// ---------------------------------------------------------------------------------------------

// LU factorisation, optionally followed by a solve (gesv).
static int do_getrf(const ofl_call *c, ofl_gpu_times *t, int solve) {
    int p = pidx(c->prec), lwork = 0, info = 0;
    size_t es = ESZ[p];
    int64_t m = solve ? c->n : c->m, n = c->n, nrhs = c->k, mn = m < n ? m : n;
    if (c->lda < m || (solve && (nrhs <= 0 || c->ldb < n))) return OFL_WHY_ARGS;

    double t0 = now();
    DEV(da, 0, m * n * es);
    DEV(dpiv, 3, mn * sizeof(int));
    DEV(dinfo, 4, sizeof(int));
    TRY(up(da, c->a, m, n, c->lda, es));
    void *db = NULL;
    if (solve) {
        db = dev(1, (size_t)(n * nrhs) * es);
        if (!db) return OFL_WHY_VRAM;
        TRY(up(db, c->b, n, nrhs, c->ldb, es));
    }
    double t1 = now();
    TRY(GETRF_BS[p](g_solver, (int)m, (int)n, da, (int)m, &lwork));
    DEV(dwork, 2, (size_t)lwork * es);
    TRY(GETRF[p](g_solver, (int)m, (int)n, da, (int)m, dwork, dpiv, dinfo));
    TRY(fetch_info(dinfo, &info));
    if (info < 0) return OFL_WHY_ERROR;
    if (solve && info == 0) {
        int info2 = 0;
        TRY(GETRS[p](g_solver, CUBLAS_OP_N, (int)n, (int)nrhs, da, (int)n, dpiv, db, (int)n, dinfo));
        TRY(fetch_info(dinfo, &info2));
        if (info2) return OFL_WHY_ERROR;
    }
    TRY(cudaDeviceSynchronize());
    double t2 = now();
    down(c->a_out, da, m, n, c->lda, es);
    down_pivots(c->ipiv_out, dpiv, mn, c->int_bytes);
    if (solve && info == 0) down(c->b_out, db, n, nrhs, c->ldb, es);  // info > 0: singular, B is left alone
    store_int(c->info_out, c->int_bytes, info);
    t->in_s = t1 - t0, t->compute_s = t2 - t1, t->out_s = now() - t2;
    return 0;
}

// Cholesky factorisation, optionally followed by a solve (posv).
static int do_potrf(const ofl_call *c, ofl_gpu_times *t, int solve) {
    int p = pidx(c->prec), lwork = 0, info = 0;
    size_t es = ESZ[p];
    int64_t n = c->n, nrhs = c->k;
    if (!valid(c->uplo, "UL") || c->lda < n || (solve && (nrhs <= 0 || c->ldb < n))) return OFL_WHY_ARGS;

    double t0 = now();
    DEV(da, 0, n * n * es);
    DEV(dinfo, 4, sizeof(int));
    TRY(up(da, c->a, n, n, c->lda, es));
    void *db = NULL;
    if (solve) {
        db = dev(1, (size_t)(n * nrhs) * es);
        if (!db) return OFL_WHY_VRAM;
        TRY(up(db, c->b, n, nrhs, c->ldb, es));
    }
    double t1 = now();
    TRY(POTRF_BS[p](g_solver, fill_of(c->uplo), (int)n, da, (int)n, &lwork));
    DEV(dwork, 2, (size_t)lwork * es);
    TRY(POTRF[p](g_solver, fill_of(c->uplo), (int)n, da, (int)n, dwork, lwork, dinfo));
    TRY(fetch_info(dinfo, &info));
    if (info < 0) return OFL_WHY_ERROR;
    if (solve && info == 0) {
        int info2 = 0;
        TRY(POTRS[p](g_solver, fill_of(c->uplo), (int)n, (int)nrhs, da, (int)n, db, (int)n, dinfo));
        TRY(fetch_info(dinfo, &info2));
        if (info2) return OFL_WHY_ERROR;
    }
    TRY(cudaDeviceSynchronize());
    double t2 = now();
    down(c->a_out, da, n, n, c->lda, es);
    if (solve && info == 0) down(c->b_out, db, n, nrhs, c->ldb, es);
    store_int(c->info_out, c->int_bytes, info);
    t->in_s = t1 - t0, t->compute_s = t2 - t1, t->out_s = now() - t2;
    return 0;
}

// Symmetric / Hermitian eigenvalues, with eigenvectors when c->variant is set.
static int do_syevd(const ofl_call *c, ofl_gpu_times *t) {
    int p = pidx(c->prec), lwork = 0, info = 0;
    size_t es = ESZ[p], rs = RSZ[p];
    int64_t n = c->n;
    if (!valid(c->uplo, "UL") || !valid(c->job_a, "NV") || c->lda < n) return OFL_WHY_ARGS;
    cusolverEigMode_t jobz = c->job_a == 'V' ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;

    double t0 = now();
    DEV(da, 0, n * n * es);
    DEV(dw, 1, n * rs);
    DEV(dinfo, 4, sizeof(int));
    TRY(up(da, c->a, n, n, c->lda, es));
    double t1 = now();
    TRY(EVD_BS[p](g_solver, jobz, fill_of(c->uplo), (int)n, da, (int)n, dw, &lwork));
    DEV(dwork, 2, (size_t)lwork * es);
    TRY(EVD[p](g_solver, jobz, fill_of(c->uplo), (int)n, da, (int)n, dw, dwork, lwork, dinfo));
    TRY(fetch_info(dinfo, &info));
    if (info != 0) return OFL_WHY_ERROR;  // did not converge: let the CPU library handle it
    TRY(cudaDeviceSynchronize());
    double t2 = now();
    down(c->w_out, dw, n, 1, n, rs);
    if (c->job_a == 'V') down(c->a_out, da, n, n, c->lda, es);  // without vectors LAPACK leaves A undefined
    store_int(c->info_out, c->int_bytes, 0);
    t->in_s = t1 - t0, t->compute_s = t2 - t1, t->out_s = now() - t2;
    return 0;
}

// Singular value decomposition of an m x n matrix with m >= n.
static int do_gesvd(const ofl_call *c, ofl_gpu_times *t) {
    int p = pidx(c->prec), lwork = 0, info = 0;
    size_t es = ESZ[p], rs = RSZ[p];
    int64_t m = c->m, n = c->n;
    if (m < n || !valid(c->job_a, "NSA") || !valid(c->job_b, "NSA") || c->lda < m) return OFL_WHY_ARGS;
    int64_t ucols = c->job_a == 'A' ? m : n;
    if ((c->job_a != 'N' && c->ldu < m) || (c->job_b != 'N' && c->ldvt < n)) return OFL_WHY_ARGS;

    double t0 = now();
    DEV(da, 0, m * n * es);
    DEV(ds, 1, n * rs);
    DEV(drwork, 5, n * rs);
    DEV(dinfo, 4, sizeof(int));
    void *du = NULL, *dvt = NULL;
    if (c->job_a != 'N' && !(du = dev(6, (size_t)(m * ucols) * es))) return OFL_WHY_VRAM;
    if (c->job_b != 'N' && !(dvt = dev(7, (size_t)(n * n) * es))) return OFL_WHY_VRAM;
    TRY(up(da, c->a, m, n, c->lda, es));
    double t1 = now();
    TRY(SVD_BS[p](g_solver, (int)m, (int)n, &lwork));
    DEV(dwork, 2, (size_t)lwork * es);
    TRY(SVD[p](g_solver, c->job_a, c->job_b, (int)m, (int)n, da, (int)m, ds, du, (int)m, dvt, (int)n, dwork, lwork,
               drwork, dinfo));
    TRY(fetch_info(dinfo, &info));
    if (info != 0) return OFL_WHY_ERROR;
    TRY(cudaDeviceSynchronize());
    double t2 = now();
    down(c->w_out, ds, n, 1, n, rs);
    if (du) down(c->u_out, du, m, ucols, c->ldu, es);
    if (dvt) down(c->vt_out, dvt, n, n, c->ldvt, es);
    store_int(c->info_out, c->int_bytes, 0);
    t->in_s = t1 - t0, t->compute_s = t2 - t1, t->out_s = now() - t2;
    return 0;
}

// ---------------------------------------------------------------------------------------------
// Backend entry points
// ---------------------------------------------------------------------------------------------

static int backend_init(void) {
    int count = 0;
    struct cudaDeviceProp prop;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 1) return 1;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) snprintf(g_name, sizeof g_name, "%s", prop.name);
    if (cublasCreate(&g_blas) != CUBLAS_STATUS_SUCCESS) return 1;
    if (cusolverDnCreate(&g_solver) != CUSOLVER_STATUS_SUCCESS) return 1;
    return 0;
}

static int backend_run(const ofl_call *c, ofl_gpu_times *t) {
    int rc;
    switch (c->op) {
    case OFL_OP_GEMM: rc = do_gemm(c, t); break;
    case OFL_OP_SYRK: rc = do_syrk(c, t, 0); break;
    case OFL_OP_HERK: rc = do_syrk(c, t, 1); break;
    case OFL_OP_TRSM: rc = do_trsm(c, t); break;
    case OFL_OP_GETRF: rc = do_getrf(c, t, 0); break;
    case OFL_OP_GESV: rc = do_getrf(c, t, 1); break;
    case OFL_OP_POTRF: rc = do_potrf(c, t, 0); break;
    case OFL_OP_POSV: rc = do_potrf(c, t, 1); break;
    case OFL_OP_SYEVD: rc = do_syevd(c, t); break;
    case OFL_OP_GESVD: rc = do_gesvd(c, t); break;
    default: rc = OFL_WHY_ARGS; break;
    }
    if (rc == OFL_WHY_VRAM || rc == OFL_WHY_ERROR) {
        cudaGetLastError();
        pool_release();
    } else {
        pool_trim();
    }
    return rc;
}

static int backend_mem_info(uint64_t *free_bytes, uint64_t *total_bytes) {
    size_t f = 0, total = 0;
    if (cudaMemGetInfo(&f, &total) != cudaSuccess) return 1;
    *free_bytes = f;
    *total_bytes = total;
    return 0;
}

// GPU utilisation through NVML, loaded at run time so the backend works without it.
static int backend_utilization(void) {
    static int state;  // 0 untried, 1 ready, -1 unavailable
    static void *device;
    static int (*get_rates)(void *, unsigned *);
    if (state == 0) {
        state = -1;
        void *h = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        if (h) {
            int (*init)(void) = (int (*)(void))dlsym(h, "nvmlInit_v2");
            int (*by_index)(unsigned, void **) = (int (*)(unsigned, void **))dlsym(h, "nvmlDeviceGetHandleByIndex_v2");
            get_rates = (int (*)(void *, unsigned *))dlsym(h, "nvmlDeviceGetUtilizationRates");
            if (init && by_index && get_rates && init() == 0 && by_index(0, &device) == 0) state = 1;
        }
    }
    unsigned rates[2] = {0, 0};  // nvmlUtilization_t: gpu, memory
    if (state != 1 || get_rates(device, rates) != 0) return -1;
    return (int)rates[0];
}

static const char *backend_device_name(void) {
    return g_name;
}

OFL_EXPORT const struct ofl_backend *ofl_cuda_backend(void) {
    static const struct ofl_backend backend = {
        .abi = OFL_BACKEND_ABI,
        .init = backend_init,
        .run = backend_run,
        .mem_info = backend_mem_info,
        .utilization = backend_utilization,
        .device_name = backend_device_name,
    };
    return &backend;
}
