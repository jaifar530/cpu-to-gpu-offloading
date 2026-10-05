// SPDX-License-Identifier: AGPL-3.0-only
// Makes a fixed set of BLAS / LAPACK / FFTW calls and checks their results.
// Run without the logger it proves the libraries work; run under the logger it proves the
// wrappers pass every argument through unchanged. tests/check_trace.py then checks the trace.
#include <cblas.h>
#include <complex.h>
#include <fftw3.h>
#include <lapacke.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void dgemm_(const char *, const char *, const int *, const int *, const int *, const double *, const double *,
            const int *, const double *, const int *, const double *, double *, const int *);
void zgemm_(const char *, const char *, const int *, const int *, const int *, const void *, const void *,
            const int *, const void *, const int *, const void *, void *, const int *);

static int failures;
#define EXPECT(cond, what)                                 \
    do {                                                   \
        if (!(cond)) {                                     \
            printf("FAIL: %s\n", what);                    \
            failures++;                                    \
        }                                                  \
    } while (0)

static double *randmat(int rows, int cols) {
    double *a = malloc(sizeof(double) * rows * cols);
    for (int i = 0; i < rows * cols; i++) a[i] = rand() / (double)RAND_MAX - 0.5;
    return a;
}

// Reference result of C = A*B for column-major A (m x k), B (k x n).
static double gemm_error(int m, int n, int k, const double *a, const double *b, const double *c) {
    double err = 0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int l = 0; l < k; l++) s += a[i + l * m] * b[l + j * k];
            err = fmax(err, fabs(s - c[i + j * m]));
        }
    return err;
}

int main(void) {
    srand(1);
    enum { M = 64, N = 32, K = 16, S = 48 };

    // --- BLAS 3: the same A and B multiplied three times (buffer reuse), then with new contents ---
    double *a = randmat(M, K), *b = randmat(K, N), *c = calloc(M * N, sizeof(double));
    for (int rep = 0; rep < 3; rep++)
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, a, M, b, K, 0.0, c, M);
    EXPECT(gemm_error(M, N, K, a, b, c) < 1e-12, "cblas_dgemm");
    a[0] += 1.0;  // same address, different data: must not count as reuse
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, a, M, b, K, 0.0, c, M);
    EXPECT(gemm_error(M, N, K, a, b, c) < 1e-12, "cblas_dgemm after edit");

    // Fortran interface, beta != 0 (C is read as well as written).
    int m = M, n = N, k = K;
    double one = 1.0, *c2 = calloc(M * N, sizeof(double));
    dgemm_("N", "N", &m, &n, &k, &one, a, &m, b, &k, &one, c2, &m);
    EXPECT(gemm_error(M, N, K, a, b, c2) < 1e-12, "dgemm_");

    // Single precision.
    float *fa = malloc(sizeof(float) * S * S), *fc = malloc(sizeof(float) * S * S);
    for (int i = 0; i < S * S; i++) fa[i] = (i % (S + 1) == 0) ? 2.0f : 0.0f;  // 2 * identity
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, S, S, S, 1.0f, fa, S, fa, S, 0.0f, fc, S);
    EXPECT(fc[0] == 4.0f && fc[1] == 0.0f && fc[S * S - 1] == 4.0f, "cblas_sgemm");

    // Complex, Fortran interface: (i * I) * (i * I) = -I.
    double complex *za = calloc(S * S, sizeof(double complex)), *zc = calloc(S * S, sizeof(double complex));
    for (int i = 0; i < S; i++) za[i + i * S] = I;
    int s = S;
    double complex zone = 1.0, zzero = 0.0;
    zgemm_("N", "N", &s, &s, &s, &zone, za, &s, za, &s, &zzero, zc, &s);
    EXPECT(creal(zc[0]) == -1.0 && cimag(zc[0]) == 0.0 && zc[1] == 0.0, "zgemm_");

    // --- other BLAS levels ---
    double *sym = calloc(S * S, sizeof(double)), *g = randmat(S, K);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, S, K, 1.0, g, S, 0.0, sym, S);
    double g00 = 0;
    for (int l = 0; l < K; l++) g00 += g[l * S] * g[l * S];
    EXPECT(fabs(sym[0] - g00) < 1e-12, "cblas_dsyrk");

    double *x = randmat(K, 1), *y = calloc(M, sizeof(double));
    cblas_dgemv(CblasColMajor, CblasNoTrans, M, K, 1.0, a, M, x, 1, 0.0, y, 1);
    double y0 = 0;
    for (int l = 0; l < K; l++) y0 += a[l * M] * x[l];
    EXPECT(fabs(y[0] - y0) < 1e-12, "cblas_dgemv");

    double dot = cblas_ddot(K, x, 1, x, 1), ref = 0;
    for (int l = 0; l < K; l++) ref += x[l] * x[l];
    EXPECT(fabs(dot - ref) < 1e-12, "cblas_ddot");
    float fdot = cblas_sdot(S, fa, S + 1, fa, S + 1);  // the diagonal of 2*I
    EXPECT(fdot == 4.0f * S, "cblas_sdot");
    EXPECT(fabs(cblas_dnrm2(K, x, 1) - sqrt(ref)) < 1e-12, "cblas_dnrm2");
    double *y2 = calloc(K, sizeof(double));
    cblas_daxpy(K, 2.0, x, 1, y2, 1);
    EXPECT(fabs(y2[3] - 2.0 * x[3]) < 1e-15, "cblas_daxpy");
    cblas_dscal(K, 0.5, y2, 1);
    EXPECT(fabs(y2[3] - x[3]) < 1e-15, "cblas_dscal");

    // --- LAPACK ---
    double *spd = calloc(S * S, sizeof(double)), *rhs = randmat(S, 2), *sol = malloc(sizeof(double) * S * 2);
    double *q = randmat(S, S);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasNoTrans, S, S, 1.0, q, S, 0.0, spd, S);
    for (int i = 0; i < S; i++) {
        spd[i + i * S] += S;
        for (int j = 0; j < i; j++) spd[i + j * S] = spd[j + i * S];
    }
    double *lu = malloc(sizeof(double) * S * S);
    lapack_int *ipiv = malloc(sizeof(lapack_int) * S);
    memcpy(lu, spd, sizeof(double) * S * S);
    memcpy(sol, rhs, sizeof(double) * S * 2);
    EXPECT(LAPACKE_dgesv(LAPACK_COL_MAJOR, S, 2, lu, S, ipiv, sol, S) == 0, "dgesv info");
    double resid = 0;
    for (int i = 0; i < S; i++) {
        double r = -rhs[i];
        for (int j = 0; j < S; j++) r += spd[i + j * S] * sol[j];
        resid = fmax(resid, fabs(r));
    }
    EXPECT(resid < 1e-10, "dgesv residual");

    memcpy(lu, spd, sizeof(double) * S * S);
    EXPECT(LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', S, lu, S) == 0, "dpotrf info");
    EXPECT(lu[0] > 0 && fabs(lu[0] * lu[0] - spd[0]) < 1e-10, "dpotrf value");

    double *w = malloc(sizeof(double) * S), trace = 0, wsum = 0;
    memcpy(lu, spd, sizeof(double) * S * S);
    for (int i = 0; i < S; i++) trace += spd[i + i * S];
    EXPECT(LAPACKE_dsyev(LAPACK_COL_MAJOR, 'V', 'U', S, lu, S, w) == 0, "dsyev info");
    for (int i = 0; i < S; i++) wsum += w[i];
    EXPECT(fabs(wsum - trace) < 1e-8 * trace, "dsyev eigenvalue sum");

    double *sv = malloc(sizeof(double) * S), *u = malloc(sizeof(double) * S * S), *vt = malloc(sizeof(double) * S * S);
    memcpy(lu, spd, sizeof(double) * S * S);
    EXPECT(LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'A', S, S, lu, S, sv, u, S, vt, S) == 0, "dgesdd info");
    EXPECT(fabs(sv[0] - w[S - 1]) < 1e-8 * sv[0], "dgesdd largest singular value");

    // --- FFTW: the transform of a unit impulse is all ones ---
    enum { F = 1024 };
    fftw_complex *fin = fftw_alloc_complex(F), *fout = fftw_alloc_complex(F);
    fftw_plan p = fftw_plan_dft_1d(F, fin, fout, FFTW_FORWARD, FFTW_ESTIMATE);
    for (int rep = 0; rep < 2; rep++) {
        memset(fin, 0, sizeof(fftw_complex) * F);
        fin[0] = 1.0;
        fftw_execute(p);
    }
    EXPECT(cabs(fout[0] - 1.0) < 1e-12 && cabs(fout[F - 1] - 1.0) < 1e-12 && cabs(fout[5] - 1.0) < 1e-12,
           "fftw_execute");
    fftw_destroy_plan(p);

    enum { R0 = 32, R1 = 64 };
    float *rin = fftwf_alloc_real(R0 * R1);
    fftwf_complex *rout = fftwf_alloc_complex(R0 * (R1 / 2 + 1));
    fftwf_plan pf = fftwf_plan_dft_r2c_2d(R0, R1, rin, rout, FFTW_ESTIMATE);
    for (int i = 0; i < R0 * R1; i++) rin[i] = 1.0f;
    fftwf_execute(pf);
    EXPECT(cabsf(rout[0] - (float)(R0 * R1)) < 1e-2f && cabsf(rout[1]) < 1e-3f, "fftwf r2c 2d");
    fftwf_destroy_plan(pf);

    // Batched transform through the "many" planner and the new-array execute.
    enum { BN = 256, BATCH = 8 };
    fftw_complex *bin = fftw_alloc_complex(BN * BATCH), *bout = fftw_alloc_complex(BN * BATCH);
    int bn[1] = {BN};
    fftw_plan pb = fftw_plan_many_dft(1, bn, BATCH, bin, NULL, 1, BN, bout, NULL, 1, BN, FFTW_FORWARD, FFTW_ESTIMATE);
    memset(bin, 0, sizeof(fftw_complex) * BN * BATCH);
    for (int t = 0; t < BATCH; t++) bin[t * BN] = t + 1.0;
    fftw_execute_dft(pb, bin, bout);
    EXPECT(cabs(bout[0] - 1.0) < 1e-12 && cabs(bout[(BATCH - 1) * BN + 7] - BATCH) < 1e-12, "fftw many");
    fftw_destroy_plan(pb);

    printf(failures ? "TEST_CALLS_FAILED (%d)\n" : "TEST_CALLS_OK\n", failures);
    return failures != 0;
}
