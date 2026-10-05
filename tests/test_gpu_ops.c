// SPDX-License-Identifier: AGPL-3.0-only
// Calls every GPU-capable operation in all four precisions and checks each result against an
// independent reference computed here in plain loops (products, residuals, reconstructions).
// The checks do not care which device produced the result, so the same program validates the
// CPU libraries (run plain) and the GPU path (run under the preload library with
// GPU_OFFLOAD_FORCE=gpu).
#define _GNU_SOURCE
#include <complex.h>
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef double complex Z;
enum { ROW = 101, COL = 102, NOTRANS = 111, TRANS = 112, CONJ = 113, UPPER = 121, LOWER = 122, NONUNIT = 131,
       UNIT = 132, LEFT = 141, RIGHT = 142 };

static char P;       // precision under test: s, d, c, z
static size_t ES;    // bytes per element
static int CX, DBL;  // complex? double?
static double TOL;
static int fails, checks;

static void *lookup(const char *fmt, const char *base) {
    char name[64];
    snprintf(name, sizeof name, fmt, P, base);
    void *p = dlsym(RTLD_DEFAULT, name);
    if (!p) {
        printf("symbol %s not found\n", name);
        exit(1);
    }
    return p;
}
#define F(base) lookup("%c%s_", base)
#define LE(base) lookup("LAPACKE_%c%s", base)
#define CB(base) lookup("cblas_%c%s", base)

// ---- typed buffers <-> double complex ----

static Z get(const void *buf, size_t i) {
    if (DBL) return CX ? ((const double *)buf)[2 * i] + I * ((const double *)buf)[2 * i + 1] : ((const double *)buf)[i];
    return CX ? ((const float *)buf)[2 * i] + I * ((const float *)buf)[2 * i + 1] : ((const float *)buf)[i];
}

static void put(void *buf, size_t i, Z v) {
    if (DBL) {
        ((double *)buf)[CX ? 2 * i : i] = creal(v);
        if (CX) ((double *)buf)[2 * i + 1] = cimag(v);
    } else {
        ((float *)buf)[CX ? 2 * i : i] = (float)creal(v);
        if (CX) ((float *)buf)[2 * i + 1] = (float)cimag(v);
    }
}

static double rnd(void) {
    return rand() / (double)RAND_MAX - 0.5;
}

// Random typed matrix with leading dimension ld (the padding rows are random too).
static void *mk(int rows, int cols, int ld) {
    (void)rows;
    void *m = malloc((size_t)ld * cols * ES);
    for (size_t i = 0; i < (size_t)ld * cols; i++) put(m, i, rnd() + (CX ? I * rnd() : 0));
    return m;
}

static void *dup(const void *m, int cols, int ld) {
    void *c = malloc((size_t)ld * cols * ES);
    memcpy(c, m, (size_t)ld * cols * ES);
    return c;
}

// Typed scalar for Fortran-style by-reference arguments.
static void *sc(Z v) {
    static double store[8][2];
    static int next;
    void *p = store[next++ & 7];
    put(p, 0, v);
    return p;
}

// Dense column-major copy in double complex.
static Z *zin(const void *buf, int rows, int cols, int ld) {
    Z *z = malloc(sizeof(Z) * rows * cols);
    for (int j = 0; j < cols; j++)
        for (int i = 0; i < rows; i++) z[i + j * rows] = get(buf, (size_t)i + (size_t)j * ld);
    return z;
}

// op(A) for a rows x cols matrix: 'N' copy, 'T' transpose, 'C' conjugate transpose.
static Z *zop(const Z *a, int rows, int cols, char t) {
    Z *r = malloc(sizeof(Z) * rows * cols);
    for (int j = 0; j < cols; j++)
        for (int i = 0; i < rows; i++) {
            if (t == 'N') r[i + j * rows] = a[i + j * rows];
            else r[j + i * cols] = t == 'C' ? conj(a[i + j * rows]) : a[i + j * rows];
        }
    return r;
}

static Z *zmul(const Z *x, int m, int k, const Z *y, int n) {  // (m x k) * (k x n)
    Z *r = calloc((size_t)m * n, sizeof(Z));
    for (int j = 0; j < n; j++)
        for (int l = 0; l < k; l++)
            for (int i = 0; i < m; i++) r[i + j * m] += x[i + l * m] * y[l + j * k];
    return r;
}

static double zdiff(const Z *x, const Z *y, size_t n) {
    double d = 0;
    for (size_t i = 0; i < n; i++) d = fmax(d, cabs(x[i] - y[i]));
    return d;
}

static double zmax(const Z *x, size_t n) {
    double d = 0;
    for (size_t i = 0; i < n; i++) d = fmax(d, cabs(x[i]));
    return d;
}

static void check(const char *what, double err, double scale) {
    checks++;
    if (!(err <= TOL * (scale > 1 ? scale : 1))) {
        printf("FAIL %c %s: error %.3e (scale %.3e)\n", P, what, err, scale);
        fails++;
    }
}

// Well-conditioned n x n matrix with leading dimension ld; Hermitian positive definite if asked.
static void *conditioned(int n, int ld, int hermitian) {
    void *m = mk(n, n, ld);
    for (int j = 0; j < n; j++) {
        if (hermitian)
            for (int i = 0; i < j; i++) put(m, (size_t)j + (size_t)i * ld, conj(get(m, (size_t)i + (size_t)j * ld)));
        put(m, (size_t)j + (size_t)j * ld, n);
    }
    return m;
}

// ---- BLAS ----

static void test_gemm(void) {
    enum { M = 37, N = 29, K = 23 };
    const char trans[3] = {'N', 'T', CX ? 'C' : 'T'};
    for (int ia = 0; ia < 3; ia++)
        for (int ib = 0; ib < 3; ib += 2) {
            char ta = trans[ia], tb = trans[ib];
            int ra = ta == 'N' ? M : K, ca = ta == 'N' ? K : M, rb = tb == 'N' ? K : N, cb = tb == 'N' ? N : K;
            int lda = ra + 3, ldb = rb, ldc = M + 2, m = M, n = N, k = K;
            void *a = mk(ra, ca, lda), *b = mk(rb, cb, ldb), *c = mk(M, N, ldc);
            Z alpha = 1.25 - (CX ? 0.5 * I : 0), beta = ia == 1 ? 0.5 : 0;  // beta != 0: C is an input too
            Z *za = zin(a, ra, ca, lda), *zb = zin(b, rb, cb, ldb), *zc0 = zin(c, M, N, ldc);
            Z *oa = zop(za, ra, ca, ta), *ob = zop(zb, rb, cb, tb), *ref = zmul(oa, M, K, ob, N);
            for (int i = 0; i < M * N; i++) ref[i] = alpha * ref[i] + beta * zc0[i];
            ((void (*)(const char *, const char *, const int *, const int *, const int *, const void *, const void *,
                       const int *, const void *, const int *, const void *, void *, const int *, size_t, size_t))F(
                "gemm"))(&ta, &tb, &m, &n, &k, sc(alpha), a, &lda, b, &ldb, sc(beta), c, &ldc, 1, 1);
            Z *zc = zin(c, M, N, ldc);
            char what[32];
            snprintf(what, sizeof what, "gemm %c%c", ta, tb);
            check(what, zdiff(zc, ref, M * N), zmax(ref, M * N));
            free(a), free(b), free(c), free(za), free(zb), free(zc0), free(oa), free(ob), free(ref), free(zc);
        }
}

static void test_gemm_rowmajor(void) {
    enum { M = 21, N = 34, K = 18 };
    // Row-major A is M x K with row length lda; viewed column-major it is the transpose.
    int lda = K + 1, ldb = N, ldc = N + 2;
    void *a = mk(lda, M, lda), *b = mk(ldb, N, ldb), *c = mk(ldc, M, ldc);  // B is N x K, used transposed
    Z alpha = 0.75, beta = 0;
    Z *ref = calloc(M * N, sizeof(Z));
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int l = 0; l < K; l++)
                ref[i + j * M] += alpha * get(a, (size_t)i * lda + l) * get(b, (size_t)j * ldb + l);  // B^T
    void *f = CB("gemm");
    if (P == 's')
        ((void (*)(int, int, int, int, int, int, float, const void *, int, const void *, int, float, void *, int))f)(
            ROW, NOTRANS, TRANS, M, N, K, (float)creal(alpha), a, lda, b, ldb, 0.0f, c, ldc);
    else if (P == 'd')
        ((void (*)(int, int, int, int, int, int, double, const void *, int, const void *, int, double, void *, int))f)(
            ROW, NOTRANS, TRANS, M, N, K, creal(alpha), a, lda, b, ldb, 0.0, c, ldc);
    else
        ((void (*)(int, int, int, int, int, int, const void *, const void *, int, const void *, int, const void *,
                   void *, int))f)(ROW, NOTRANS, TRANS, M, N, K, sc(alpha), a, lda, b, ldb, sc(beta), c, ldc);
    Z *got = malloc(sizeof(Z) * M * N);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) got[i + j * M] = get(c, (size_t)i * ldc + j);
    check("cblas gemm row-major", zdiff(got, ref, M * N), zmax(ref, M * N));
    free(a), free(b), free(c), free(ref), free(got);
}

// C = alpha * op(A) * op(A)^T (syrk) or ^H (herk): only one triangle may change.
static void test_rank_k(int herk, char uplo, char trans) {
    enum { N = 31, K = 19 };
    int ra = trans == 'N' ? N : K, ca = trans == 'N' ? K : N, lda = ra + 1, ldc = N + 3, n = N, k = K;
    void *a = mk(ra, ca, lda), *c = mk(N, N, ldc);
    Z *zc0 = zin(c, N, N, ldc), *za = zin(a, ra, ca, lda);
    // Reference: X * X^T or X * X^H with X = op(A), N x K.
    Z *x = zop(za, ra, ca, trans), *xt = zop(x, N, K, herk ? 'C' : 'T'), *ref = zmul(x, N, K, xt, N);
    Z alpha = 0.5, beta = 0;
    ((void (*)(const char *, const char *, const int *, const int *, const void *, const void *, const int *,
               const void *, void *, const int *, size_t, size_t))F(herk ? "herk" : "syrk"))(
        &uplo, &trans, &n, &k, sc(alpha), a, &lda, sc(beta), c, &ldc, 1, 1);
    Z *zc = zin(c, N, N, ldc);
    double err = 0, untouched = 0;
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++) {
            int in_triangle = uplo == 'U' ? i <= j : i >= j;
            if (in_triangle) err = fmax(err, cabs(zc[i + j * N] - alpha * ref[i + j * N]));
            else untouched = fmax(untouched, cabs(zc[i + j * N] - zc0[i + j * N]));
        }
    char what[48];
    snprintf(what, sizeof what, "%s %c%c", herk ? "herk" : "syrk", uplo, trans);
    check(what, err, zmax(ref, N * N));
    snprintf(what, sizeof what, "%s %c%c other triangle untouched", herk ? "herk" : "syrk", uplo, trans);
    check(what, untouched > 0 ? 1e300 : 0, 1);
    free(a), free(c), free(zc0), free(za), free(x), free(xt), free(ref), free(zc);
}

// Solves op(A) X = B (side L) or X op(A) = B (side R), then checks by multiplying back.
static void test_trsm(char side, char uplo, char trans, char diag) {
    enum { M = 26, N = 17 };
    int ka = side == 'L' ? M : N, lda = ka + 2, ldb = M + 1, m = M, n = N;
    void *a = conditioned(ka, lda, 0), *b = mk(M, N, ldb);
    if (diag == 'U')  // a unit diagonal needs small off-diagonal entries to stay well conditioned
        for (size_t i = 0; i < (size_t)lda * ka; i++) put(a, i, 0.1 * get(a, i));
    Z *zb0 = zin(b, M, N, ldb), *full = zin(a, ka, ka, lda);
    for (int j = 0; j < ka; j++)  // the triangle trsm actually uses
        for (int i = 0; i < ka; i++) {
            if (uplo == 'U' ? i > j : i < j) full[i + j * ka] = 0;
            if (i == j && diag == 'U') full[i + j * ka] = 1;
        }
    ((void (*)(const char *, const char *, const char *, const char *, const int *, const int *, const void *,
               const void *, const int *, void *, const int *, size_t, size_t, size_t, size_t))F("trsm"))(
        &side, &uplo, &trans, &diag, &m, &n, sc(1.0), a, &lda, b, &ldb, 1, 1, 1, 1);
    Z *x = zin(b, M, N, ldb), *oa = zop(full, ka, ka, trans);
    Z *back = side == 'L' ? zmul(oa, M, M, x, N) : zmul(x, M, N, oa, N);
    char what[32];
    snprintf(what, sizeof what, "trsm %c%c%c%c", side, uplo, trans, diag);
    check(what, zdiff(back, zb0, M * N), zmax(zb0, M * N) * ka);
    free(a), free(b), free(zb0), free(full), free(x), free(oa), free(back);
}

// ---- LAPACK ----

static void test_getrf(int m, int n) {
    int big = m > n ? m : n, lda = big + 2, mn = m < n ? m : n, info = -1;
    void *a = conditioned(big, lda, 0);  // the m x n matrix under test is its top-left corner
    Z *za0 = zin(a, m, n, lda);
    int *ipiv = malloc(sizeof(int) * mn);
    ((void (*)(const int *, const int *, void *, const int *, int *, int *))F("getrf"))(&m, &n, a, &lda, ipiv, &info);
    Z *lu = zin(a, m, n, lda), *l = calloc((size_t)m * mn, sizeof(Z)), *u = calloc((size_t)mn * n, sizeof(Z));
    for (int j = 0; j < mn; j++)
        for (int i = j; i < m; i++) l[i + j * m] = i == j ? 1 : lu[i + j * m];
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= j && i < mn; i++) u[i + j * mn] = lu[i + j * m];
    Z *plu = zmul(l, m, mn, u, n);
    for (int i = mn - 1; i >= 0; i--)  // undo the row interchanges
        for (int j = 0; j < n; j++) {
            Z t = plu[i + j * m];
            plu[i + j * m] = plu[ipiv[i] - 1 + j * m];
            plu[ipiv[i] - 1 + j * m] = t;
        }
    char what[32];
    snprintf(what, sizeof what, "getrf %dx%d", m, n);
    check(what, info == 0 ? zdiff(plu, za0, (size_t)m * n) : 1e300, zmax(za0, (size_t)m * n) * n);
    free(a), free(za0), free(ipiv), free(lu), free(l), free(u), free(plu);
}

static void test_solve(int spd) {
    enum { N = 44, NRHS = 3 };
    int lda = N + 1, ldb = N + 2, info;
    void *a = conditioned(N, lda, spd), *b = mk(N, NRHS, ldb);
    Z *za = zin(a, N, N, lda), *zb = zin(b, N, NRHS, ldb);
    if (spd) {
        info = ((int (*)(int, char, int, int, void *, int, void *, int))LE("posv"))(COL, 'L', N, NRHS, a, lda, b, ldb);
    } else {
        int ipiv[N];
        info = ((int (*)(int, int, int, void *, int, int *, void *, int))LE("gesv"))(COL, N, NRHS, a, lda, ipiv, b, ldb);
    }
    Z *x = zin(b, N, NRHS, ldb), *ax = zmul(za, N, N, x, NRHS);
    check(spd ? "posv residual" : "gesv residual", info == 0 ? zdiff(ax, zb, N * NRHS) : 1e300, N * zmax(zb, N * NRHS));
    free(a), free(b), free(za), free(zb), free(x), free(ax);
}

static void test_potrf(char uplo) {
    enum { N = 39 };
    int n = N, lda = N + 4, info = -1;
    void *a = conditioned(N, lda, 1);
    // Put junk in the triangle potrf must not touch, to see that it survives.
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
            if (uplo == 'U' ? i > j : i < j) put(a, (size_t)i + (size_t)j * lda, 7.0);
    void *a_before = dup(a, N, lda);
    Z *za = zin(a, N, N, lda);
    for (int j = 0; j < N; j++)  // the Hermitian matrix that the referenced triangle describes
        for (int i = 0; i < N; i++)
            if (uplo == 'U' ? i > j : i < j) za[i + j * N] = conj(za[j + i * N]);
    ((void (*)(const char *, const int *, void *, const int *, int *, size_t))F("potrf"))(&uplo, &n, a, &lda, &info, 1);
    Z *f = zin(a, N, N, lda), *before = zin(a_before, N, N, lda);
    double untouched = 0;
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
            if (uplo == 'U' ? i > j : i < j) {
                untouched = fmax(untouched, cabs(f[i + j * N] - before[i + j * N]));
                f[i + j * N] = 0;
            }
    Z *fh = zop(f, N, N, 'C'), *back = uplo == 'U' ? zmul(fh, N, N, f, N) : zmul(f, N, N, fh, N);
    char what[40];
    snprintf(what, sizeof what, "potrf %c", uplo);
    check(what, info == 0 ? zdiff(back, za, N * N) : 1e300, N * zmax(za, N * N));
    snprintf(what, sizeof what, "potrf %c other triangle untouched", uplo);
    check(what, untouched > 0 ? 1e300 : 0, 1);
    free(a), free(a_before), free(za), free(f), free(before), free(fh), free(back);
}

static double real_at(const void *w, int i) {
    return DBL ? ((const double *)w)[i] : ((const float *)w)[i];
}

static void test_eigen(const char *routine) {
    enum { N = 33 };
    int lda = N + 1;
    void *a0 = conditioned(N, lda, 1);
    Z *za = zin(a0, N, N, lda);
    char name[16];
    snprintf(name, sizeof name, "%s%s", CX ? "he" : "sy", routine);  // syev / syevd / heev / heevd
    double wn[N], wv[N];
    for (int pass = 0; pass < 2; pass++) {
        void *a = dup(a0, N, lda), *w = malloc(N * sizeof(double));
        int info = ((int (*)(int, char, char, int, void *, int, void *))LE(name))(COL, pass ? 'V' : 'N', 'U', N, a, lda, w);
        for (int i = 0; i < N; i++) (pass ? wv : wn)[i] = real_at(w, i);
        char what[48];
        if (pass) {
            Z *v = zin(a, N, N, lda), *av = zmul(za, N, N, v, N);
            double err = 0;
            for (int j = 0; j < N; j++)
                for (int i = 0; i < N; i++) err = fmax(err, cabs(av[i + j * N] - wv[j] * v[i + j * N]));
            snprintf(what, sizeof what, "%s vectors: A v = lambda v", name);
            check(what, info == 0 ? err : 1e300, N * zmax(za, N * N));
            free(v), free(av);
        }
        free(a), free(w);
    }
    double err = 0, order = 0;
    for (int i = 0; i < N; i++) {
        err = fmax(err, fabs(wn[i] - wv[i]));
        if (i && wn[i] < wn[i - 1]) order = 1e300;
    }
    char what[48];
    snprintf(what, sizeof what, "%s values agree and ascend", name);
    check(what, err + order, N * zmax(za, N * N));
    free(a0), free(za);
}

// jobz for gesdd; for gesvd the same letter is used for both jobu and jobvt.
static void test_svd(int use_gesdd, char job, int m, int n) {
    int lda = m + 1, mn = m < n ? m : n, ucols = job == 'A' ? m : mn, vrows = job == 'A' ? n : mn, info;
    void *a0 = mk(m, n, lda), *a = dup(a0, n, lda);
    Z *za = zin(a0, m, n, lda);
    void *s = malloc(mn * sizeof(double)), *u = malloc((size_t)m * m * ES), *vt = malloc((size_t)n * n * ES);
    if (use_gesdd) {
        info = ((int (*)(int, char, int, int, void *, int, void *, void *, int, void *, int))LE("gesdd"))(
            COL, job, m, n, a, lda, s, u, m, vt, vrows);
    } else {
        void *superb = malloc(mn * sizeof(double));
        info = ((int (*)(int, char, char, int, int, void *, int, void *, void *, int, void *, int, void *))LE("gesvd"))(
            COL, job, job, m, n, a, lda, s, u, m, vt, vrows, superb);
        free(superb);
    }
    char what[64];
    double order = 0;
    for (int i = 1; i < mn; i++)
        if (real_at(s, i) > real_at(s, i - 1) || real_at(s, i) < 0) order = 1e300;
    snprintf(what, sizeof what, "%s %c %dx%d values descend", use_gesdd ? "gesdd" : "gesvd", job, m, n);
    check(what, info == 0 ? order : 1e300, 1);
    // Singular values must match the eigenvalues of A^H A regardless of device: compare the sum of squares
    // with the squared Frobenius norm of A.
    double fro = 0, sum = 0;
    for (int i = 0; i < m * n; i++) fro += creal(za[i] * conj(za[i]));
    for (int i = 0; i < mn; i++) sum += real_at(s, i) * real_at(s, i);
    snprintf(what, sizeof what, "%s %c %dx%d sum of squares", use_gesdd ? "gesdd" : "gesvd", job, m, n);
    check(what, fabs(fro - sum), fro * mn);
    if (job != 'N') {  // A = U * diag(S) * VT using the first mn columns / rows
        Z *zu = zin(u, m, ucols, m), *zvt = zin(vt, vrows, n, vrows), *us = malloc(sizeof(Z) * m * mn);
        Z *vtop = malloc(sizeof(Z) * mn * n);
        for (int j = 0; j < mn; j++)
            for (int i = 0; i < m; i++) us[i + j * m] = zu[i + j * m] * real_at(s, j);
        for (int j = 0; j < n; j++)
            for (int i = 0; i < mn; i++) vtop[i + j * mn] = zvt[i + j * vrows];
        Z *back = zmul(us, m, mn, vtop, n);
        snprintf(what, sizeof what, "%s %c %dx%d A = U S VT", use_gesdd ? "gesdd" : "gesvd", job, m, n);
        check(what, zdiff(back, za, (size_t)m * n), mn * zmax(za, (size_t)m * n));
        free(zu), free(zvt), free(us), free(vtop), free(back);
    }
    free(a0), free(a), free(za), free(s), free(u), free(vt);
}

int main(void) {
    srand(7);
    for (const char *p = "sdcz"; *p; p++) {
        P = *p;
        CX = P == 'c' || P == 'z';
        DBL = P == 'd' || P == 'z';
        ES = (DBL ? 8 : 4) * (CX ? 2 : 1);
        TOL = DBL ? 1e-11 : 5e-4;

        test_gemm();
        test_gemm_rowmajor();
        test_rank_k(0, 'U', 'N');
        test_rank_k(0, 'L', 'T');
        if (CX) {
            test_rank_k(1, 'U', 'N');
            test_rank_k(1, 'L', 'C');
        }
        test_trsm('L', 'U', 'N', 'N');
        test_trsm('R', 'L', 'T', 'U');
        test_trsm('L', 'L', CX ? 'C' : 'T', 'N');
        test_getrf(40, 40);
        test_getrf(50, 30);
        test_solve(0);
        test_solve(1);
        test_potrf('U');
        test_potrf('L');
        test_eigen("ev");
        test_eigen("evd");
        test_svd(1, 'N', 40, 25);
        test_svd(1, 'S', 40, 25);
        test_svd(1, 'A', 40, 25);
        test_svd(0, 'S', 36, 36);
        test_svd(1, 'S', 25, 40);  // wide matrix: not handled by the GPU path, must fall back cleanly
    }
    printf("%d checks, %d failed\n", checks, fails);
    printf(fails ? "TEST_GPU_OPS_FAILED\n" : "TEST_GPU_OPS_OK\n");
    return fails != 0;
}
