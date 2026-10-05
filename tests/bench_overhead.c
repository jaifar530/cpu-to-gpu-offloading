// SPDX-License-Identifier: AGPL-3.0-only
// Prints the average cost in nanoseconds of a tiny (4x4) DGEMM call.
// Run with and without the logger to see the logger's per-call overhead.
#include <cblas.h>
#include <stdio.h>
#include <time.h>

int main(void) {
    enum { N = 4, CALLS = 500000 };
    double a[N * N], b[N * N], c[N * N];
    for (int i = 0; i < N * N; i++) {
        a[i] = i * 0.5;
        b[i] = 1.0 / (i + 1);
    }
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < CALLS; i++)
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, a, N, b, N, 0.0, c, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ns = (t1.tv_sec - t0.tv_sec) * 1e9 + (t1.tv_nsec - t0.tv_nsec);
    printf("%.0f\n", ns / CALLS);
    return c[0] == 0.0;
}
