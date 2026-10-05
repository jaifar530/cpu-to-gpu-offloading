// SPDX-License-Identifier: AGPL-3.0-only
// FFTW wrappers for one precision. Included by fftw.c with FX, RB, FN_PLAN and FN_EXEC set.

// ---- complex to complex ----
PLANNER(plan_dft_1d, (int n, void *in, void *out, int sign, unsigned flags), (n, in, out, sign, flags),
        reg_n(plan, K_C2C, 1, n, 0, 0, in, out, RB))
PLANNER(plan_dft_2d, (int n0, int n1, void *in, void *out, int sign, unsigned flags), (n0, n1, in, out, sign, flags),
        reg_n(plan, K_C2C, 2, n0, n1, 0, in, out, RB))
PLANNER(plan_dft_3d, (int n0, int n1, int n2, void *in, void *out, int sign, unsigned flags),
        (n0, n1, n2, in, out, sign, flags), reg_n(plan, K_C2C, 3, n0, n1, n2, in, out, RB))
PLANNER(plan_dft, (int rank, const int *n, void *in, void *out, int sign, unsigned flags),
        (rank, n, in, out, sign, flags), reg_v(plan, K_C2C, rank, n, 1, in, out, RB))
PLANNER(plan_many_dft,
        (int rank, const int *n, int howmany, void *in, const int *inembed, int istride, int idist, void *out,
         const int *onembed, int ostride, int odist, int sign, unsigned flags),
        (rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, sign, flags),
        reg_v(plan, K_C2C, rank, n, howmany, in, out, RB))
PLANNER(plan_guru_dft,
        (int rank, const iodim *dims, int hrank, const iodim *hdims, void *in, void *out, int sign, unsigned flags),
        (rank, dims, hrank, hdims, in, out, sign, flags), reg_guru(plan, K_C2C, rank, dims, hrank, hdims, in, out, RB))

// ---- real to complex ----
PLANNER(plan_dft_r2c_1d, (int n, void *in, void *out, unsigned flags), (n, in, out, flags),
        reg_n(plan, K_R2C, 1, n, 0, 0, in, out, RB))
PLANNER(plan_dft_r2c_2d, (int n0, int n1, void *in, void *out, unsigned flags), (n0, n1, in, out, flags),
        reg_n(plan, K_R2C, 2, n0, n1, 0, in, out, RB))
PLANNER(plan_dft_r2c_3d, (int n0, int n1, int n2, void *in, void *out, unsigned flags), (n0, n1, n2, in, out, flags),
        reg_n(plan, K_R2C, 3, n0, n1, n2, in, out, RB))
PLANNER(plan_dft_r2c, (int rank, const int *n, void *in, void *out, unsigned flags), (rank, n, in, out, flags),
        reg_v(plan, K_R2C, rank, n, 1, in, out, RB))
PLANNER(plan_many_dft_r2c,
        (int rank, const int *n, int howmany, void *in, const int *inembed, int istride, int idist, void *out,
         const int *onembed, int ostride, int odist, unsigned flags),
        (rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, flags),
        reg_v(plan, K_R2C, rank, n, howmany, in, out, RB))
PLANNER(plan_guru_dft_r2c,
        (int rank, const iodim *dims, int hrank, const iodim *hdims, void *in, void *out, unsigned flags),
        (rank, dims, hrank, hdims, in, out, flags), reg_guru(plan, K_R2C, rank, dims, hrank, hdims, in, out, RB))

// ---- complex to real ----
PLANNER(plan_dft_c2r_1d, (int n, void *in, void *out, unsigned flags), (n, in, out, flags),
        reg_n(plan, K_C2R, 1, n, 0, 0, in, out, RB))
PLANNER(plan_dft_c2r_2d, (int n0, int n1, void *in, void *out, unsigned flags), (n0, n1, in, out, flags),
        reg_n(plan, K_C2R, 2, n0, n1, 0, in, out, RB))
PLANNER(plan_dft_c2r_3d, (int n0, int n1, int n2, void *in, void *out, unsigned flags), (n0, n1, n2, in, out, flags),
        reg_n(plan, K_C2R, 3, n0, n1, n2, in, out, RB))
PLANNER(plan_dft_c2r, (int rank, const int *n, void *in, void *out, unsigned flags), (rank, n, in, out, flags),
        reg_v(plan, K_C2R, rank, n, 1, in, out, RB))
PLANNER(plan_many_dft_c2r,
        (int rank, const int *n, int howmany, void *in, const int *inembed, int istride, int idist, void *out,
         const int *onembed, int ostride, int odist, unsigned flags),
        (rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, flags),
        reg_v(plan, K_C2R, rank, n, howmany, in, out, RB))
PLANNER(plan_guru_dft_c2r,
        (int rank, const iodim *dims, int hrank, const iodim *hdims, void *in, void *out, unsigned flags),
        (rank, dims, hrank, hdims, in, out, flags), reg_guru(plan, K_C2R, rank, dims, hrank, hdims, in, out, RB))

// ---- real to real (DCT / DST / halfcomplex) ----
PLANNER(plan_r2r_1d, (int n, void *in, void *out, int kind, unsigned flags), (n, in, out, kind, flags),
        reg_n(plan, K_R2R, 1, n, 0, 0, in, out, RB))
PLANNER(plan_r2r_2d, (int n0, int n1, void *in, void *out, int kind0, int kind1, unsigned flags),
        (n0, n1, in, out, kind0, kind1, flags), reg_n(plan, K_R2R, 2, n0, n1, 0, in, out, RB))
PLANNER(plan_r2r_3d, (int n0, int n1, int n2, void *in, void *out, int kind0, int kind1, int kind2, unsigned flags),
        (n0, n1, n2, in, out, kind0, kind1, kind2, flags), reg_n(plan, K_R2R, 3, n0, n1, n2, in, out, RB))
PLANNER(plan_r2r, (int rank, const int *n, void *in, void *out, const void *kind, unsigned flags),
        (rank, n, in, out, kind, flags), reg_v(plan, K_R2R, rank, n, 1, in, out, RB))
PLANNER(plan_many_r2r,
        (int rank, const int *n, int howmany, void *in, const int *inembed, int istride, int idist, void *out,
         const int *onembed, int ostride, int odist, const void *kind, unsigned flags),
        (rank, n, howmany, in, inembed, istride, idist, out, onembed, ostride, odist, kind, flags),
        reg_v(plan, K_R2R, rank, n, howmany, in, out, RB))

// ---- execution ----
EXEC(execute, (void *plan), (plan), NULL, NULL)
EXEC(execute_dft, (void *plan, void *in, void *out), (plan, in, out), in, out)
EXEC(execute_dft_r2c, (void *plan, void *in, void *out), (plan, in, out), in, out)
EXEC(execute_dft_c2r, (void *plan, void *in, void *out), (plan, in, out), in, out)
EXEC(execute_r2r, (void *plan, void *in, void *out), (plan, in, out), in, out)

OFL_EXPORT void FX(destroy_plan)(void *plan) {
    typedef void (*fn_t)(void *);
    static fn_t real;
    if (__builtin_expect(!real, 0))
        real = (fn_t)ofl_resolve(FXS(destroy_plan), (void *)FX(destroy_plan), __builtin_return_address(0));
    plan_del(plan);
    real(plan);
}
