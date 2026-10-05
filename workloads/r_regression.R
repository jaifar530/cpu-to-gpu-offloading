# Statistics-style R script: linear models, cross products, PCA and matrix algebra.
set.seed(1)
n <- 400000; p <- 60
x <- matrix(rnorm(n * p), n, p)
beta <- rnorm(p)
y <- drop(x %*% beta) + rnorm(n)

fit <- lm.fit(cbind(1, x), y)                      # R's own QR (LINPACK), not LAPACK
xtx <- crossprod(x)                                # BLAS dsyrk
coef_ne <- solve(xtx, crossprod(x, y))             # LAPACK dgesv
pca <- prcomp(x[1:100000, ], center = TRUE)        # LAPACK SVD

m <- 2500
a <- matrix(rnorm(m * m), m, m)
prod <- a %*% t(a)                                 # BLAS dgemm
ch <- chol(prod / m + diag(m))                     # LAPACK dpotrf
ev <- eigen(prod[1:1000, 1:1000], symmetric = TRUE, only.values = TRUE)$values

cat(sprintf("lm coef error %.2e, normal-eq error %.2e, first PC sd %.3f, chol diag %.3f, top eigenvalue %.1f\n",
            max(abs(fit$coefficients[-1] - beta)), max(abs(coef_ne - beta)), pca$sdev[1], ch[1, 1], ev[1]))
