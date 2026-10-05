% Octave script mixing FFT filtering (FFTW) with dense linear algebra (BLAS / LAPACK).
rand("seed", 1); randn("seed", 1);

% FFT low-pass filtering, block by block.
rate = 48000; block = 2^16; blocks = 300;
keep = [ones(1, 1400), zeros(1, block - 2799), ones(1, 1399)];
energy = 0;
for i = 1:blocks
  x = randn(1, block);
  y = real(ifft(fft(x) .* keep));
  energy = energy + sum(y .^ 2);
end

% 2-D FFT of an image stack.
stack = rand(1024, 1024, 12);
spec = 0;
for i = 1:12
  spec = spec + abs(fft2(stack(:, :, i)))(1, 1);
end

% Dense linear algebra.
n = 2500;
a = randn(n); a = a * a' / n + eye(n);
b = randn(n, 4);
sol = a \ b;
resid = max(abs(a * sol - b)(:));
ev = eig(a(1:1200, 1:1200));
p = a * a;

printf("filter energy %.3e, spectrum %.1f, residual %.2e, largest eigenvalue %.3f, trace %.1f\n", ...
       energy, spec, resid, max(ev), trace(p));
