"""Signal-processing style NumPy script: FFT filtering and spectrograms.
NumPy's FFT (pocketfft) is compiled into NumPy itself, so no library boundary exists to intercept."""
import numpy as np

rng = np.random.default_rng(0)
rate, seconds = 48000, 120
t = np.arange(rate * seconds) / rate
signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 3000 * t) + rng.standard_normal(t.size)

# Low-pass filter in the frequency domain, block by block.
block = 1 << 16
freqs = np.fft.rfftfreq(block, 1 / rate)
mask = freqs < 1000
out = np.empty_like(signal[: signal.size // block * block])
for start in range(0, out.size, block):
    spec = np.fft.rfft(signal[start:start + block])
    out[start:start + block] = np.fft.irfft(spec * mask, block)

# Spectrogram with overlapping windows.
win = 2048
frames = np.lib.stride_tricks.sliding_window_view(out, win)[::512]
power = np.abs(np.fft.rfft(frames * np.hanning(win), axis=1)) ** 2
peak_hz = np.fft.rfftfreq(win, 1 / rate)[power.mean(axis=0).argmax()]

print(f"filtered {out.size / rate:.0f} s of audio, {frames.shape[0]} spectrogram frames, peak at {peak_hz:.0f} Hz")
