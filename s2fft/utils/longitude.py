"""Map physical GL longitude FFTs to the fixed spherical harmonic order domain."""

import numpy as np

from s2fft.sampling import s2_samples as samples


def _validate_gl_shape(f, L):
    """Validate a single physical GL map and return its longitude count."""
    if f.ndim != 2 or f.shape[0] != L:
        raise ValueError("GL input must have shape (L, nphi).")
    return samples.nphi_equiang(L, "gl", f.shape[-1])


def _forward_gl(f, L, reality, xp=np):
    """Select full-bandwidth GL orders, normalized for existing latitude weights."""
    nphi = _validate_gl_shape(f, L)
    if reality:
        # rfft bins are m >= 0. Retain 0..L-1, excluding Nyquist and higher orders.
        positive = xp.fft.rfft(xp.real(f), axis=-1, norm="backward")[..., :L]
        ftm = xp.pad(positive, ((0, 0), (L - 1, 0)))
    else:
        spectrum = xp.fft.fftshift(xp.fft.fft(f, axis=-1, norm="backward"), axes=-1)
        center = nphi // 2
        ftm = spectrum[..., center - L + 1 : center + L]
    # GL latitude weights (including precomputed kernels) contain 2pi/(2L-1).
    return ftm * ((2 * L - 1) / nphi)


def _inverse_gl(ftm, L, nphi, reality, xp=np):
    """Embed fixed GL orders into a physical FFT spectrum with zero extra modes."""
    nphi = samples.nphi_equiang(L, "gl", nphi)
    if reality:
        positive = xp.pad(ftm[..., L - 1 :], ((0, 0), (0, nphi // 2 + 1 - L)))
        # For even nphi, the final rfft bin is Nyquist and is explicitly zero.
        return xp.fft.irfft(positive, n=nphi, axis=-1, norm="forward")
    # In fftshift order, m=0 is at floor(nphi/2). The even Nyquist bin at index
    # zero lies outside |m| <= L-1, so padding leaves it (and other extras) zero.
    left = nphi // 2 - L + 1
    right = nphi - (2 * L - 1) - left
    spectrum = xp.pad(ftm, ((0, 0), (left, right)))
    return xp.fft.ifft(xp.fft.ifftshift(spectrum, axes=-1), axis=-1, norm="forward")
