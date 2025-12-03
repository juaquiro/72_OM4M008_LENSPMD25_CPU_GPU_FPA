import numpy as np


def DemodHilbert2D_NumPy(g, M=None, *, wTh=3.0, filter_orientation=0.0):
    """
    Demodulate 2D interferogram using a steerable Hilbert filter (NumPy version).

    Parameters
    ----------
    g : array_like, shape (NR, NC)
        Input interferogram (real 2D array).
    M : array_like or None, optional
        ROI mask, same shape as g. If None, uses an all-ones mask.
        Values != 1 will mask out the corresponding pixels.
    wTh : float, optional
        Spatial frequency threshold (Gaussian high-pass filter).
        Removes frequencies below wTh in the Fourier domain.
        Default is 3.
    filter_orientation : float, optional
        Orientation of the 2D steerable Hilbert filter [rad].
          0     → horizontal-sensitive
          pi/2  → vertical-sensitive
        Default is 0.

    Returns
    -------
    z : ndarray, complex64, shape (NR, NC)
        Complex analytic signal: z ≈ m * exp(1j * phi)

    Notes
    -----
    Hilbert filter (half-plane):
        H = (u * cos(theta) + v * sin(theta)) > 0

    High-pass filter (Gaussian):
        H1 = 1 - exp(-0.5 * (|q| / wTh)**2)

    This is a NumPy translation of the MATLAB function DemodHilbert2D.m,
    using float32 for most real-valued arrays for speed. Input interferogram
    g is assumed real.
    """

    # ------------------------------------------------------------------
    # Input checks and dtype normalization
    # ------------------------------------------------------------------
    g = np.asarray(g, dtype=np.float32)
    if g.ndim != 2:
        raise ValueError("Input g must be a 2D array.")

    nr, nc = g.shape

    if M is None:
        M = np.ones_like(g, dtype=np.float32)
    else:
        M = np.asarray(M, dtype=np.float32)
        if M.shape != g.shape:
            raise ValueError("Mask M must have the same shape as g.")

    # Keyword arguments as float32 for consistency/speed
    wTh = np.float32(wTh)
    filter_orientation = np.float32(filter_orientation)

    # ------------------------------------------------------------------
    # Apply ROI mask (if not all ones)
    # ------------------------------------------------------------------
    # In MATLAB: if ~all(M(:) == 1), g = g .* M;
    # Here we can just always multiply; if M is all ones it's a no-op.
    g = g * M

    # ------------------------------------------------------------------
    # Spatial frequency grid (centered FFT coordinates)
    # ------------------------------------------------------------------
    # MATLAB:
    # [NR, NC] = size(g);
    # [u, v]   = meshgrid(1:NC, 1:NR);
    # u = u - (floor(NC/2) + 1);
    # v = v - (floor(NR/2) + 1);
    #
    # Python uses 0-based indexing, so the equivalent centered grid is:
    # u = 0..NC-1 - floor(NC/2)
    # v = 0..NR-1 - floor(NR/2)
    u = np.arange(nc, dtype=np.float32) - (nc // 2)
    v = np.arange(nr, dtype=np.float32) - (nr // 2)
    u, v = np.meshgrid(u, v)  # shape (nr, nc) each

    # Frequency magnitude q = |u + 1j v|
    # Use hypot to keep everything real and float32
    q = np.hypot(u, v).astype(np.float32)

    # ------------------------------------------------------------------
    # Gaussian high-pass filter
    # ------------------------------------------------------------------
    # H1 = 1 - exp(-0.5 * (q / wTh).^2);
    H1 = 1.0 - np.exp(-0.5 * (q / wTh) ** 2, dtype=np.float32)

    # ------------------------------------------------------------------
    # Steerable Hilbert filter (half-plane)
    # ------------------------------------------------------------------
    # H = (u * cos(filter_orientation) + v * sin(filter_orientation)) > 0;
    c = np.cos(filter_orientation, dtype=np.float32)
    s = np.sin(filter_orientation, dtype=np.float32)
    H = (u * c + v * s) > 0  # boolean
    H = H.astype(np.float32)

    # Total filter in Fourier domain, with ifftshift as in MATLAB:
    # G = fft2(g); G = G .* ifftshift(H .* H1);
    H_total = H * H1
    H_total = np.fft.ifftshift(H_total)

    # ------------------------------------------------------------------
    # Demodulation
    # ------------------------------------------------------------------
    # fft2 of real g; NumPy will use complex128 internally.
    G = np.fft.fft2(g)

    G *= H_total
    z = 2.0 * np.fft.ifft2(G)

    # Use complex64 to match the "float32 for speed" spirit
    return z.astype(np.complex64)
