import numpy as np
import cv2


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



def phaseGradient(
    z,
    M,
    NS: int = 5,
    Nmed: int = 2,
    LPCycles: int = 2,
):
    """
    Phase-gradient estimation from complex phasor data.

    Python/OpenCV translation of the MATLAB function phaseGradient.m

        [phi_x, phi_y, M_proc] = phaseGradient(z, M, NS, Nmed, LPCycles)

    Parameters
    ----------
    z : array_like, complex
        Complex phasor array z = b * exp(1j * phi).
    M : array_like, bool or numeric
        ROI mask with valid points. Same shape as z.
    NS : int, optional
        Half-size of the low-pass box filter window.
        Filter size is (2*NS + 1) x (2*NS + 1). Default: 5.
    Nmed : int, optional
        Median filter window size for phi_x and phi_y.
        Square window [Nmed x Nmed] (0 disables). Default: 2.
        (In OpenCV we use a nearest odd kernel size.)
    LPCycles : int, optional
        Number of times the low-pass filter is applied
        to the derivatives and mask. Default: 2.

    Returns
    -------
    phi_x : ndarray, float32
        Phase gradient along x in px^-1.
    phi_y : ndarray, float32
        Phase gradient along y in px^-1.
    M_proc : ndarray, float32
        ROI mask with valid centered differences (1.0 = valid, 0.0 = invalid).

    Note
    -------
    Centered differences are used, so the practical limit for the
    phase variation is pi/2 rad/px instead of pi rad/px.
    
    """
    z = np.asarray(z)
    M = np.asarray(M)

    if z.shape != M.shape:
        raise ValueError("z and M must have the same shape.")

    # Force mask as logical
    if M.dtype != bool:
        M = (M != 0)

    NR, NC = z.shape

    # ------------------------
    # x- and y-indices for centered first difference (0-based)
    # ------------------------
    # MATLAB:
    #   A = [2:NC, NC];  B = [1, 1:NC-1];
    #   C = [2:NR, NR];  D = [1, 1:NR-1];
    #
    # Python (0-based):
    #   A = [1..NC-1, NC-1]; B = [0, 0..NC-2]
    #   C = [1..NR-1, NR-1]; D = [0, 0..NR-2]
    A = np.concatenate([np.arange(1, NC, dtype=np.int64), np.array([NC - 1])])
    B = np.concatenate([np.array([0]), np.arange(0, NC - 1, dtype=np.int64)])
    C = np.concatenate([np.arange(1, NR, dtype=np.int64), np.array([NR - 1])])
    D = np.concatenate([np.array([0]), np.arange(0, NR - 1, dtype=np.int64)])

    # ------------------------
    # Set borders to zero in the ROI mask
    # ------------------------
    # M(:,1)=false; M(1,:)=false; M(NR,:)=false; M(:,NC)=false; in MATLAB
    M[:, 0] = False
    M[0, :] = False
    M[NR - 1, :] = False
    M[:, NC - 1] = False

    # ------------------------
    # Centered phase differences
    # ------------------------

    # dx: centered difference along x
    zd = z[:, A] / z[:, B]
    zd[np.isnan(zd)] = 0
    phi_x = 0.5 * np.angle(zd)

    # dy: centered difference along y
    zd = z[C, :] / z[D, :]
    zd[np.isnan(zd)] = 0
    phi_y = 0.5 * np.angle(zd)

    phi_x = phi_x.astype(np.float32)
    phi_y = phi_y.astype(np.float32)

    # ------------------------
    # Optional median filtering (OpenCV)
    # ------------------------
    if Nmed > 0:
        # OpenCV's medianBlur needs an odd kernel size >= 3.
        # Approximate [Nmed x Nmed] with the nearest odd integer.
        ksize = Nmed if (Nmed % 2 == 1) else (Nmed + 1)
        if ksize < 3:
            ksize = 3

        # cv2.medianBlur expects 2D float32 or uint8
        phi_x = cv2.medianBlur(phi_x, ksize)
        phi_y = cv2.medianBlur(phi_y, ksize)

    # ------------------------
    # Valid-difference mask
    # ------------------------
    # M_proc = M(:,A) & M(:,B) & M & M(C,:) & M(D,:);
    M_proc = (
        M[:, A] & M[:, B] & M & M[C, :] & M[D, :]
    )

    # ------------------------
    # Low-pass smoothing (box filter)
    # ------------------------
    # hLP = ones(2*NS+1)/(2*NS+1)^2
    ksize = 2 * int(NS) + 1
    if ksize < 1:
        ksize = 1
    hLP = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)

    # We'll use cv2.filter2D with BORDER_CONSTANT (zero padding) to mimic conv2 'same'
    for _ in range(int(LPCycles)):
        phi_x = cv2.filter2D(phi_x, ddepth=-1, kernel=hLP, borderType=cv2.BORDER_CONSTANT)
        phi_y = cv2.filter2D(phi_y, ddepth=-1, kernel=hLP, borderType=cv2.BORDER_CONSTANT)

        M_proc_f = M_proc.astype(np.float32)
        M_proc_f = cv2.filter2D(
            M_proc_f, ddepth=-1, kernel=hLP, borderType=cv2.BORDER_CONSTANT
        )
        M_proc = M_proc_f > 0.999

    # ------------------------
    # Apply ROI to gradients
    # ------------------------
    phi_x = np.where(M_proc, phi_x, 0.0).astype(np.float32)
    phi_y = np.where(M_proc, phi_y, 0.0).astype(np.float32)

    return phi_x, phi_y, M_proc.astype(np.float32)


def calcSpatialFreqsHilbert2D_NumPy(
    g,
    M=None,
    *,
    wTh: float = 5.0,
    filter_orientation: float = 0.0,
    phasor_filter_size: int = 5,
):
    """
    Local spatial frequency and fringe orientation (Hilbert-based).

    Python/OpenCV translation of calcSpatialFreqsHilbert2D.m

      [w_phi, theta_or, phi_x, phi_y, M_proc] = calcSpatialFreqsHilbert2D(g, M, opts)

    The input interferogram g is assumed real.

    Parameters
    ----------
    g : array_like, shape (NR, NC)
        Input interferogram, real-valued 2D numeric array.
    M : array_like or None, optional
        Region of interest mask. Default: ones(size(g)).
    wTh : float, optional
        Spatial frequency cutoff for Hilbert demodulation.
        Frequencies below wTh (in frequency units) are attenuated.
        Default: 5.
    filter_orientation : float, optional
        Orientation of the steerable 2D Hilbert filter [rad].
          0     → Hilbert filter sensitive to horizontal fringes
          pi/2  → sensitive to vertical fringes
        Default: 0.
    phasor_filter_size : int, optional
        Half-window size used in phaseGradient().
        Neighborhood = (2 * phasor_filter_size + 1).
        Default: 5.

    Returns
    -------
    w_phi : ndarray, float32
        Spatial frequency magnitude |∇φ| in rad/px.
    theta_or : ndarray, float32
        Fringe orientation angle θ = atan2(-phi_y, phi_x) [rad].
    phi_x : ndarray, float32
        Phase gradient component ∂φ/∂x in rad/px.
    phi_y : ndarray, float32
        Phase gradient component ∂φ/∂y in rad/px.
    M_proc : ndarray, float32
        Processed mask from phaseGradient (validity map).
    """
    # Ensure real float32 interferogram
    g = np.asarray(g, dtype=np.float32)
    if g.ndim != 2:
        raise ValueError("Input g must be a 2D array.")

    NR, NC = g.shape

    # ROI mask
    if M is None:
        M = np.ones((NR, NC), dtype=np.float32)
    else:
        M = np.asarray(M)
        if M.shape != g.shape:
            raise ValueError("Mask M must have the same shape as g.")
        M = M.astype(np.float32)

    # ---------------------------------------------------------------------
    # 1) Hilbert Demodulation: z ≈ m * exp(1j φ)
    # ---------------------------------------------------------------------
    z = DemodHilbert2D_NumPy(
        g,
        M,
        wTh=wTh,
        filter_orientation=filter_orientation,
    )

    # ---------------------------------------------------------------------
    # 2) Phase-gradient estimation (via phasor filtering)
    # ---------------------------------------------------------------------
    Nmed = 2      # median-filter size (fixed as in MATLAB)
    LPCycles = 2  # low-pass cycles (fixed as in MATLAB)

    phi_x, phi_y, M_proc = phaseGradient(
        z,
        M,
        NS=phasor_filter_size,
        Nmed=Nmed,
        LPCycles=LPCycles,
    )

    # ---------------------------------------------------------------------
    # 3) Spatial frequency magnitude and fringe orientation
    # ---------------------------------------------------------------------
    w_phi = np.abs(phi_x + 1j * phi_y).astype(np.float32)
    theta_or = np.arctan2(-phi_y, phi_x).astype(np.float32)

    return w_phi, theta_or, phi_x.astype(np.float32), phi_y.astype(np.float32), M_proc.astype(np.float32)
