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


def calcFreqFromFilterRespose_NumPy(M, NR, NC, vmgH, nFilters, qList):
    """
    Vectorized 3-point parabolic interpolation of the frequency
    from the filter-bank response (NumPy version of the MATLAB helper).

    Parameters
    ----------
    M : ndarray, shape (NR, NC), bool or numeric
        ROI mask; used only to set w[~M] = NaN at the end.
    NR, NC : int
        Image size (rows, cols).
    vmgH : ndarray, shape (nFilters, NR*NC)
        Filter-bank magnitudes for all pixels.
    nFilters : int
        Number of filters along the frequency axis.
    qList : ndarray, shape (nFilters,)
        List of spatial frequencies (in 'ff' units).

    Returns
    -------
    w : ndarray, shape (NR, NC), float32
        Interpolated frequency per pixel (same units as qList),
        with NaN outside the mask M.
    """
    M = np.asarray(M, dtype=bool)
    qList = np.asarray(qList, dtype=np.float32).ravel()  # ensure 1D
    dq = qList[1] - qList[0]
    q0 = qList[0]

    nFilters_check, Npix = vmgH.shape
    assert nFilters_check == nFilters, "vmgH first dim must be nFilters"

    # 1) Discrete maximum index for each pixel (across filters)
    k = np.argmax(vmgH, axis=0).astype(np.int64)  # 0..nFilters-1

    # 2) Neighbor indices (clamped)
    km = np.maximum(k - 1, 0)
    kp = np.minimum(k + 1, nFilters - 1)

    idx = np.arange(Npix, dtype=np.int64)

    y0 = vmgH[k, idx]
    ym = vmgH[km, idx]
    yp = vmgH[kp, idx]

    # 3) Parabolic interpolation (index units)
    den = ym - 2.0 * y0 + yp
    tol = 1e-12
    denAbsSmall = np.abs(den) < tol
    den_safe = den.copy()
    den_safe[denAbsSmall] = 1.0  # avoid division by 0

    delta = 0.5 * (ym - yp) / den_safe
    delta[denAbsSmall] = 0.0

    # 4) Map (index + delta) → frequency using linear qList mapping
    kEff = k.astype(np.float32) + delta.astype(np.float32)  # effective index (0-based)
    qMax_vec = q0 + kEff * dq  # same units as qList

    # 5) Reshape and apply mask
    w = qMax_vec.reshape(NR, NC).astype(np.float32)
    w[~M] = np.nan
    return w


def calcSpatialFreqsFilterbank_ParVer_NumPy(
    g,
    M=None,
    *,
    wTh: float = 5.0,
    wmin: float | None = None,
    wmax: float | None = None,
    Dw: float = 1.0,
    freqUnits: str = "rad/px",
    calcMethod: str = "interpFreq",
    filterFlag: bool = True,
):
    """
    Local spatial frequency estimation for fringe patterns (NumPy/OpenCV).

    Python translation of calcSpatialFreqsFilterbank_ParVer.m, using:
      - numpy.fft for FFTs
      - cv2.medianBlur for medfilt2
      - cv2.filter2D for conv2(...,'same')

    Parameters
    ----------
    g : ndarray, shape (NR, NC)
        Input fringe pattern (igram), real-valued. Model: g = a + b*cos(phi).
    M : ndarray or None, shape (NR, NC), bool or numeric
        ROI mask. Non-zero / True means valid. Default: full image valid.
    wTh : float, optional
        Spatial-frequency magnitude threshold (in "ff" units). Default: 5.
    wmin : float or None, optional
        Min spatial frequency (in "ff" units) for scanning. Default: wTh.
    wmax : float or None, optional
        Max spatial frequency (in "ff" units). Default: 0.25*mean(size(g)).
    Dw : float, optional
        Frequency step (in "ff" units) between successive filters. Default: 1.
    freqUnits : {"ff","rad/px"}, optional
        Units for output: normalized fft units ("ff") or radians/pixel ("rad/px").
        Default: "rad/px".
    calcMethod : {"interpFreq","maxFreq"}, optional
        Estimation method:
          - "interpFreq": 3-point parabolic interpolation around maximum.
          - "maxFreq": use discrete frequency of maximum response.
        Default: "interpFreq".
    filterFlag : bool, optional
        If True, applies phasor-based smoothing to phi_x and phi_y (and M_proc).
        If False, returns raw estimates and M_proc = M. Default: True.

    Returns
    -------
    w_phi : ndarray, float32, shape (NR, NC)
        Local spatial frequency magnitude at each pixel.
        Units: "ff" or "rad/px" depending on freqUnits.
    theta_or : ndarray, float32, shape (NR, NC)
        Local fringe orientation in radians, in [0, pi].
    phi_x : ndarray, float32, shape (NR, NC)
        Local phase gradient component in x.
    phi_y : ndarray, float32, shape (NR, NC)
        Local phase gradient component in y.
    M_proc : ndarray, bool, shape (NR, NC)
        Processed mask after optional filtering of the ROI borders.
    """
    # ------------------------------------------------------------------
    # Input & defaults
    # ------------------------------------------------------------------
    g = np.asarray(g, dtype=np.float32)
    if g.ndim != 2:
        raise ValueError("Input g must be a 2D array.")

    NR, NC = g.shape

    if M is None:
        M = np.ones_like(g, dtype=bool)
    else:
        M = np.asarray(M)
        if M.shape != g.shape:
            raise ValueError("Mask M must have the same shape as g.")
        M = M.astype(bool)

    if wmin is None:
        wmin = float(wTh)
    if wmax is None:
        wmax = 0.25 * float((NR + NC) / 2.0)

    freqUnits = str(freqUnits)
    if freqUnits not in ("ff", "rad/px"):
        raise ValueError("freqUnits must be 'ff' or 'rad/px'.")

    calcMethod = str(calcMethod)
    if calcMethod not in ("interpFreq", "maxFreq"):
        raise ValueError("calcMethod must be 'interpFreq' or 'maxFreq'.")

    filterFlag = bool(filterFlag)

    # ------------------------------------------------------------------
    # Spatial freqs: cartesian freq units in "ff"
    # ------------------------------------------------------------------
    # MATLAB:
    #   [u,v] = meshgrid(1:NC, 1:NR); u0=floor(NC/2)+1; u=u-u0; etc.
    # Python (0-based): equivalent centered grid
    u = np.arange(NC, dtype=np.float32) - (NC // 2)
    v = np.arange(NR, dtype=np.float32) - (NR // 2)
    u, v = np.meshgrid(u, v)  # shape (NR, NC)

    q = np.abs(u + 1j * v).astype(np.float32)

    # ------------------------------------------------------------------
    # g's FT and high-pass filter
    # ------------------------------------------------------------------
    G = np.fft.fft2(g).astype(np.complex64)

    H1 = 1.0 - np.exp(-0.5 * (q / np.float32(wTh)) ** 2).astype(np.float32)
    H1_shift = np.fft.ifftshift(H1)
    G *= H1_shift.astype(np.complex64)

    # ------------------------------------------------------------------
    # Filter bank (parallel 3D implementation)
    # ------------------------------------------------------------------
    sw = 10.0 * float(Dw)
    nFilters = int(round((wmax - wmin) / Dw))
    if nFilters <= 0:
        raise ValueError("nFilters <= 0: check wmin, wmax, Dw.")

    qList = np.linspace(wmin, wmax, nFilters, dtype=np.float32)

    # Broadcasted 3D grids
    qList3 = qList.reshape(1, 1, -1)              # 1 x 1 x nFilters
    u3 = np.broadcast_to(u[:, :, None], (NR, NC, nFilters))
    v3 = np.broadcast_to(v[:, :, None], (NR, NC, nFilters))

    # 3D Gabor filters in (u,v) frequency plane
    Hx = np.exp(-0.5 * ((u3 - qList3) / sw) ** 2).astype(np.float32)
    Hy = np.exp(-0.5 * ((v3 - qList3) / sw) ** 2).astype(np.float32)

    # Match fft2 convention: move DC to (0,0) for each slice
    Hx = np.fft.ifftshift(np.fft.ifftshift(Hx, axes=0), axes=1)
    Hy = np.fft.ifftshift(np.fft.ifftshift(Hy, axes=0), axes=1)

    # Apply filter bank: G has shape (NR,NC), broadcast over third dim
    G3 = G[:, :, None]  # NR x NC x 1, broadcast with Hx/Hy
    gHx = np.fft.ifft2(G3 * Hx, axes=(0, 1))  # NR x NC x nFilters
    gHy = np.fft.ifft2(G3 * Hy, axes=(0, 1))  # NR x NC x nFilters

    # Magnitudes (float32)
    mgHx = np.abs(gHx).astype(np.float32)
    mgHy = np.abs(gHy).astype(np.float32)

    # Arrange as [nFilters, NR*NC]
    vmgHx = mgHx.transpose(2, 0, 1).reshape(nFilters, NR * NC)
    vmgHy = mgHy.transpose(2, 0, 1).reshape(nFilters, NR * NC)

    # ------------------------------------------------------------------
    # Spatial frequency estimation: X component
    # ------------------------------------------------------------------
    if calcMethod == "maxFreq":
        # Fast but less precise (depends on Dw)
        pos = np.argmax(vmgHx, axis=0)
        phi_x = qList[pos].reshape(NR, NC).astype(np.float32)
    else:
        # "interpFreq": parabolic interpolation
        phi_x = calcFreqFromFilterRespose_NumPy(M, NR, NC, vmgHx, nFilters, qList)

    # Filter impulsive noise (median 5x5)
    phi_x = cv2.medianBlur(phi_x.astype(np.float32), 5)

    # Optional weighted filtering
    if filterFlag:
        phi_x = phi_x.astype(np.float32)
        phi_x[np.isnan(phi_x)] = 0.0

        phaseFactor = 0.01
        # wxQuality = reshape(std(vmgHx).^2, NR, NC);
        wxQuality = (np.std(vmgHx, axis=0) ** 2).reshape(NR, NC).astype(np.float32)

        zx = wxQuality * np.exp(1j * phaseFactor * phi_x)
        kernel = np.ones((10, 10), dtype=np.float32) / 100.0

        zx_real = cv2.filter2D(np.real(zx).astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        zx_imag = cv2.filter2D(np.imag(zx).astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        zxf = zx_real + 1j * zx_imag

        phi_x = (np.angle(zxf) / phaseFactor).astype(np.float32)

        # Filter mask
        M_float = M.astype(np.float32)
        M_conv = cv2.filter2D(M_float, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        M_proc = M_conv > 0.999
    else:
        M_proc = M.copy()

    # ------------------------------------------------------------------
    # Spatial frequency estimation: Y component
    # ------------------------------------------------------------------
    if calcMethod == "maxFreq":
        pos = np.argmax(vmgHy, axis=0)
        phi_y = (qList[pos] ** 2).reshape(NR, NC).astype(np.float32)
    else:
        phi_y = calcFreqFromFilterRespose_NumPy(M, NR, NC, vmgHy, nFilters, qList)

    phi_y = cv2.medianBlur(phi_y.astype(np.float32), 5)

    if filterFlag:
        phi_y = phi_y.astype(np.float32)
        phi_y[np.isnan(phi_y)] = 0.0

        phaseFactor = 0.01
        wyQuality = (np.std(vmgHy, axis=0) ** 2).reshape(NR, NC).astype(np.float32)

        zy = wyQuality * np.exp(1j * phaseFactor * phi_y)
        kernel = np.ones((10, 10), dtype=np.float32) / 100.0

        zy_real = cv2.filter2D(np.real(zy).astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        zy_imag = cv2.filter2D(np.imag(zy).astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        zyf = zy_real + 1j * zy_imag

        phi_y = (np.angle(zyf) / phaseFactor).astype(np.float32)

    # ------------------------------------------------------------------
    # Units conversion and orientation
    # ------------------------------------------------------------------
    if freqUnits == "rad/px":
        Cx = 2.0 * np.pi / float(NC)
        Cy = 2.0 * np.pi / float(NR)
        phi_x = phi_x * Cx
        phi_y = phi_y * Cy
        w_phi = np.abs(phi_x + 1j * phi_y).astype(np.float32)
    else:  # "ff"
        w_phi = np.abs(phi_x + 1j * phi_y).astype(np.float32)

    theta_or = np.arctan2(-phi_y, phi_x).astype(np.float32)

    return (
        w_phi.astype(np.float32),
        theta_or.astype(np.float32),
        phi_x.astype(np.float32),
        phi_y.astype(np.float32),
        M_proc.astype(bool),
    )
