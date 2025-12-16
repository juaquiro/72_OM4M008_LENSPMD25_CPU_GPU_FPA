# FPA_TensorFlow.py
#
# GPU-optimized TensorFlow versions of:
#   - DemodHilbert2D_NumPy  → DemodHilbert2D_TF
#   - calcSpatialFreqsHilbert2D_NumPy → calcSpatialFreqsHilbert2D_TF
#
# Both functions accept g and M as either NumPy arrays or tf.Tensors.
# Typical use case: g, M are NumPy arrays; everything runs on GPU
# via TensorFlow with float32 / complex64 precision.

import tensorflow as tf
import numpy as np


# ---------------------------------------------------------------------
# Utility: 2D box filter (conv2 'same') using tf.nn.conv2d
# ---------------------------------------------------------------------
def _box_filter_2d(x: tf.Tensor, ksize: int) -> tf.Tensor:
    """
    2D box filter with SAME padding.

    Parameters
    ----------
    x : tf.Tensor, shape [H, W], float32
    ksize : int
        Kernel size (ksize x ksize).

    Returns
    -------
    y : tf.Tensor, shape [H, W], float32
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    if ksize <= 1:
        return x

    ksize = int(ksize)
    if ksize < 1:
        ksize = 1

    kernel = tf.ones((ksize, ksize, 1, 1), dtype=tf.float32)
    kernel = kernel / tf.cast(ksize * ksize, tf.float32)

    x4 = tf.expand_dims(tf.expand_dims(x, axis=0), axis=-1)  # [1,H,W,1]
    y4 = tf.nn.conv2d(x4, kernel, strides=[1, 1, 1, 1], padding="SAME")
    y = tf.squeeze(y4, [0, 3])
    return y


# ---------------------------------------------------------------------
# Utility: 2D median filter using tf.image.extract_patches
# ---------------------------------------------------------------------
def _median_filter_2d(x: tf.Tensor, ksize: int) -> tf.Tensor:
    """
    2D median filter (approximate medfilt2 with [ksize ksize]).

    Parameters
    ----------
    x : tf.Tensor, shape [H, W], float32
    ksize : int
        Window size; if even, it is promoted to the next odd.

    Returns
    -------
    y : tf.Tensor, shape [H, W], float32
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    if ksize <= 1:
        return x

    ksize = int(ksize)
    if ksize % 2 == 0:
        ksize += 1
    if ksize < 3:
        ksize = 3

    x4 = tf.expand_dims(tf.expand_dims(x, axis=0), axis=-1)  # [1,H,W,1]

    patches = tf.image.extract_patches(
        images=x4,
        sizes=[1, ksize, ksize, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding="SAME",
    )  # [1,H,W,ksize*ksize]

    patches_sorted = tf.sort(patches, axis=-1)
    mid = (ksize * ksize) // 2
    med = patches_sorted[..., mid]  # [1,H,W]

    y = tf.squeeze(med, axis=0)  # [H,W]
    return y


# ---------------------------------------------------------------------
# DemodHilbert2D_TF  (GPU-optimized version of DemodHilbert2D_NumPy)
# ---------------------------------------------------------------------
def DemodHilbert2D_TF(
    g,
    M=None,
    *,
    wTh: float = 3.0,
    filter_orientation: float = 0.0,
):
    """
    Demodulate 2D interferogram using a steerable Hilbert filter (TensorFlow/GPU).

    TensorFlow adaptation of DemodHilbert2D_NumPy.:contentReference[oaicite:1]{index=1}

    Parameters
    ----------
    g : np.ndarray or tf.Tensor, shape (NR, NC)
        Input interferogram (real 2D array).
    M : np.ndarray, tf.Tensor, or None, optional
        ROI mask, same shape as g. If None, uses an all-ones mask.
    wTh : float, optional (keyword)
        Spatial frequency threshold (Gaussian high-pass filter).
        Removes frequencies below wTh in the Fourier domain.
        Default: 3.
    filter_orientation : float, optional (keyword)
        Orientation of the 2D steerable Hilbert filter [rad].
          0     → horizontal-sensitive
          pi/2  → vertical-sensitive
        Default: 0.

    Returns
    -------
    z : tf.Tensor, complex64, shape (NR, NC)
        Complex analytic signal: z ≈ m * exp(1j * phi)
    """
    # ------------------------------------------------------------------
    # Input to TF tensors (single cast per tensor, always FP32)
    # ------------------------------------------------------------------
    g_tf = tf.convert_to_tensor(g, dtype=tf.float32)
    if g_tf.shape.rank != 2:
        raise ValueError("Input g must be a 2D array/tensor.")

    if M is None:
        M_tf = tf.ones_like(g_tf, dtype=tf.float32)
    else:
        M_tf = tf.convert_to_tensor(M)
        if M_tf.dtype.is_bool:
            M_tf = tf.cast(M_tf, tf.float32)
        else:
            M_tf = tf.cast(M_tf, tf.float32)
        tf.debugging.assert_equal(
            tf.shape(g_tf),
            tf.shape(M_tf),
            message="Mask M must have the same shape as g.",
        )

    # ------------------------------------------------------------------
    # Apply ROI mask (one multiply, no type changes)
    # ------------------------------------------------------------------
    g_tf = g_tf * M_tf  # real float32

    # ------------------------------------------------------------------
    # Spatial frequency grid (centered FFT coordinates)
    # ------------------------------------------------------------------
    shape = tf.shape(g_tf)
    NR = shape[0]
    NC = shape[1]

    u = tf.range(NC, dtype=tf.float32) - tf.cast(NC // 2, tf.float32)
    v = tf.range(NR, dtype=tf.float32) - tf.cast(NR // 2, tf.float32)

    U = tf.tile(u[tf.newaxis, :], [NR, 1])  # [NR,NC]
    V = tf.tile(v[:, tf.newaxis], [1, NC])  # [NR,NC]

    q = tf.sqrt(tf.square(U) + tf.square(V))  # float32

    wTh_tf = tf.convert_to_tensor(wTh, dtype=tf.float32)
    filter_orientation_tf = tf.convert_to_tensor(filter_orientation, dtype=tf.float32)

    # Gaussian high-pass filter
    H1 = 1.0 - tf.exp(-0.5 * tf.square(q / wTh_tf))

    # Steerable Hilbert filter (half-plane)
    c = tf.cos(filter_orientation_tf)
    s = tf.sin(filter_orientation_tf)
    H = tf.cast(U * c + V * s > 0.0, tf.float32)

    H_total = H * H1
    H_total = tf.signal.ifftshift(H_total)

    # ------------------------------------------------------------------
    # Demodulation via 2D FFT (GPU if available)
    # ------------------------------------------------------------------
    G = tf.signal.fft2d(tf.cast(g_tf, tf.complex64))
    G = G * tf.cast(H_total, tf.complex64)
    z = 2.0 * tf.signal.ifft2d(G)  # complex64

    return z


# ---------------------------------------------------------------------
# phaseGradient_TF  (TF analogue of NumPy 'phaseGradient')
# ---------------------------------------------------------------------
def phaseGradient_TF(
    z,
    M,
    NS: int = 5,
    Nmed: int = 2,
    LPCycles: int = 2,
):
    """
    Phase-gradient estimation from complex phasor data (TensorFlow/GPU).

    TensorFlow analogue of the NumPy/OpenCV phaseGradient function.

        [phi_x, phi_y, M_proc] = phaseGradient_TF(z, M, NS, Nmed, LPCycles)

    Parameters
    ----------
    z : np.ndarray or tf.Tensor, complex, shape (NR, NC)
        Complex phasor array z = b * exp(1j * phi).
    M : np.ndarray or tf.Tensor, bool or numeric, shape (NR, NC)
        ROI mask with valid points.
    NS : int, optional
        Half-size of the low-pass box filter window.
        Filter size is (2*NS + 1) x (2*NS + 1). Default: 5.
    Nmed : int, optional
        Median filter window size for phi_x and phi_y.
        Square window [Nmed x Nmed] (0 disables). Default: 2.
    LPCycles : int, optional
        Number of times the low-pass filter is applied
        to the derivatives and mask. Default: 2.

    Returns
    -------
    phi_x : tf.Tensor, float32, shape (NR, NC)
        Phase gradient along x in px^-1.
    phi_y : tf.Tensor, float32, shape (NR, NC)
        Phase gradient along y in px^-1.
    M_proc : tf.Tensor, float32, shape (NR, NC)
        ROI mask with valid centered differences (1.0 = valid, 0.0 = invalid).

    Note
    ----
    Centered differences are used, so the practical limit for the
    phase variation is pi/2 rad/px instead of pi rad/px.
    """
    # Ensure tensors
    z_tf = tf.convert_to_tensor(z, dtype=tf.complex64)
    M_tf = tf.convert_to_tensor(M)
    if not M_tf.dtype.is_bool:
        M_tf = tf.not_equal(M_tf, 0)

    shape = tf.shape(z_tf)
    NR = shape[0]
    NC = shape[1]

    # Set borders to zero in the ROI mask
    rows = tf.range(NR)[:, tf.newaxis]  # [NR,1]
    cols = tf.range(NC)[tf.newaxis, :]  # [1,NC]
    border = tf.logical_or(
        tf.logical_or(tf.equal(rows, 0), tf.equal(rows, NR - 1)),
        tf.logical_or(tf.equal(cols, 0), tf.equal(cols, NC - 1)),
    )
    M_tf = tf.logical_and(M_tf, tf.logical_not(border))

    # ------------------------
    # Centered phase differences
    # ------------------------

    # dx: centered difference along x
    z_right = tf.concat([z_tf[:, 1:], z_tf[:, -1:]], axis=1)
    z_left  = tf.concat([z_tf[:, :1], z_tf[:, :-1]], axis=1)
    zd_x = z_right / z_left  # complex64

    # dy: centered difference along y
    z_down = tf.concat([z_tf[1:, :], z_tf[-1:, :]], axis=0)
    z_up   = tf.concat([z_tf[:1, :], z_tf[:-1, :]], axis=0)
    zd_y = z_down / z_up  # complex64

    # Phase of the complex ratios (this is real float32)
    phi_x = 0.5 * tf.math.angle(zd_x)
    phi_y = 0.5 * tf.math.angle(zd_y)

    # Clean NaNs *after* taking the angle
    phi_x = tf.where(tf.math.is_nan(phi_x), tf.zeros_like(phi_x), phi_x)
    phi_y = tf.where(tf.math.is_nan(phi_y), tf.zeros_like(phi_y), phi_y)

    phi_x = tf.cast(phi_x, tf.float32)
    phi_y = tf.cast(phi_y, tf.float32)

    # ------------------------
    # Optional median filtering
    # ------------------------
    if Nmed > 0:
        phi_x = _median_filter_2d(phi_x, Nmed)
        phi_y = _median_filter_2d(phi_y, Nmed)

    # ------------------------
    # Valid-difference mask
    # ------------------------
    M_right = tf.concat([M_tf[:, 1:], M_tf[:, -1:]], axis=1)
    M_left  = tf.concat([M_tf[:, :1], M_tf[:, :-1]], axis=1)
    M_down  = tf.concat([M_tf[1:, :], M_tf[-1:, :]], axis=0)
    M_up    = tf.concat([M_tf[:1, :], M_tf[:-1, :]], axis=0)

    M_proc_bool = M_tf & M_right & M_left & M_up & M_down

    # ------------------------
    # Low-pass smoothing (box filter)
    # ------------------------
    ksize = 2 * int(NS) + 1
    if ksize < 1:
        ksize = 1

    M_proc_f = tf.cast(M_proc_bool, tf.float32)
    for _ in range(int(LPCycles)):
        phi_x = _box_filter_2d(phi_x, ksize)
        phi_y = _box_filter_2d(phi_y, ksize)
        M_proc_f = _box_filter_2d(M_proc_f, ksize)

    # Threshold back to boolean
    M_proc_bool = M_proc_f > 0.999

    # Apply ROI to gradients
    phi_x = tf.where(M_proc_bool, phi_x, tf.zeros_like(phi_x))
    phi_y = tf.where(M_proc_bool, phi_y, tf.zeros_like(phi_y))

    return phi_x, phi_y, tf.cast(M_proc_bool, tf.float32)


# ---------------------------------------------------------------------
# calcSpatialFreqsHilbert2D_TF  (GPU-optimized version of _NumPy)
# ---------------------------------------------------------------------
def calcSpatialFreqsHilbert2D_TF(
    g,
    M=None,
    *,
    wTh: float = 5.0,
    filter_orientation: float = 0.0,
    phasor_filter_size: int = 5,
):
    """
    Local spatial frequency and fringe orientation (Hilbert-based, TensorFlow/GPU).

    Identical API to the NumPy version but GPU-accelerated.
    INPUT:  g, M can be NumPy arrays or tf.Tensors.
    OUTPUT: ALWAYS NumPy arrays (float32).
    """

    # g: single conversion to float32 Tensor
    g_tf = tf.convert_to_tensor(g, dtype=tf.float32)

    shape = tf.shape(g_tf)
    NR = shape[0]
    NC = shape[1]

    # ROI mask: preserve boolean info
    if M is None:
        M_bool = tf.ones((NR, NC), dtype=tf.bool)
    else:
        M_tf = tf.convert_to_tensor(M)
        M_bool = M_tf if M_tf.dtype.is_bool else tf.not_equal(M_tf, 0)

        tf.debugging.assert_equal(
            tf.shape(g_tf),
            tf.shape(M_bool),
            message="Mask M must have the same shape as g.",
        )

    # Float mask for demodulation
    M_float = tf.cast(M_bool, tf.float32)

    # ------------------------------------------------------------------
    # 1) Hilbert Demodulation on GPU
    # ------------------------------------------------------------------
    z = DemodHilbert2D_TF(
        g_tf,
        M_float,
        wTh=wTh,
        filter_orientation=filter_orientation,
    )  # complex64

    # ------------------------------------------------------------------
    # 2) Phase Gradients (GPU)
    # ------------------------------------------------------------------
    Nmed = 2
    LPCycles = 2

    phi_x, phi_y, M_proc = phaseGradient_TF(
        z,
        M_bool,
        NS=phasor_filter_size,
        Nmed=Nmed,
        LPCycles=LPCycles,
    )

    # ------------------------------------------------------------------
    # 3) Spatial frequency magnitude & orientation
    # ------------------------------------------------------------------
    grad_complex = tf.complex(phi_x, phi_y)
    w_phi      = tf.abs(grad_complex)
    theta_or   = tf.math.atan2(-phi_y, phi_x)

    # ------------------------------------------------------------------
    # Convert ALL outputs to NumPy arrays (float32)
    # ------------------------------------------------------------------
    return (
        w_phi.numpy().astype(np.float32),
        theta_or.numpy().astype(np.float32),
        phi_x.numpy().astype(np.float32),
        phi_y.numpy().astype(np.float32),
        M_proc.numpy().astype(np.float32),
    )


# ---------------------------------------------------------------------
# TF version of calcFreqFromFilterRespose_NumPy
# ---------------------------------------------------------------------
def calcFreqFromFilterRespose_TF(M, NR, NC, vmgH, nFilters, qList):
    """
    Vectorized 3-point parabolic interpolation of the frequency
    from the filter-bank response (TensorFlow/GPU version).

    Parameters
    ----------
    M : ndarray or tf.Tensor, shape (NR, NC), bool or numeric
        ROI mask; used only to set w[~M] = NaN at the end.
    NR, NC : int
        Image size (rows, cols).
    vmgH : ndarray or tf.Tensor, shape (nFilters, NR*NC)
        Filter-bank magnitudes for all pixels.
    nFilters : int
        Number of filters along the frequency axis.
    qList : ndarray or tf.Tensor, shape (nFilters,)
        List of spatial frequencies (in 'ff' units).

    Returns
    -------
    w_tf : tf.Tensor, shape (NR, NC), float32
        Interpolated frequency per pixel (same units as qList),
        with NaN outside the mask M.
    """
    # Mask to bool
    M_tf = tf.convert_to_tensor(M)
    if not M_tf.dtype.is_bool:
        M_tf = tf.not_equal(M_tf, 0)

    qList_tf = tf.reshape(tf.convert_to_tensor(qList, dtype=tf.float32), [-1])
    dq = qList_tf[1] - qList_tf[0]
    q0 = qList_tf[0]

    vmgH_tf = tf.convert_to_tensor(vmgH, dtype=tf.float32)
    shape_v = tf.shape(vmgH_tf)
    nFilters_check = shape_v[0]
    Npix = shape_v[1]
    tf.debugging.assert_equal(
        nFilters_check, nFilters, message="vmgH first dim must be nFilters"
    )

    # 1) Discrete maximum index for each pixel (across filters)
    k = tf.argmax(vmgH_tf, axis=0, output_type=tf.int32)  # [Npix]

    # 2) Neighbor indices (clamped)
    km = tf.maximum(k - 1, 0)
    kp = tf.minimum(k + 1, nFilters - 1)

    idx = tf.range(Npix, dtype=tf.int32)

    # Gather helper
    def _gather_at(indices_filter, indices_pix):
        return tf.gather_nd(
            vmgH_tf,
            tf.stack([indices_filter, indices_pix], axis=1),
        )

    y0 = _gather_at(k, idx)
    ym = _gather_at(km, idx)
    yp = _gather_at(kp, idx)

    # 3) Parabolic interpolation (index units)
    den = ym - 2.0 * y0 + yp
    tol = tf.constant(1e-12, dtype=tf.float32)
    denAbsSmall = tf.abs(den) < tol
    den_safe = tf.where(denAbsSmall, tf.ones_like(den), den)

    delta = 0.5 * (ym - yp) / den_safe
    delta = tf.where(denAbsSmall, tf.zeros_like(delta), delta)

    # 4) Map (index + delta) → frequency using linear qList mapping
    kEff = tf.cast(k, tf.float32) + tf.cast(delta, tf.float32)  # [Npix]
    qMax_vec = q0 + kEff * dq

    # 5) Reshape and apply mask
    w_tf = tf.reshape(qMax_vec, (NR, NC))
    # set outside mask to NaN
    nan_val = tf.constant(np.float32(np.nan))
    w_tf = tf.where(M_tf, w_tf, tf.fill([NR, NC], nan_val))

    return w_tf  # float32 tensor


# ---------------------------------------------------------------------
# TF version of calcSpatialFreqsFilterbank_ParVer_NumPy
# ---------------------------------------------------------------------
# OLD 
def OLD_calcSpatialFreqsFilterbank_ParVer_TF(
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
    Local spatial frequency estimation for fringe patterns (TensorFlow/GPU).

    TF/GPU translation of calcSpatialFreqsFilterbank_ParVer_NumPy.
    Heavy operations (FFT, filter bank, smoothing) run in TensorFlow,
    final results are returned as NumPy arrays (float32 / bool).

    Parameters
    ----------
    g : np.ndarray or tf.Tensor, shape (NR, NC)
        Input fringe pattern (igram), real-valued. Model: g = a + b*cos(phi).
    M : np.ndarray or tf.Tensor or None, shape (NR, NC), bool or numeric
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
    w_phi : np.ndarray, float32, shape (NR, NC)
        Local spatial frequency magnitude at each pixel.
    theta_or : np.ndarray, float32, shape (NR, NC)
        Local fringe orientation in radians, in [0, pi].
    phi_x : np.ndarray, float32, shape (NR, NC)
        Local phase gradient component in x.
    phi_y : np.ndarray, float32, shape (NR, NC)
        Local phase gradient component in y.
    M_proc : np.ndarray, bool, shape (NR, NC)
        Processed mask after optional filtering of the ROI borders.
    """
    # ------------------------------------------------------------------
    # Input & defaults
    # ------------------------------------------------------------------
    g_tf = tf.convert_to_tensor(g, dtype=tf.float32)
    if g_tf.shape.rank != 2:
        raise ValueError("Input g must be a 2D array/tensor.")

    shape = tf.shape(g_tf)
    NR = int(shape[0].numpy())
    NC = int(shape[1].numpy())

    if M is None:
        M_bool = tf.ones((NR, NC), dtype=tf.bool)
    else:
        M_tf = tf.convert_to_tensor(M)
        if M_tf.dtype.is_bool:
            M_bool = M_tf
        else:
            M_bool = tf.not_equal(M_tf, 0)

        tf.debugging.assert_equal(
            tf.shape(g_tf),
            tf.shape(M_bool),
            message="Mask M must have the same shape as g.",
        )

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
    u = tf.range(NC, dtype=tf.float32) - tf.cast(NC // 2, tf.float32)
    v = tf.range(NR, dtype=tf.float32) - tf.cast(NR // 2, tf.float32)
    U, V = tf.meshgrid(u, v)  # shape [NR, NC]

    q = tf.sqrt(tf.square(U) + tf.square(V))  # float32

    # ------------------------------------------------------------------
    # g's FT and high-pass filter (2D FFT)
    # ------------------------------------------------------------------
    G = tf.signal.fft2d(tf.cast(g_tf, tf.complex64))  # last 2 dims = (NR,NC)

    H1 = 1.0 - tf.exp(-0.5 * tf.square(q / tf.constant(wTh, tf.float32)))
    H1_shift = tf.signal.ifftshift(H1)

    G = G * tf.cast(H1_shift, tf.complex64)

    # ------------------------------------------------------------------
    # Filter bank (parallel 3D implementation on GPU)
    # ------------------------------------------------------------------
    sw = 10.0 * float(Dw)
    nFilters = int(round((wmax - wmin) / Dw))
    if nFilters <= 0:
        raise ValueError("nFilters <= 0: check wmin, wmax, Dw.")

    qList_tf = tf.linspace(
        tf.constant(wmin, tf.float32),
        tf.constant(wmax, tf.float32),
        nFilters,
    )  # [nFilters]

    # Hx / Hy shapes: [nFilters, NR, NC] so that fft2d works on last two dims
    qList3 = tf.reshape(qList_tf, (nFilters, 1, 1))  # [nFilters,1,1]
    U3 = tf.reshape(U, (1, NR, NC))
    V3 = tf.reshape(V, (1, NR, NC))

    Hx = tf.exp(-0.5 * tf.square((U3 - qList3) / sw))
    Hy = tf.exp(-0.5 * tf.square((V3 - qList3) / sw))

    # Match fft2d convention: move DC to (0,0) for each filter slice
    Hx = tf.signal.ifftshift(Hx, axes=(1, 2))
    Hy = tf.signal.ifftshift(Hy, axes=(1, 2))

    # Broadcast G to [nFilters, NR, NC] and apply filters
    G2 = tf.expand_dims(G, axis=0)  # [1,NR,NC]
    G2 = tf.broadcast_to(G2, (nFilters, NR, NC))  # [nFilters,NR,NC]

    gHx = tf.signal.ifft2d(G2 * tf.cast(Hx, tf.complex64))  # [nFilters,NR,NC]
    gHy = tf.signal.ifft2d(G2 * tf.cast(Hy, tf.complex64))  # [nFilters,NR,NC]

    # Magnitudes
    mgHx = tf.abs(gHx)  # float32, [nFilters,NR,NC]
    mgHy = tf.abs(gHy)  # float32, [nFilters,NR,NC]

    # Flatten to [nFilters, NR*NC]
    vmgHx = tf.reshape(mgHx, (nFilters, NR * NC))
    vmgHy = tf.reshape(mgHy, (nFilters, NR * NC))

    # ------------------------------------------------------------------
    # Spatial frequency estimation: X component
    # ------------------------------------------------------------------
    if calcMethod == "maxFreq":
        pos_x = tf.argmax(vmgHx, axis=0, output_type=tf.int32)  # [NR*NC]
        phi_x = tf.gather(qList_tf, pos_x)  # [NR*NC]
        phi_x = tf.reshape(phi_x, (NR, NC))
    else:
        phi_x = calcFreqFromFilterRespose_TF(
            M_bool, NR, NC, vmgHx, nFilters, qList_tf
        )

    # Median 5x5
    phi_x = _median_filter_2d(phi_x, 5)

    # Optional weighted filtering
    if filterFlag:
        phi_x = tf.where(tf.math.is_nan(phi_x), tf.zeros_like(phi_x), phi_x)

        phaseFactor = tf.constant(0.01, tf.float32)

        wxQuality = tf.math.reduce_std(vmgHx, axis=0) ** 2  # [NR*NC]
        wxQuality = tf.reshape(wxQuality, (NR, NC))

        # Complex phase term: i * phaseFactor * phi_x
        arg_x = tf.complex(
            tf.zeros_like(phi_x),        # real part = 0
            phaseFactor * phi_x,         # imag part = phaseFactor * phi_x
        )
        zx = tf.complex(wxQuality, tf.zeros_like(wxQuality)) * tf.exp(arg_x)

        kernel_size = 10
        kernel = tf.ones((kernel_size, kernel_size), tf.float32) / (
            kernel_size * kernel_size
        )

        zx_real = _box_filter_2d(tf.math.real(zx), kernel_size)
        zx_imag = _box_filter_2d(tf.math.imag(zx), kernel_size)
        zxf = tf.complex(zx_real, zx_imag)

        phi_x = tf.math.angle(zxf) / phaseFactor

        # Filter mask
        M_float = tf.cast(M_bool, tf.float32)
        M_conv = _box_filter_2d(M_float, kernel_size)
        M_proc_bool = M_conv > 0.999
    else:
        M_proc_bool = M_bool

    # ------------------------------------------------------------------
    # Spatial frequency estimation: Y component
    # ------------------------------------------------------------------
    if calcMethod == "maxFreq":
        pos_y = tf.argmax(vmgHy, axis=0, output_type=tf.int32)
        phi_y = tf.gather(qList_tf, pos_y)  # NOTE: NumPy had (qList[pos]**2)
        phi_y = tf.reshape(phi_y, (NR, NC))
    else:
        phi_y = calcFreqFromFilterRespose_TF(
            M_bool, NR, NC, vmgHy, nFilters, qList_tf
        )

    phi_y = _median_filter_2d(phi_y, 5)

    if filterFlag:
        phi_y = tf.where(tf.math.is_nan(phi_y), tf.zeros_like(phi_y), phi_y)

        phaseFactor = tf.constant(0.01, tf.float32)

        wyQuality = tf.math.reduce_std(vmgHy, axis=0) ** 2
        wyQuality = tf.reshape(wyQuality, (NR, NC))

        # Complex phase term: i * phaseFactor * phi_y
        arg_y = tf.complex(
            tf.zeros_like(phi_y),
            phaseFactor * phi_y,
        )
        zy = tf.complex(wyQuality, tf.zeros_like(wyQuality)) * tf.exp(arg_y)

        kernel_size = 10
        zy_real = _box_filter_2d(tf.math.real(zy), kernel_size)
        zy_imag = _box_filter_2d(tf.math.imag(zy), kernel_size)
        zyf = tf.complex(zy_real, zy_imag)

        phi_y = tf.math.angle(zyf) / phaseFactor

    # ------------------------------------------------------------------
    # Units conversion and orientation
    # ------------------------------------------------------------------
    if freqUnits == "rad/px":
        Cx = tf.constant(2.0 * np.pi / float(NC), tf.float32)
        Cy = tf.constant(2.0 * np.pi / float(NR), tf.float32)
        phi_x = phi_x * Cx
        phi_y = phi_y * Cy
        w_phi = tf.abs(tf.complex(phi_x, phi_y))
    else:  # "ff"
        w_phi = tf.abs(tf.complex(phi_x, phi_y))

    theta_or = tf.math.atan2(-phi_y, phi_x)

    # ------------------------------------------------------------------
    # Convert outputs to NumPy
    # ------------------------------------------------------------------
    w_phi_np = w_phi.numpy().astype(np.float32)
    theta_or_np = theta_or.numpy().astype(np.float32)
    phi_x_np = phi_x.numpy().astype(np.float32)
    phi_y_np = phi_y.numpy().astype(np.float32)
    M_proc_np = M_proc_bool.numpy().astype(bool)

    return w_phi_np, theta_or_np, phi_x_np, phi_y_np, M_proc_np


#NEW
def calcSpatialFreqsFilterbank_ParVer_TF(
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
    Local spatial frequency estimation for fringe patterns (TensorFlow/GPU).

    This is a GPU-oriented wrapper around the pure-TensorFlow core
    function ``_calcSpatialFreqsFilterbank_ParVer_TF_core``.

    Design / motivation
    -------------------
    - **Wrapper (this function)**:
        * Accepts ``g`` and ``M`` as NumPy arrays or tf.Tensors.
        * Normalizes dtypes (float32 / bool).
        * Computes scalar parameters (NR, NC, wmin, wmax) as Python
          floats/ints.
        * Calls the pure-TF core function, which is decorated with
          ``@tf.function`` and runs entirely on GPU.
        * Converts the final TF tensors back to NumPy arrays
          (float32 / bool) for compatibility with the original API.

    - **Core (``_calcSpatialFreqsFilterbank_ParVer_TF_core``)**:
        * Receives only tf.Tensors (no NumPy) and scalar Python
          arguments.
        * Contains all heavy GPU work (FFTs, filter-bank, parabolic
          interpolation, phasor smoothing, etc.).
        * **Does not call ``.numpy()`` anywhere**: all computations
          remain on device until the wrapper converts the outputs.
        * Can be compiled by TensorFlow as a graph (via
          ``@tf.function``), reducing Python↔GPU overhead and allowing
          more efficient GPU execution.

    This separation is important for GPU performance because:
        - Any ``.numpy()`` call inside the heavy part forces a
          synchronous GPU→CPU transfer and prevents graph compilation.
        - Having a pure-TF core allows TensorFlow to fuse operations,
          schedule kernels efficiently, and minimize the number of
          low-level ``TFE_Py_FastPathExecute`` calls observed in the
          profiler.

    Parameters
    ----------
    g : np.ndarray or tf.Tensor, shape (NR, NC)
        Input fringe pattern (igram), real-valued. Model: g = a + b*cos(phi).
    M : np.ndarray or tf.Tensor or None, shape (NR, NC), bool or numeric
        ROI mask. Non-zero / True means valid. Default: full image valid.
    wTh : float, optional
        Spatial-frequency magnitude threshold (in "ff" units). Default: 5.
    wmin : float or None, optional
        Min spatial frequency (in "ff" units) for scanning. Default: wTh.
    wmax : float or None, optional
        Max spatial frequency (in "ff" units) for scanning.
        Default: 0.25 * ((NR+NC)/2).
    Dw : float, optional
        Frequency sampling step (in "ff" units). Default: 1.0.
    freqUnits : {"rad/px", "ff"}, optional
        Units of the output spatial frequency.
        - "ff": normalized frequency units used internally.
        - "rad/px": radians per pixel (scaling applied at the end).
    calcMethod : {"interpFreq", "ff"}, optional
        Method to obtain w_phi:
        - "interpFreq": parabolic interpolation of filter-bank response.
        - "ff": magnitude sqrt(phi_x^2 + phi_y^2) of local phase gradient.
    filterFlag : bool, optional
        If True, apply phasor-based smoothing of phi_x, phi_y and refine
        mask M_proc. If False, skip smoothing and use the original mask.

    Returns
    -------
    w_phi : np.ndarray, float32, shape (NR, NC)
        Estimated local spatial frequency magnitude.
    theta_or : np.ndarray, float32, shape (NR, NC)
        Local fringe orientation (wrapped angle).
    phi_x : np.ndarray, float32, shape (NR, NC)
        x-component of the local phase gradient.
    phi_y : np.ndarray, float32, shape (NR, NC)
        y-component of the local phase gradient.
    M_proc : np.ndarray, bool, shape (NR, NC)
        Processed mask after smoothing / support refinement.
    """
    # ------------------------------------------------------------------
    # 1) Normalize inputs (NumPy → tf.Tensor, dtypes)
    # ------------------------------------------------------------------
    g_tf = tf.convert_to_tensor(g, dtype=tf.float32)
    if g_tf.shape.rank != 2:
        raise ValueError("Input g must be a 2D array/tensor.")

    if M is None:
        M_bool = tf.ones(tf.shape(g_tf), dtype=tf.bool)
    else:
        M_tf = tf.convert_to_tensor(M)
        # any non-zero / True is valid
        if M_tf.dtype.is_bool:
            M_bool = M_tf
        else:
            M_bool = tf.not_equal(M_tf, 0)

        # Sanity check on shapes (runtime)
        tf.debugging.assert_equal(
            tf.shape(g_tf),
            tf.shape(M_bool),
            message="Mask M must have the same shape as g.",
        )

    # ------------------------------------------------------------------
    # 2) Determine NR, NC and default wmin / wmax as Python scalars
    #    (outside TF core to avoid .numpy() inside the graph)
    # ------------------------------------------------------------------
    static_shape = g_tf.shape
    NR = static_shape[0] if static_shape[0] is not None else int(tf.shape(g_tf)[0])
    NC = static_shape[1] if static_shape[1] is not None else int(tf.shape(g_tf)[1])

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
    # 3) Call pure-TF core (GPU-heavy part)
    # ------------------------------------------------------------------
    (
        w_phi_tf,
        theta_or_tf,
        phi_x_tf,
        phi_y_tf,
        M_proc_bool_tf,
    ) = _calcSpatialFreqsFilterbank_ParVer_TF_core(
        g_tf=g_tf,
        M_bool=M_bool,
        wTh=wTh,
        wmin=wmin,
        wmax=wmax,
        Dw=Dw,
        freqUnits=freqUnits,
        calcMethod=calcMethod,
        filterFlag=filterFlag,
    )

    # ------------------------------------------------------------------
    # 4) Convert outputs back to NumPy (API compatibility)
    # ------------------------------------------------------------------
    w_phi_np = w_phi_tf.numpy().astype(np.float32)
    theta_or_np = theta_or_tf.numpy().astype(np.float32)
    phi_x_np = phi_x_tf.numpy().astype(np.float32)
    phi_y_np = phi_y_tf.numpy().astype(np.float32)
    M_proc_np = M_proc_bool_tf.numpy().astype(bool)

    return w_phi_np, theta_or_np, phi_x_np, phi_y_np, M_proc_np


#@tf.function
def _calcSpatialFreqsFilterbank_ParVer_TF_core(
    *,
    g_tf: tf.Tensor,
    M_bool: tf.Tensor,
    wTh: float,
    wmin: float,
    wmax: float,
    Dw: float,
    freqUnits: str,
    calcMethod: str,
    filterFlag: bool,
):
    """
    Pure-TensorFlow core for calcSpatialFreqsFilterbank_ParVer_TF.

    This function contains the heavy GPU work (FFTs, filter bank,
    parabolic interpolation, phasor-based smoothing). It assumes that:

    - ``g_tf`` is a 2D float32 tensor of shape (NR, NC).
    - ``M_bool`` is a 2D bool tensor of shape (NR, NC).
    - All scalar parameters (NR, NC, wTh, wmin, wmax, Dw, freqUnits,
      calcMethod, filterFlag) are Python scalars / strings.

    IMPORTANT:
    - **No ``.numpy()`` calls** are allowed here. All computations must
      remain in TensorFlow so that this function can be traced and
      executed as a graph on GPU.
    - Any conversion to NumPy must be done in the outer wrapper function.
    - Use ``tf.shape(g_tf)`` and ``tf.cast`` instead of converting shapes
      to Python ints via ``.numpy()``.

    Returns
    -------
    w_phi : tf.Tensor, float32, shape (NR, NC)
    theta_or : tf.Tensor, float32, shape (NR, NC)
    phi_x : tf.Tensor, float32, shape (NR, NC)
    phi_y : tf.Tensor, float32, shape (NR, NC)
    M_proc_bool : tf.Tensor, bool, shape (NR, NC)
    """
    # ------------------------------------------------------------------
    # 0) Basic shapes and frequency grids (TF only, no .numpy())
    # ------------------------------------------------------------------
    shape = tf.shape(g_tf)
    NR_tf = shape[0]
    NC_tf = shape[1]
    
    # build u, v, U, V, q in TF (as in your original implementation)
    u = tf.range(NC_tf, dtype=tf.float32) - tf.cast(NC_tf // 2, tf.float32)
    v = tf.range(NR_tf, dtype=tf.float32) - tf.cast(NR_tf // 2, tf.float32)
    U, V = tf.meshgrid(u, v)  # (NR, NC)
    q = tf.sqrt(tf.square(U) + tf.square(V))

    # ------------------------------------------------------------------
    # 1) FFT of g, high-pass filtering (H1)
    #    (copy your existing TF code here, but using NR_tf/NC_tf and no .numpy())
    # ------------------------------------------------------------------
    G = tf.signal.fft2d(tf.cast(g_tf, tf.complex64))

    H1 = 1.0 - tf.exp(-0.5 * tf.square(q / tf.constant(wTh, tf.float32)))
    H1_shift = tf.signal.ifftshift(H1)
    G = G * tf.cast(H1_shift, tf.complex64)

    # ------------------------------------------------------------------
    # 2) Build filter bank Hx, Hy and apply to G
    #    (same logic as your current version: qList_tf, Hx, Hy, etc.)
    # ------------------------------------------------------------------
    sw = 10.0 * float(Dw)
    nFilters = tf.cast(tf.round((wmax - wmin) / Dw), tf.int32)
    tf.debugging.assert_greater(
        nFilters,
        0,
        message="nFilters <= 0: check wmin, wmax, Dw.",
    )

    qList_tf = tf.linspace(
        tf.constant(wmin, tf.float32),
        tf.constant(wmax, tf.float32),
        nFilters,
    )

    qList3 = tf.reshape(qList_tf, (nFilters, 1, 1))
    U3 = tf.reshape(U, (1, NR_tf, NC_tf))
    V3 = tf.reshape(V, (1, NR_tf, NC_tf))

    Hx = tf.exp(-0.5 * tf.square((U3 - qList3) / sw))
    Hy = tf.exp(-0.5 * tf.square((V3 - qList3) / sw))
    
    # FFT of G replicated over filter dimension if needed,
    # apply Hx/Hy, inverse FFT, magnitude, etc. (as in your original code)
    # ------------------------------------------------------------------
    # 3) Compute vmgHx, vmgHy, phi_x, phi_y, M_proc_bool
    #    and w_phi via calcFreqFromFilterRespose_TF or |grad phi|
    # ------------------------------------------------------------------
    # Match fft2d convention: move DC to (0,0) for each filter slice
    Hx = tf.signal.ifftshift(Hx, axes=(1,2))
    Hy = tf.signal.ifftshift(Hy, axes=(1,2))

    # Broadcast G to [nFilters, NR, NC] and apply filters
    G2 = tf.expand_dims(G, axis=0)  # [1,NR,NC]
    G2 = tf.broadcast_to(G2, (nFilters, NR_tf, NC_tf))  # [nFilters,NR,NC]

    gHx = tf.signal.ifft2d(G2 * tf.cast(Hx, tf.complex64))  # [nFilters,NR,NC]
    gHy = tf.signal.ifft2d(G2 * tf.cast(Hy, tf.complex64))  # [nFilters,NR,NC]

    # Magnitudes
    mgHx = tf.abs(gHx)  # float32, [nFilters,NR,NC]
    mgHy = tf.abs(gHy)  # float32, [nFilters,NR,NC]

    # Flatten to [nFilters, NR*NC]
    vmgHx = tf.reshape(mgHx, (nFilters, NR_tf * NC_tf))
    vmgHy = tf.reshape(mgHy, (nFilters, NR_tf * NC_tf))

    # ------------------------------------------------------------------
    # Spatial frequency estimation: X component
    # ------------------------------------------------------------------
    if calcMethod == "maxFreq":
        pos_x = tf.argmax(vmgHx, axis=0, output_type=tf.int32)  # [NR*NC]
        phi_x = tf.gather(qList_tf, pos_x)  # [NR*NC]
        phi_x = tf.reshape(phi_x, (NR_tf, NC_tf))
    else:
        phi_x = calcFreqFromFilterRespose_TF(
            M_bool, NR_tf, NC_tf, vmgHx, nFilters, qList_tf
        )

    # Median 5x5
    phi_x = _median_filter_2d(phi_x, 5)

    # Optional weighted filtering
    if filterFlag:
        phi_x = tf.where(tf.math.is_nan(phi_x), tf.zeros_like(phi_x), phi_x)

        phaseFactor = tf.constant(0.01, tf.float32)

        wxQuality = tf.math.reduce_std(vmgHx, axis=0) ** 2  # [NR*NC]
        wxQuality = tf.reshape(wxQuality, (NR_tf, NC_tf))

        # Complex phase term: i * phaseFactor * phi_x
        arg_x = tf.complex(
            tf.zeros_like(phi_x),        # real part = 0
            phaseFactor * phi_x,         # imag part = phaseFactor * phi_x
        )
        zx = tf.complex(wxQuality, tf.zeros_like(wxQuality)) * tf.exp(arg_x)

        kernel_size = 10
        kernel = tf.ones((kernel_size, kernel_size), tf.float32) / (
            kernel_size * kernel_size
        )

        zx_real = _box_filter_2d(tf.math.real(zx), kernel_size)
        zx_imag = _box_filter_2d(tf.math.imag(zx), kernel_size)
        zxf = tf.complex(zx_real, zx_imag)

        phi_x = tf.math.angle(zxf) / phaseFactor

        # Filter mask
        M_float = tf.cast(M_bool, tf.float32)
        M_conv = _box_filter_2d(M_float, kernel_size)
        M_proc_bool = M_conv > 0.999
    else:
        M_proc_bool = M_bool

    # ------------------------------------------------------------------
    # Spatial frequency estimation: Y component
    # ------------------------------------------------------------------
    if calcMethod == "maxFreq":
        pos_y = tf.argmax(vmgHy, axis=0, output_type=tf.int32)
        phi_y = tf.gather(qList_tf, pos_y)  # NOTE: NumPy had (qList[pos]**2)
        phi_y = tf.reshape(phi_y, (NR_tf, NC_tf))
    else:
        phi_y = calcFreqFromFilterRespose_TF(
            M_bool, NR_tf, NC_tf, vmgHy, nFilters, qList_tf
        )

    phi_y = _median_filter_2d(phi_y, 5)

    if filterFlag:
        phi_y = tf.where(tf.math.is_nan(phi_y), tf.zeros_like(phi_y), phi_y)

        phaseFactor = tf.constant(0.01, tf.float32)

        wyQuality = tf.math.reduce_std(vmgHy, axis=0) ** 2
        wyQuality = tf.reshape(wyQuality, (NR_tf, NC_tf))

        # Complex phase term: i * phaseFactor * phi_y
        arg_y = tf.complex(
            tf.zeros_like(phi_y),
            phaseFactor * phi_y,
        )
        zy = tf.complex(wyQuality, tf.zeros_like(wyQuality)) * tf.exp(arg_y)

        kernel_size = 10
        zy_real = _box_filter_2d(tf.math.real(zy), kernel_size)
        zy_imag = _box_filter_2d(tf.math.imag(zy), kernel_size)
        zyf = tf.complex(zy_real, zy_imag)

        phi_y = tf.math.angle(zyf) / phaseFactor

    # ------------------------------------------------------------------
    # Units conversion and orientation
    # ------------------------------------------------------------------
    if freqUnits == "rad/px":
        pi_tf = tf.acos(-1.0)
        Cx = 2.0 * pi_tf / tf.cast(NC_tf, tf.float32)
        Cy = 2.0 * pi_tf / tf.cast(NR_tf, tf.float32)
        phi_x = phi_x * Cx
        phi_y = phi_y * Cy
        w_phi = tf.abs(tf.complex(phi_x, phi_y))
    else:  # "ff"
        w_phi = tf.abs(tf.complex(phi_x, phi_y))

    theta_or = tf.math.atan2(-phi_y, phi_x)

    return w_phi, theta_or, phi_x, phi_y, M_proc_bool

