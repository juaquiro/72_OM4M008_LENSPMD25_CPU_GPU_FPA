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
