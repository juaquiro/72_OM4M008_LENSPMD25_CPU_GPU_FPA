#!/usr/bin/env python3
"""
debug_calcSpatialFreqsFilterbank_ParVer_TF.py

Standalone driver to debug `calcSpatialFreqsFilterbank_ParVer_TF` from:
    Python_src/FPA_TensorFlow.py

Why this exists
---------------
Debugging through a Jupyter notebook in VS Code often complicates breakpoints,
figure rendering, and module reloading. This script is meant to be placed under:

    Python_src/Scripts/

and launched with the VS Code debugger ("Python: Current File"). Because it is
a normal Python process, breakpoints set inside `FPA_TensorFlow.py` should be hit
reliably (subject to TensorFlow graph compilation notes below).

TensorFlow debugging note
-------------------------
If the target function (or any inner function it calls) is wrapped with
`@tf.function`, Python-level stepping may be skipped because execution happens
in a traced graph. For step-by-step debugging, run with:

    --eager

which enables eager execution for tf.function via:
    tf.config.run_functions_eagerly(True)

This is slower, but makes breakpoints behave as expected.

Usage
-----
From the project root (one level above Python_src):

    python Python_src/Scripts/debug_calcSpatialFreqsFilterbank_ParVer_TF.py --help

Typical debug run (recommended):
    python Python_src/Scripts/debug_calcSpatialFreqsFilterbank_ParVer_TF.py --eager --break

"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np


def _add_project_src_to_path(verbose: bool = True) -> Path:
    """
    Add Python_src/ (project source dir) to sys.path based on this script location.

    Returns
    -------
    Path
        The directory added to sys.path (Python_src).
    """
    base_dir = Path(__file__).resolve().parent  # Python_src/Scripts
    project_src = base_dir.parent               # Python_src
    if project_src.is_dir():
        s = str(project_src)
        if s not in sys.path:
            sys.path.insert(0, s)
            if verbose:
                print(f"[path] added to sys.path: {project_src}")
        else:
            if verbose:
                print(f"[path] already in sys.path: {project_src}")
    else:
        raise FileNotFoundError(f"Could not locate Python_src at: {project_src}")
    return project_src


def make_synth_interferogram(
    nr: int = 558,
    nc: int = 553,
    w0: Tuple[float, float] = (np.pi / 4, np.pi / 4),
    bias: float = 127.0,
    contrast: float = 60.0,
    noise_std: float = 3.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a simple synthetic cosine interferogram and a full-true mask.

    Returns
    -------
    g : (nr, nc) float32
        Interferogram intensity image.
    M : (nr, nc) bool
        ROI mask.
    """
    rng = np.random.default_rng(seed)
    y, x = np.meshgrid(
        np.arange(nr, dtype=np.float32),
        np.arange(nc, dtype=np.float32),
        indexing="ij",
    )
    x = x - 0.5 * nc
    y = y - 0.5 * nr

    wx, wy = map(np.float32, w0)
    phi = wx * x + wy * y

    g = bias + contrast * np.cos(phi) + noise_std * rng.standard_normal((nr, nc)).astype(np.float32)
    g = np.clip(np.round(g), 0, 255).astype(np.float32)

    M = np.ones((nr, nc), dtype=bool)
    g = g * M.astype(np.float32)
    return g, M


def configure_tensorflow(eager: bool, gpu: bool, verbose: bool = True) -> None:
    import tensorflow as tf

    if verbose:
        print(f"[tf] version: {tf.__version__}")
        print(f"[tf] built with CUDA: {tf.test.is_built_with_cuda()}")
        print(f"[tf] GPUs visible: {tf.config.list_physical_devices('GPU')}")

    # Optionally hide GPUs (forces CPU)
    if not gpu:
        try:
            tf.config.set_visible_devices([], "GPU")
            if verbose:
                print("[tf] GPU disabled (CPU-only run).")
        except Exception as e:
            # If TF already initialized, visibility changes may fail.
            print(f"[tf] Could not disable GPU (maybe already initialized): {e}")

    # Avoid TF grabbing all VRAM at once (when GPUs are used)
    try:
        for d in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(d, True)
    except Exception as e:
        if verbose:
            print(f"[tf] Could not set memory growth: {e}")

    if eager:
        tf.config.run_functions_eagerly(True)
        if verbose:
            print("[tf] run_functions_eagerly(True) enabled (better for debugging).")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eager", action="store_true", help="Force eager execution for tf.function (debug-friendly).")
    parser.add_argument("--cpu", action="store_true", help="Force CPU by hiding GPUs from TensorFlow.")
    parser.add_argument("--break", dest="do_break", action="store_true", help="Call breakpoint() before running.")
    parser.add_argument("--nr", type=int, default=558, help="Rows for synthetic interferogram.")
    parser.add_argument("--nc", type=int, default=553, help="Cols for synthetic interferogram.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for synthetic data.")
    parser.add_argument("--calcMethod", type=str, default="maxFreq", help="Forwarded to calcSpatialFreqsFilterbank_ParVer_TF.")
    parser.add_argument("--no-plots", action="store_true", help="Do not show matplotlib figures.")
    args = parser.parse_args()

    _add_project_src_to_path(verbose=True)

    # TensorFlow config must happen before importing your TF-heavy module in many cases.
    configure_tensorflow(eager=args.eager, gpu=(not args.cpu), verbose=True)

    # Now import the function under test.
    # NOTE: use a plain import (no importlib.reload) for normal script debugging.
    from FPA_TensorFlow import calcSpatialFreqsFilterbank_ParVer_TF

    g, M = make_synth_interferogram(nr=args.nr, nc=args.nc, seed=args.seed)

    if args.do_break:
        # If you run via VS Code Debugger, you'll stop here. Then set/confirm breakpoints
        # inside FPA_TensorFlow.py and continue.
        breakpoint()

    print("[run] calling calcSpatialFreqsFilterbank_ParVer_TF(...)")
    pred_w_phi, pred_theta_or, pred_phi_x, pred_phi_y, M_proc = calcSpatialFreqsFilterbank_ParVer_TF(
        g, M, calcMethod=args.calcMethod
    )
    print("[run] done.")
    print(f"[out] pred_w_phi: {getattr(pred_w_phi, 'shape', None)}  dtype={getattr(pred_w_phi, 'dtype', None)}")
    print(f"[out] pred_theta_or: {getattr(pred_theta_or, 'shape', None)}  dtype={getattr(pred_theta_or, 'dtype', None)}")
    print(f"[out] pred_phi_x: {getattr(pred_phi_x, 'shape', None)}  dtype={getattr(pred_phi_x, 'dtype', None)}")
    print(f"[out] pred_phi_y: {getattr(pred_phi_y, 'shape', None)}  dtype={getattr(pred_phi_y, 'dtype', None)}")
    print(f"[out] M_proc: {getattr(M_proc, 'shape', None)}  dtype={getattr(M_proc, 'dtype', None)}")

    # Optional quick visuals (useful to sanity check during debugging)
    if not args.no_plots:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(g, cmap="gray")
        plt.title("Synthetic interferogram (g)")
        plt.colorbar()
        plt.show()

        # If outputs are tensors, convert to numpy for display
        def _to_np(a):
            try:
                import tensorflow as tf
                if isinstance(a, tf.Tensor):
                    return a.numpy()
            except Exception:
                pass
            return np.asarray(a)

        w = _to_np(pred_w_phi)
        plt.figure()
        plt.imshow(w, cmap="viridis")
        plt.title("pred_w_phi")
        plt.colorbar()
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
