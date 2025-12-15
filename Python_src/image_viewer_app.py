"""
image_viewer_app.py

Simple image viewer similar in spirit to MATLAB's Image Viewer app.

Features
--------
- Display a 2D grayscale or RGB image (NumPy array or image file).
- Independent GUI process (does not block the calling notebook/console).
- Multiple instances can be opened simultaneously.
- Zoom & pan via Matplotlib's navigation toolbar.
- Contrast adjustment for grayscale images (vmin/vmax sliders).
- Colormap selection for grayscale images (gray, viridis, jet, etc.).
- Pixel info shown in status bar (x, y, value).

Requirements
------------
pip install matplotlib PyQt5 numpy

(If you prefer PySide6, you can adapt the Qt imports accordingly.)

Usage
-----
From console (image file):
    python image_viewer_app.py path/to/image.png --title "My image"

From Python / Jupyter (NumPy array):
    from image_viewer_app import launch_image_viewer
    launch_image_viewer(img, title="My image")
"""

import argparse
import os
import sys
import tempfile
import subprocess
from typing import Optional

import numpy as np
import matplotlib

# Use generic Qt backend (works with Qt5 or Qt6)
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

# Try PyQt5 first; if not available, use PyQt6
try:
    from PyQt5 import QtWidgets, QtCore
    QT_LIB = "PyQt5"
except ImportError:
    from PyQt6 import QtWidgets, QtCore
    QT_LIB = "PyQt6"


# ---------------------------------------------------------------------
# Core Qt viewer
# ---------------------------------------------------------------------

class ImageViewer(QtWidgets.QMainWindow):
    def __init__(self, image: np.ndarray, title: str = "Image Viewer"):
        super().__init__()

        self.setWindowTitle(title)

        if image.ndim == 2:
            self.is_grayscale = True
        elif image.ndim == 3 and image.shape[2] in (3, 4):
            self.is_grayscale = False
        else:
            raise ValueError("Only 2D grayscale or 3D RGB/RGBA images are supported.")

        self.image = image.astype(np.float32, copy=False)
        self._data_min = float(np.min(self.image))
        self._data_max = float(np.max(self.image))
        if self._data_max == self._data_min:
            self._data_max = self._data_min + 1.0

        # For grayscale contrast mapping
        self._vmin = self._data_min
        self._vmax = self._data_max

        self._current_cmap = "gray"

        self._build_ui()
        self._connect_events()
        self._update_image()

    def _build_ui(self):
        # Central widget + layout
        central = QtWidgets.QWidget(self)
        vlayout = QtWidgets.QVBoxLayout(central)

        # Matplotlib figure & canvas
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()

        # First add toolbar, then canvas
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        vlayout.addWidget(self.toolbar)
        vlayout.addWidget(self.canvas, stretch=1)

        # Controls (only most relevant: contrast, colormap)
        controls = QtWidgets.QWidget(self)
        controls_layout = QtWidgets.QHBoxLayout(controls)

        if self.is_grayscale:
            # Contrast group
            contrast_group = QtWidgets.QGroupBox("Contrast (display range)")
            contrast_layout = QtWidgets.QGridLayout(contrast_group)

            # Orientation differs between PyQt5 and PyQt6
            if QT_LIB == "PyQt5":
                orientation = QtCore.Qt.Horizontal
            else:  # PyQt6
                orientation = QtCore.Qt.Orientation.Horizontal

            self.slider_min = QtWidgets.QSlider(orientation)
            self.slider_max = QtWidgets.QSlider(orientation)


            self.slider_min.setRange(0, 1000)
            self.slider_max.setRange(0, 1000)
            self.slider_min.setValue(0)
            self.slider_max.setValue(1000)

            self.label_min = QtWidgets.QLabel(f"Min: {self._data_min:.3g}")
            self.label_max = QtWidgets.QLabel(f"Max: {self._data_max:.3g}")

            # Colormap combo
            self.cmap_combo = QtWidgets.QComboBox()
            self.cmap_combo.addItems([
                "gray", "viridis", "plasma", "inferno",
                "magma", "jet", "hot", "cool", "bone"
            ])
            self.cmap_combo.setCurrentText("gray")

            contrast_layout.addWidget(QtWidgets.QLabel("Lower"), 0, 0)
            contrast_layout.addWidget(self.slider_min, 0, 1)
            contrast_layout.addWidget(self.label_min, 0, 2)

            contrast_layout.addWidget(QtWidgets.QLabel("Upper"), 1, 0)
            contrast_layout.addWidget(self.slider_max, 1, 1)
            contrast_layout.addWidget(self.label_max, 1, 2)

            contrast_layout.addWidget(QtWidgets.QLabel("Colormap"), 2, 0)
            contrast_layout.addWidget(self.cmap_combo, 2, 1, 1, 2)

            controls_layout.addWidget(contrast_group)

        controls_layout.addStretch(1)
        vlayout.addWidget(controls)

        self.setCentralWidget(central)

        # Status bar for pixel info
        self.status = self.statusBar()
        self.status.showMessage("Ready")

    # ------------------------------------------------------------------
    # Event wiring
    # ------------------------------------------------------------------
    def _connect_events(self):
        if self.is_grayscale:
            self.slider_min.valueChanged.connect(self._on_contrast_change)
            self.slider_max.valueChanged.connect(self._on_contrast_change)
            self.cmap_combo.currentTextChanged.connect(self._on_cmap_change)

        # Matplotlib event for mouse motion
        self.cid_motion = self.canvas.mpl_connect(
            "motion_notify_event", self._on_motion
        )

    # ------------------------------------------------------------------
    # Image update
    # ------------------------------------------------------------------
    def _update_image(self):
        self.ax.clear()
        self.ax.set_axis_off()

        if self.is_grayscale:
            self._im = self.ax.imshow(
                self.image,
                cmap=self._current_cmap,
                vmin=self._vmin,
                vmax=self._vmax,
                interpolation="nearest",
                origin="upper"
            )
        else:
            # Assume already in [0,1] or [0,255]; scale to [0,1]
            img = self.image
            if img.dtype != np.float32 and img.dtype != np.float64:
                img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0

            self._im = self.ax.imshow(
                img,
                interpolation="nearest",
                origin="upper"
            )

        self.ax.set_xlim(0, self.image.shape[1])
        self.ax.set_ylim(self.image.shape[0], 0)

        self.canvas.draw_idle()

    def _on_contrast_change(self):
        vmin_frac = self.slider_min.value() / 1000.0
        vmax_frac = self.slider_max.value() / 1000.0

        # Ensure min < max
        if vmin_frac >= vmax_frac:
            # Tiny step to avoid zero-width window
            vmax_frac = min(vmin_frac + 0.001, 1.0)
            self.slider_max.blockSignals(True)
            self.slider_max.setValue(int(vmax_frac * 1000))
            self.slider_max.blockSignals(False)

        self._vmin = self._data_min + vmin_frac * (self._data_max - self._data_min)
        self._vmax = self._data_min + vmax_frac * (self._data_max - self._data_min)

        self.label_min.setText(f"Min: {self._vmin:.3g}")
        self.label_max.setText(f"Max: {self._vmax:.3g}")

        if hasattr(self, "_im"):
            self._im.set_clim(self._vmin, self._vmax)
            self.canvas.draw_idle()

    def _on_cmap_change(self, text: str):
        self._current_cmap = text
        if hasattr(self, "_im"):
            self._im.set_cmap(self._current_cmap)
            self.canvas.draw_idle()

    def _on_motion(self, event):
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        j = int(round(x))
        i = int(round(y))

        if i < 0 or j < 0 or i >= self.image.shape[0] or j >= self.image.shape[1]:
            return

        if self.is_grayscale:
            val = self.image[i, j]
            msg = f"x={j}, y={i}, value={val:.3g}"
        else:
            pix = self.image[i, j, :]
            if pix.size == 3:
                msg = f"x={j}, y={i}, RGB=({pix[0]:.3g}, {pix[1]:.3g}, {pix[2]:.3g})"
            else:
                msg = (f"x={j}, y={i}, RGBA="
                       f"({pix[0]:.3g}, {pix[1]:.3g}, {pix[2]:.3g}, {pix[3]:.3g})")

        self.status.showMessage(msg)


# ---------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------

def _load_image_from_path(path: str) -> np.ndarray:
    import matplotlib.image as mpimg

    if path.lower().endswith(".npy"):
        arr = np.load(path)
        return arr
    else:
        img = mpimg.imread(path)
        return img


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------

def _run_gui(image_path: str, title: Optional[str] = None):
    img = _load_image_from_path(image_path)
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    viewer = ImageViewer(img, title=title or os.path.basename(image_path))
    viewer.resize(900, 700)
    viewer.show()

    # exec_() in PyQt5, exec() in PyQt6
    if QT_LIB == "PyQt5":
        ret = app.exec_()
    else:
        ret = app.exec()

    sys.exit(ret)



def main():
    parser = argparse.ArgumentParser(description="Standalone image viewer app.")
    parser.add_argument("image_path", help="Path to image file or .npy array.")
    parser.add_argument("--title", default=None, help="Optional window title.")
    args = parser.parse_args()

    _run_gui(args.image_path, title=args.title)


# ---------------------------------------------------------------------
# Public API: launch_image_viewer (spawns separate process)
# ---------------------------------------------------------------------

def launch_image_viewer(image: np.ndarray, title: str = "Image Viewer"):
    """
    Launch the image viewer in a separate Python process.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale or 3D RGB/RGBA array.
    title : str, optional
        Window title.

    Notes
    -----
    - Does NOT block the caller (non-modal).
    - Works from a notebook or from a regular Python REPL.
    - Several instances can be launched independently.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")

    # Save image to a temporary .npy file
    tmp_dir = tempfile.gettempdir()
    tmp_name = next(tempfile._get_candidate_names()) + ".npy"
    tmp_path = os.path.join(tmp_dir, tmp_name)
    np.save(tmp_path, image)

    # Build command to spawn new process
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        tmp_path,
        "--title",
        title
    ]

    # Start process without waiting
    subprocess.Popen(cmd)


if __name__ == "__main__":
    main()
