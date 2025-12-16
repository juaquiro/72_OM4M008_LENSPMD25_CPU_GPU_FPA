"""
notebook_bootstrap.py

Development-only path helper helper for Jupyter notebooks.

This module is intended to be placed inside the `Notebooks/` directory of
a Python project with the following layout:

    Python_src/
    ├── image_viewer_app.py
    ├── other_modules.py
    └── Notebooks/
        ├── notebook_add_path.py
        └── example_notebook.ipynb

When imported from a notebook located in `Python_src/Notebooks/`, this module
temporarily adds the parent directory (`Python_src/`) to `sys.path`. This allows
the notebook to import project modules under active development, e.g.:

    from image_viewer_app import launch_image_viewer

Important notes:
- The modification to `sys.path` is temporary and applies only to the current
  Python process / Jupyter kernel.
- No environment variables (e.g. PYTHONPATH) are modified.
- No changes persist after the kernel is restarted.
- This helper is meant for interactive development and debugging only.

Once the project is converted into a proper Python package and installed
(e.g. using `pip install -e .`), this bootstrap module should be removed and
imports should rely on standard package mechanisms instead.
"""

import sys
from pathlib import Path


def _add_project_src_to_path(verbose: bool = True) -> Path:
    """
    Add the project source directory (parent of the notebook/script)
    to sys.path.

    Returns
    -------
    Path
        The project source directory added to sys.path.
    """
    try:
        # Script execution (file lives in Python_src/Notebooks/)
        base_dir = Path(__file__).resolve().parent
    except NameError:
        # Jupyter / IPython (cwd == Python_src/Notebooks/)
        base_dir = Path.cwd().resolve()

    # Project source directory is the parent of Notebooks/
    project_src = base_dir.parent

    if project_src.is_dir() and str(project_src) not in sys.path:
        sys.path.insert(0, str(project_src))
        if verbose:
            print(f"[notebook_bootstrap] Added to sys.path: {project_src}")
    else:
        if verbose:
            print(f"[notebook_bootstrap] Path already present: {project_src}")

    return project_src

# Execute on import notebook_add_path
PROJECT_SRC = _add_project_src_to_path()