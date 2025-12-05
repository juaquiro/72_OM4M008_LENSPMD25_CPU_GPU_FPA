import time
import numpy as np
from dataclasses import dataclass

def peaks_2d(nr, nc):
    """Approximate MATLAB `peaks` function over an nr x nc grid.

    Domain: x,y in [-3, 3].
    Returns
    -------
    Z : ndarray, shape (nr, nc), float32
        Values of the peaks test surface.
    """
    x = np.linspace(-3.0, 3.0, nc, dtype=np.float32)
    y = np.linspace(-3.0, 3.0, nr, dtype=np.float32)
    X, Y = np.meshgrid(x, y)
    Z = (3.0 * (1 - X)**2 * np.exp(-(X**2) - (Y + 1)**2)
         - 10.0 * (X/5.0 - X**3 - Y**5) * np.exp(-X**2 - Y**2)
         - (1.0/3.0) * np.exp(-(X + 1)**2 - Y**2))
    return Z.astype(np.float32)

@dataclass
class TimingResult:
    """
    Object returned by time_function() containing all timing information.

    Attributes
    ----------
    name : str
        Identifier associated with the timed function.
    n_runs : int
        Number of timed evaluations performed.
    times : np.ndarray
        Array of length n_runs with the wall-clock time of each run (in seconds).
    mean_ms : float
        Mean execution time in milliseconds.
    std_ms : float
        Standard deviation of the execution time in milliseconds.
    last_output : object
        The final returned value of the function being timed.
    """
    name: str
    n_runs: int
    times: np.ndarray
    mean_ms: float
    std_ms: float
    last_output: object


def time_function(name, f, n_runs=10, verbose=False):
    """
    Universal timer for benchmarking functions whose parameters are already bound.

    This utility accepts any *zero-argument* callable `f`, which should encapsulate
    both the function reference and its parameters (via a lambda expression or
    functools.partial). It performs one warm-up run (important for GPU/TensorFlow),
    followed by ``n_runs`` timed runs, computing mean and standard deviation.

    Parameters
    ----------
    name : str
        Identifier for printing/logging (e.g. "TF Hilbert2D", "NumPy Filterbank").
    f : callable
        Zero-argument function reference. All parameters must already be bound.
        Examples: ``lambda: func(x, y, param=3)`` or ``partial(func, x, y, param=3)``.
    n_runs : int, optional
        Number of timed runs to perform (default: 10).
    verbose : bool, optional
        If True, prints individual run times and warm-up time. Default: False.

    Returns
    -------
    TimingResult
        A dataclass containing mean time, std, list of run times, and last output.

    Examples
    --------
    **1) Using a lambda to bind parameters**

    >>> f = lambda: calcSpatialFreqsHilbert2D_NumPy(g, M=M, filter_orientation=10)
    >>> res = time_function("NumPy Hilbert2D", f, n_runs=10, verbose=False)
    >>> pred_w_phi, pred_theta_or, pred_phi_x, pred_phi_y, M_proc = res.last_output

    **2) Using functools.partial to freeze arguments**

    >>> from functools import partial
    >>> f = partial(calcSpatialFreqsHilbert2D_TF, g, M=M, filter_orientation=theta0)
    >>> res = time_function("TF Hilbert2D", f, n_runs=20)

    **3) TensorFlow GPU function**

    >>> f_fb = lambda: calcSpatialFreqsFilterbank_ParVer_TF(g, M=M)
    >>> res_fb = time_function("TF Filterbank", f_fb, n_runs=5)
    >>> w_phi, theta_or, phi_x, phi_y, Mproc = res_fb.last_output

    **4) Timing a simple function**

    >>> f = lambda: np.sum(np.random.randn(1000,1000))
    >>> res = time_function("Random matrix sum", f, n_runs=50)

    Notes
    -----
    **Difference between lambda and partial**

    *Lambda (anonymous function)*  
    - Defines a **new function**.  
    - Can contain arbitrary Python logic: loops, multiple calls, noise injection, etc.  
    - Captures variables from the surrounding scope.  
    - Does **not** preserve metadata (function name, signature).  

    Example:  
    >>> f = lambda: calcSpatialFreqsHilbert2D_NumPy(g + noise(), M=M)

    *functools.partial (freezes some parameters)*  
    - Does **not** create new logic.  
    - Produces a version of an existing function with some arguments “pre-filled”.  
    - Preserves metadata (function name, docstring).  
    - Safer and more explicit when you only want to bind arguments.  

    Example:  
    >>> f = partial(calcSpatialFreqsHilbert2D_NumPy, g, M=M, filter_orientation=10)

    Use **partial** for clean binding of arguments.  
    Use **lambda** when you need custom inline logic.

    - The warm-up run is *not* included in the timing statistics.
    - The function returns the **last output** of the timed function, enabling
      direct use in plotting or numerical validation.

    """
    # Warm-up run
    t0 = time.perf_counter()
    last_output = f()
    t1 = time.perf_counter()
    warmup_time = t1 - t0

    if verbose:
        print(f"[{name}] Warm-up time: {warmup_time*1000:.2f} ms")

    # Timed runs
    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        last_output = f()
        t1 = time.perf_counter()
        dt = t1 - t0
        times.append(dt)

        if verbose:
            print(f"[{name}] Run {i+1}/{n_runs}: {dt*1000:.2f} ms")

    times = np.array(times, dtype=np.float64)
    mean_ms = times.mean() * 1000.0
    std_ms = times.std() * 1000.0

    # Summary
    print(f"{name}: {n_runs} runs -> mean = {mean_ms:.2f} ms, std = {std_ms:.2f} ms")

    return TimingResult(
        name=name,
        n_runs=n_runs,
        times=times,
        mean_ms=mean_ms,
        std_ms=std_ms,
        last_output=last_output,
    )
