import time
import numpy as np

def time_method(name, func, *args, n_runs=10, **kwargs):
    """Run func(*args, **kwargs) n_runs times and report mean/std in ms.
    Returns (last_result, times_array).
    """
    # Warm-up run (especially important for TensorFlow/GPU)
    result = func(*args, **kwargs)
    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    times = np.array(times, dtype=np.float64)
    print(f"{name}: {n_runs} runs -> mean = {times.mean()*1000:.2f} ms, std = {times.std()*1000:.2f} ms")
    return result, times


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