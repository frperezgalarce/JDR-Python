from src.metrics import integrate_cross_spectrum_real
from src.read_data import read_ogle_dat
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from concurrent.futures import ProcessPoolExecutor


def _compute_Ix(args):
    f_min, f_max, t1, x = args
    return integrate_cross_spectrum_real(f_min, f_max, t1, x, t1, x, jacobian="df")


def _compute_Iy(args):
    f_min, f_max, t2, y = args
    return integrate_cross_spectrum_real(f_min, f_max, t2, y, t2, y, jacobian="df")


def _compute_Ixy(args):
    f_min, f_max, t1, x, t2, y = args
    return integrate_cross_spectrum_real(f_min, f_max, t1, x, t2, y, jacobian="df")


def jdr_parallel(file1, file2, alpha=0.5, delta_f=0.001, n_jobs=3):

    Irrlyr1 = read_ogle_dat(file1)
    Irrlyr2 = read_ogle_dat(file2)

    t1 = Irrlyr1.iloc[:, 0].to_numpy().astype(np.float32)
    x = Irrlyr1.iloc[:, 1].to_numpy().astype(np.float32)

    t2 = Irrlyr2.iloc[:, 0].to_numpy().astype(np.float32)
    y = Irrlyr2.iloc[:, 1].to_numpy().astype(np.float32)

    beta = alpha * (1 - alpha)

    delta_t = (np.max(t1) - np.min(t1)) / len(t1)

    f_min = delta_f
    f_max = 1.0 / (2.0 * delta_t)

    # Prepare arguments
    args_Ix = (f_min, f_max, t1, x)
    args_Iy = (f_min, f_max, t2, y)
    args_Ixy = (f_min, f_max, t1, x, t2, y)

    # Parallel execution
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(_compute_Ix, args_Ix),
            executor.submit(_compute_Iy, args_Iy),
            executor.submit(_compute_Ixy, args_Ixy),
        ]

        Ix, Iy, Ixy = [f.result() for f in futures]

    J = beta * (Ix + Iy - 2.0 * Ixy) / (2.0 * np.pi)

    '''
    print("\nResults:")
    print(file1, file2)
    print(f"Ix  = {Ix}")
    print(f"Iy  = {Iy}")
    print(f"Ixy = {Ixy}")
    print(f"J   = {J}")
    ''' 
    return J



def jdr(file1, file2, alpha=0.5, delta_f=0.001):
    """
    Compute the J-distance ratio (JDR) between two irregular time series
    based on their cross-spectral representation.

    This function reads two time series from OGLE-like `.dat` files and
    evaluates a distance measure derived from the integrated auto- and
    cross-spectra over a specified frequency band.

    Parameters
    ----------
    file1 : str
        Path to the first time series file. The file must contain at least
        two columns: time and observed value.

    file2 : str
        Path to the second time series file. Same format as `file1`.

    alpha : float, optional (default=0.5)
        Weighting parameter controlling the contribution of each signal.
        The scaling factor is defined as:
            beta = alpha * (1 - alpha)
        Typical values are in the range (0, 1).

    delta_f : float, optional (default=0.001)
        Lower bound of the frequency integration range. This avoids
        numerical instability near zero frequency.

    Returns
    -------
    J : float
        J-distance ratio between the two time series. Lower values indicate
        higher similarity in their spectral structure, while larger values
        indicate greater dissimilarity.

    Notes
    -----
    The method computes:

        Ix  = ∫ S_xx(f) df   (auto-spectrum of x)
        Iy  = ∫ S_yy(f) df   (auto-spectrum of y)
        Ixy = ∫ S_xy(f) df   (cross-spectrum)

    over the frequency band [f_min, f_max], where:

        f_min = delta_f
        f_max = 1 / (2 * Δt)

    and Δt is the average sampling interval.

    The final metric is:

        J = β * (Ix + Iy - 2 * Ixy) / (2π)

    where β = α(1 - α).

    This formulation is analogous to a spectral distance measure and is
    closely related to energy differences in the frequency domain.

    Assumptions
    -----------
    - Time series may be irregularly sampled.
    - Signals are internally demeaned and normalized during spectral estimation.
    - Integration is performed using numerical quadrature (SciPy backend).

    Side Effects
    ------------
    Prints intermediate results (Ix, Iy, Ixy, J) to stdout.

    Example
    -------
    >>> J = jdr("series1.dat", "series2.dat", alpha=0.5)
    >>> print(J)

    References
    ----------
    - xxxx
    """
    Irrlyr1 = read_ogle_dat(file1)
    Irrlyr2 = read_ogle_dat(file2)
    
    t1 = Irrlyr1.iloc[:, 0].to_numpy()
    x = Irrlyr1.iloc[:, 1].to_numpy()

    t2 = Irrlyr2.iloc[:, 0].to_numpy()
    y = Irrlyr2.iloc[:, 1].to_numpy()
    
    
    beta = alpha * (1 - alpha)
    st = t1

    delta_t = (np.max(st) - np.min(st)) / len(st)

    f_min = delta_f

    f_max = 1.0 / (2.0 * delta_t)

    Ix = integrate_cross_spectrum_real(f_min, f_max, t1, x, t1, x,  jacobian="df")
    Iy = integrate_cross_spectrum_real(f_min, f_max, t2, y, t2, y,  jacobian="df")

    f_max_xy = min(f_max, f_max)  # kept for structural similarity to your R
    Ixy = integrate_cross_spectrum_real(f_min, f_max_xy, t1, x, t2, y,  jacobian="df")

    J = beta * (Ix + Iy - 2.0 * Ixy) / (2.0 * np.pi)

    print("\nResults:")
    print(file1, file2)
    print(f"Ix  = {Ix}")
    print(f"Iy  = {Iy}")
    print(f"Ixy = {Ixy}")
    print(f"J   = {J}")

    return J
