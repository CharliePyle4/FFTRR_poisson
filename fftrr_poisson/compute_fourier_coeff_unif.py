"""
Fourier analysis utility routines for uniform angular (theta) grids.

Contains functions for computing the fast discrete Fourier transform (DFT)
of data arranged in polar coordinates on the unit disk.
"""
import numpy as np

def compute_fourier_coeff_unif(f_values: np.ndarray) -> np.ndarray:
    """
    Compute the discrete Fourier coefficients for function values sampled
    on a uniform angular grid, using the FFT and outputting coefficients in
    symmetric order from -N/2 to N/2.

    Handles both 2D grids (shape (N, M), for multiple radii and angles)
    and 1D vectors (shape (N,), for boundary or fixed-radius slices).
    Applies FFT normalization, adjusts for grid convention, and ensures
    correct output labeling for both complex and real data.

    Parameters
    ----------
    f_values : np.ndarray
        Array of function values sampled over angle (and optionally, radius).
        Shape is (N, M) for grids, or (N,) for vectors, where N is the number of angles
        (should typically be a power of 2).

    Returns
    -------
    f_fourier_coeff : np.ndarray
        Fourier coefficients re-indexed on (-N/2, ..., 0, ..., N/2).
        Shape is (N+1, M) for 2D input or (N+1,) for 1D input.

    Notes
    -----
    - The function assumes the input grid starts at theta=2*pi/N and cycles
      through to reproduce MATLAB grid conventions.
    - The output uses an ordering where zero frequency is central, negative
      frequencies precede, positive follow (i.e., [-N/2, ..., 0, ..., N/2]).
    - Applies normalization so coefficients sum to recover the original values.

    Examples
    --------
    >>> N = 8
    >>> M = 3
    >>> f_grid = np.random.rand(N, M)
    >>> coeffs = compute_fourier_coeff_unif(f_grid)
    >>> coeffs.shape
    (9, 3)
    """
    N = f_values.shape[0]

    if f_values.ndim > 1 and f_values.shape[1] > 1:
        # Standard 2D case: each column is a fixed radius, rows are angles
        # Wrap/shift first row to end for MATLAB-style theta grid
        f_values = np.vstack([f_values[N-1, :], f_values[0:N-1, :]])
        
        # Compute FFT (DFT) along theta (rows) for every radius
        f_fourier_coeff = np.fft.fft(f_values, axis=0) / N
        
        # Reindex frequencies: [-N/2, ..., 0, ..., N/2]
        f_fourier_coeff = np.vstack([f_fourier_coeff[N//2:N, :], f_fourier_coeff[0:N//2+1, :]])
        
        # Compensate for the periodicity/node overlap at N/2 frequency
        f_fourier_coeff[0, :] = f_fourier_coeff[0, :] / 2  # Divide by 2 for the double count.
        f_fourier_coeff[N, :] = f_fourier_coeff[N, :] / 2

    else:
        # Special case: 1D array (used for boundary data)
        f_values = np.hstack([f_values[N-1], f_values[0:N-1]])
        f_fourier_coeff = np.fft.fft(f_values) / N
        f_fourier_coeff = np.hstack([f_fourier_coeff[N//2:N], f_fourier_coeff[0:N//2+1]])
        f_fourier_coeff[0] = f_fourier_coeff[0] / 2 
        f_fourier_coeff[N] = f_fourier_coeff[N] / 2

    return f_fourier_coeff 
