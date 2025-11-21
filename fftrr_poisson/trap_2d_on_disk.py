"""
2D trapezoidal integration rule on the disk.

Implements a composite trapezoidal rule for integrating functions sampled
on a polar mesh covering the unit disk. Handles potentially non-uniform
meshes in both the radial and angular directions.
"""

import numpy as np

def trap_2d_on_disk(
    f: np.ndarray,
    iRadius: np.ndarray,
    iAngle: np.ndarray
) -> float:
    """
    Numerically integrate a function over the unit disk using the 2D
    composite trapezoidal rule in polar coordinates.

    Approximates the integral of a function sampled at nodes on a polar mesh,
    accounting for possible non-uniform spacing in both radial and angular directions.

    Parameters
    ----------
    f : ndarray of shape (N, M)
        Function values evaluated at mesh grid (angles × radii),
        where N is the number of angular points and M is the number of radii.
    iRadius : ndarray of shape (M,)
        Radial mesh points, increasing from 0 up to disk radius.
    iAngle : ndarray of shape (N-1,)
        Angular mesh points (should cover (0, 2π); the routine prepends 0).

    Returns
    -------
    trap_2d : float
        Approximation of the double integral of f over the disk:
        ∫∫ f(r, θ) r dr dθ ≈ trap_2d

    Notes
    -----
    - The first and last angular segments are handled to ensure periodicity (0/2π wrap-around).
    - The method forms a composite rule combining radial and angular mesh widths.
    - Intended for use with solutions or source terms sampled on polar grids.

 
    Examples
    --------
    >>> N, M = 8, 5
    >>> theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    >>> radii = np.linspace(0, 1, M)
    >>> X = np.cos(theta[:, None]) * radii
    >>> Y = np.sin(theta[:, None]) * radii
    >>> F = X**2 + Y**2
    >>> trap_2d_on_disk(F, radii, theta[1:])
    ...
    """
    # Ensure angle array covers [0, 2π] by prepending 0
    iAngle = np.concatenate(([0], iAngle.ravel()))
    # Wrap f for angular periodicity (append first row at end)
    f = np.vstack([f, f[0, :]])

    N = len(iAngle)
    M = len(iRadius)

    # Compute radial mesh widths; delta[m] = r_m - r_{m-1}
    delta = np.zeros(M + 1)
    for m in range(1, M):
        delta[m] = iRadius[m] - iRadius[m - 1]

    # Compute angular mesh widths; tau[n] = θ_n - θ_{n-1}
    tau = np.zeros(N + 1)
    for n in range(1, N):
        tau[n] = iAngle[n] - iAngle[n - 1]

    # Compute contribution from outermost (largest radius) ring
    trap_2d_rm = 0
    for n in range(N):
        trap_2d_rm += f[n, M - 1] * (tau[n] + tau[n + 1])
    trap_2d_rm = (delta[1] * iRadius[M - 1] / 4) * trap_2d_rm

    # Main computation: sum over all interior radial shells
    trap_2d = 0
    for m in range(1, M - 1):
        trap_2d_sum = 0
        for n in range(N):
            trap_2d_sum += f[n, m] * (tau[n] + tau[n + 1])
        trap_2d_sum = (1 / 4) * iRadius[m] * (delta[m] + delta[m + 1]) * trap_2d_sum
        trap_2d += trap_2d_sum

    trap_2d = trap_2d + trap_2d_rm

    return float(trap_2d)
