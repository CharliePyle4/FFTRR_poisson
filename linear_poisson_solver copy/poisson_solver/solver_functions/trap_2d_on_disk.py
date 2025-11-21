import numpy as np

def trap_2d_on_disk(f: np.ndarray, iRadius: np.ndarray, iAngle: np.ndarray) -> float:
    """
    Parameters
    ----------
    f : np.ndarray
        Function values sampled on the polar grid (angles × radii).
    iRadius : np.ndarray
        Radial mesh points.
    iAngle : np.ndarray
        Angular mesh points (excluding 0 and 2π).

    Returns
    -------
    trap_2d : float
        Approximation of the 2D integral on the disk using trapezoidal rule.
    """

    # We need 0 and 2pi in iAngle for the integration.
    iAngle = np.concatenate(([0], iAngle.ravel()))

    # Similarly, we also need to modify 'f' for the integration.
    f = np.vstack([f, f[0, :]])

    # Length indices.
    N = len(iAngle)
    M = len(iRadius)

    # Radial mesh length.
    delta = np.zeros(M + 1)
    for m in range(1, M):
        # We leave the first and last entry 0, to simplify the notation
        # for the computation of the integral.
        delta[m] = iRadius[m] - iRadius[m - 1]

    # Azimuthal mesh length.
    tau = np.zeros(N + 1)
    for n in range(1, N):
        # Same comment for 'delta'.
        tau[n] = iAngle[n] - iAngle[n - 1]

    # The discretization of the 2D integral is quite complicated. 
    # See the attachment for the discretization process.

    # Compute the integral at 'r = M' separately.
    trap_2d_rm = 0
    for n in range(N):
        trap_2d_rm += f[n, M - 1] * (tau[n] + tau[n + 1])
    trap_2d_rm = (delta[1] * iRadius[M - 1] / 4) * trap_2d_rm

    # The main computation (double summation).
    trap_2d = 0
    for m in range(1, M - 1):
        trap_2d_sum = 0
        for n in range(N):
            trap_2d_sum += f[n, m] * (tau[n] + tau[n + 1])
        trap_2d_sum = (1 / 4) * iRadius[m] * (delta[m] + delta[m + 1]) * trap_2d_sum
        trap_2d += trap_2d_sum

    trap_2d = trap_2d + trap_2d_rm

    return float(trap_2d)
