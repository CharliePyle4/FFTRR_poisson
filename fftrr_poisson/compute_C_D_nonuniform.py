"""
Radial quadrature coefficients for non-uniform mesh FFTRR Poisson solver.

This module computes the C and D integrals required for solving the Poisson
equation on a disk with a non-uniform radial mesh, supporting both trapezoidal
and generalized (non-uniform) Simpson's rule quadrature, as adapted from the
work of Borges and Daripa.
"""
import numpy as np
from .nonuniform_simps_rule import nonuniform_simps_rule

def compute_C_D_nonuniform(r_m: np.ndarray, f_fourier_coeff: np.ndarray, quad_rule: int):
    """
    Compute the C and D coefficient matrices for each angular Fourier mode,
    using either the trapezoidal or non-uniform Simpson's rule, over
    arbitrary (monotonically increasing) radial mesh points.

    Parameters
    ----------
    r_m : ndarray of shape (M,)
        Radial mesh points defining the boundaries of each shell (must be strictly increasing).
    f_fourier_coeff : ndarray of shape (N+1, M)
        Angular Fourier coefficients of the source function at each radius.
        Indexed [mode, radius], where mode runs from -N/2 to N/2.
    quad_rule : int
        Which quadrature rule to use:
            1 : Trapezoidal rule
            2 : Non-uniform Simpson's rule (uses nonuniform_simps_rule)

    Returns
    -------
    C : ndarray of shape (N//2 + 1, M-1)
        Positive angular frequency coefficient matrix.
    D : ndarray of shape (N//2 + 1, M-1)
        Positive angular frequency coefficient matrix.

    Notes
    -----
    - All arrays use only the non-negative ("unique") angular Fourier modes.
    - Simpson's rule fallback to trapezoidal for the boundary/end intervals.
    - The special handling of n=0 (mean frequency) avoids undefined log terms.

    References
    ----------
    See: L. Borges and P. Daripa, "A Fast Parallel Algorithm for the Poisson Equation on a Disk," J Comput Phys 169(1):151–192, 2001.
    """

    M = len(r_m)
    N = f_fourier_coeff.shape[0] - 1

    # Declare matrices
    C = np.zeros((N // 2 + 1, M - 1), dtype=complex)
    D = np.zeros((N // 2 + 1, M - 1), dtype=complex)

    # Radial step sizes (may be non-uniform):
    delta = np.zeros(M - 1)
    for i in range(M - 1):
        delta[i] = r_m[i + 1] - r_m[i]

    # The frequency index convention is 0 <= n <= N//2 only.

    # Trapezoidal Rule %
    if quad_rule == 1:
        for i in range(M - 1):  # Loop over radial intervals
            for n in range(1, N // 2 + 1):
                # Forward (C) and backward (D) quadrature for each frequency
                C[n - 1, i] = delta[i] / (4 * (-N / 2 + n - 1)) * (
                    r_m[i] * (r_m[i] / r_m[i + 1]) ** (-(-N / 2 + n - 1)) * f_fourier_coeff[n - 1, i]
                    + r_m[i + 1] * f_fourier_coeff[n - 1, i + 1]
                )
                D[n, i] = -(delta[i] / (4 * n)) * (
                    r_m[i + 1] * (r_m[i] / r_m[i + 1]) ** n * f_fourier_coeff[n + N // 2, i + 1]
                    + r_m[i] * f_fourier_coeff[n + N // 2, i]
                )
            # Highest frequency mode treated as a special case (no sign flip)
            C[N // 2, i] = delta[i] / 2 * (
                r_m[i] * f_fourier_coeff[N // 2, i]
                + r_m[i + 1] * f_fourier_coeff[N // 2, i + 1]
            )
            # Special n=0 (mean mode): avoid 0*log(0), compute from i > 0 only
            if i != 0:
                D[0, i] = delta[i] / 2 * (
                    r_m[i + 1] * np.log(r_m[i + 1]) * f_fourier_coeff[N // 2, i + 1]
                    + r_m[i] * np.log(r_m[i]) * f_fourier_coeff[N // 2, i]
                )
        # 0*log(0) handled by endpoint formula
        D[0, 0] = delta[0] / 2 * (r_m[1] * np.log(r_m[1]) * f_fourier_coeff[N // 2, 1])

    # --- Simpson's Rule (quad_rule=2) ---
    elif quad_rule == 2:
        for i in range(1, M - 1):
            # Build local three-point stencil for non-uniform Simpson's rule
            r_temp = np.array([r_m[i - 1], r_m[i], r_m[i + 1]])

            for n in range(1, N // 2 + 1):
                # f_temp for C integral, depends on mode and mesh shape
                f_temp = np.array([
                    r_m[i - 1] / (2 * (-N / 2 + n - 1)) * (r_m[i + 1] / r_m[i - 1]) ** (-N / 2 + n - 1) * f_fourier_coeff[n - 1, i - 1],
                    r_m[i]     / (2 * (-N / 2 + n - 1)) * (r_m[i + 1] / r_m[i])     ** (-N / 2 + n - 1) * f_fourier_coeff[n - 1, i],
                    r_m[i + 1] / (2 * (-N / 2 + n - 1)) *                            f_fourier_coeff[n - 1, i + 1]
                ])
                C[n - 1, i] = nonuniform_simps_rule(r_temp, f_temp)

                f_temp = np.array([
                    -r_m[i - 1] / (2 * n) * f_fourier_coeff[n + N // 2, i - 1],
                    -r_m[i]     / (2 * n) * (r_m[i - 1] / r_m[i])     ** n * f_fourier_coeff[n + N // 2, i],
                    -r_m[i + 1] / (2 * n) * (r_m[i - 1] / r_m[i + 1]) ** n * f_fourier_coeff[n + N // 2, i + 1]
                ])
                D[n, i] = nonuniform_simps_rule(r_temp, f_temp)

                # Left endpoint C/D using trapezoidal rule
                if i == 1:
                    C[n - 1, 0] = (delta[0] ** 2 / (4 * (-N / 2 + n - 1))) * f_fourier_coeff[n - 1, 1]
                    D[n, 0] = -(delta[M - 2] / (4 * n)) * (
                        r_m[M - 2] * f_fourier_coeff[n + N // 2, M - 2]
                        + r_m[M - 1] * (r_m[M - 2] / r_m[M - 1]) ** n * f_fourier_coeff[n + N // 2, M - 1]
                    )

            # n=max mode for C
            f_temp = np.array([
                r_m[i - 1] * f_fourier_coeff[N // 2, i - 1],
                r_m[i]     * f_fourier_coeff[N // 2, i],
                r_m[i + 1] * f_fourier_coeff[N // 2, i + 1]
            ])
            C[N // 2, i] = nonuniform_simps_rule(r_temp, f_temp)

            # n=0 mode for D, using log for nonzero radii
            if i != 1:
                f_temp = np.array([
                    r_m[i - 1] * np.log(r_m[i - 1]) * f_fourier_coeff[N // 2, i - 1],
                    r_m[i]     * np.log(r_m[i])     * f_fourier_coeff[N // 2, i],
                    r_m[i + 1] * np.log(r_m[i + 1]) * f_fourier_coeff[N // 2, i + 1]
                ])
                D[0, i] = nonuniform_simps_rule(r_temp, f_temp)

        # Special-case explicit endpoint integrals using trapezoidal rule
        C[N // 2, 0] = (r_m[1] ** 2 / 2) * f_fourier_coeff[N // 2, 1]
        r_temp = np.array([r_m[0], r_m[1], r_m[2]])
        f_temp = np.array([0,
            r_m[1] * np.log(r_m[1]) * f_fourier_coeff[N // 2, 1],
            r_m[2] * np.log(r_m[2]) * f_fourier_coeff[N // 2, 2]])
        D[0, 1] = nonuniform_simps_rule(r_temp, f_temp)
        D[0, 0] = delta[M - 2] / 2 * (
            r_m[M - 2] * np.log(r_m[M - 2]) * f_fourier_coeff[N // 2, M - 2]
            + r_m[M - 1] * np.log(r_m[M - 1]) * f_fourier_coeff[N // 2, M - 1]
        )

    else:
        raise ValueError("quad_rule must be 1 (trapezoidal) or 2 (Simpson).")

    return C, D
