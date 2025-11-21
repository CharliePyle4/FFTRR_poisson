"""
Radial integration utility for spectral-Poisson algorithms on the disk. Used in
the FFTRR method for solving the Poisson equation on the unit disk.

Implements the computation of C and D integrals (matrices) using either
the trapezoidal or Simpson's rule in the radial direction
"""
import numpy as np

def compute_C_D_uniform(r_m: np.ndarray, f_fourier_coeff: np.ndarray, quad_rule: int):
    """
    Compute the C and D coefficient matrices for each angular Fourier mode
    using quadrature along a uniform radial mesh.

    Parameters
    ----------
    r_m : ndarray of shape (M,)
        Array of radial mesh points (must be uniformly spaced).
    f_fourier_coeff : ndarray of shape (N+1, M)
        Discrete Fourier coefficients of the source function, for each radius.
        Indexed as [mode, radius], where mode is from -N/2 to N/2.
    quad_rule : int
        Which quadrature rule to use:
            1 : Trapezoidal Rule
            2 : Simpson's Rule

    Returns
    -------
    C : ndarray of shape (N//2 + 1, M-1)
        C coefficient matrix (for positive frequencies).
    D : ndarray of shape (N//2 + 1, M-1)
        D coefficient matrix (for positive frequencies).

    Notes
    -----
    - Supports both trapezoidal (quad_rule=1) and Simpson's (quad_rule=2) quadrature.
    - Output matrices C, D may be complex, depending on the input.
    - Indexing starts at 1 in the mathematical/matlab algorithm, so pay attention to Python offsets.
    - For the n=0 mode (mean), logarithmic weights are used to avoid 0*log(0) errors.

    References
    ----------
    See: L. Borges and P. Daripa, "A Fast Parallel Algorithm for the Poisson Equation on a Disk," J Comput Phys 169(1):151–192, 2001.

    """
    M = len(r_m)
    N = f_fourier_coeff.shape[0] - 1

    # Output coefficient matrices (each frequency, each radial shell)
    C = np.zeros((N // 2 + 1, M - 1), dtype=complex)
    D = np.zeros((N // 2 + 1, M - 1), dtype=complex)

    # Uniform mesh spacing
    delta = r_m[1] - r_m[0]

    # Much of the index notation below is based around the
    # fact that the array indexing is always positive, whereas
    # the mathematics uses negative indexing.

    # TRAPEZOIDAL RULE
    if quad_rule == 1:
        for i in range(1, M):  # Loop over each radial interval
            for n in range(1, N // 2 + 1):  # Positive frequency modes
                # C: forward
                C[n-1, i-1] = (delta**2 / (4 * (-N/2 + n - 1))) * (
                    (i-1) * ((i-1)/i)**(-(-N/2 + n - 1)) * f_fourier_coeff[n-1, i-1]
                    + i * f_fourier_coeff[n-1, i]
                )
                # D: backward
                D[n, i-1] = -(delta**2 / (4 * n)) * (
                    i * ((i-1)/i)**n * f_fourier_coeff[n+N//2, i]
                    + (i-1) * f_fourier_coeff[n+N//2, i-1]
                )
            # n = N//2 case (highest positive frequency)
            C[N//2, i-1] = (delta**2 / 2) * (
                (i-1) * f_fourier_coeff[N//2, i-1]
                + i * f_fourier_coeff[N//2, i]
            )
            # n = 0 mode: special log term to avoid 0*log(0)
            if i != 1:
                D[0, i-1] = (delta**2 / 2) * (
                    i * np.log(i * delta) * f_fourier_coeff[N//2, i]
                    + (i-1) * np.log((i-1) * delta) * f_fourier_coeff[N//2, i-1]
                )
        # Explicitly handle the 0*log(0) endpoint
        D[0, 0] = (delta**2 / 2) * (
            np.log(delta) * f_fourier_coeff[N//2, 1]
        )

    # SIMPSON'S RULE
    elif quad_rule == 2:
        for i in range(2, M):  # Loop for Simpson's quadrature
            for n in range(1, N // 2 + 1):
                C[n-1, i-1] = (delta**2 / (6 * (-N/2 + n - 1))) * (
                    (i-2) * ((i-2)/i)**(-(-N/2 + n - 1)) * f_fourier_coeff[n-1, i-2]
                    + 4 * (i-1) * ((i-1)/i)**(-(-N/2 + n - 1)) * f_fourier_coeff[n-1, i-1]
                    + i * f_fourier_coeff[n-1, i]
                )
                D[n, i-1] = -(delta**2 / (6 * n)) * (
                    (i-2) * f_fourier_coeff[n+N//2, i-2]
                    + 4*(i-1) * ((i-2)/(i-1))**n * f_fourier_coeff[n+N//2, i-1]
                    + i * ((i-2)/i)**n * f_fourier_coeff[n+N//2, i]
                )
                if i == 2:  # Special: left endpoint coefficients only once
                    C[n-1, 0] = (delta**2 / (4 * (-N/2 + n - 1))) * f_fourier_coeff[n-1, 1]
                    D[n, 0] = -(delta**2 / (4 * n)) * (
                        (M-1) * ((M-2)/(M-1))**n * f_fourier_coeff[n+N//2, M-1]
                        + (M-2) * f_fourier_coeff[n+N//2, M-2]
                    )
            C[N//2, i-1] = (delta**2 / 3) * (
                (i-2) * f_fourier_coeff[N//2, i-2]
                + 4*(i-1) * f_fourier_coeff[N//2, i-1]
                + i * f_fourier_coeff[N//2, i]
            )
            # n=0 mode w/ logs, avoid endpoints as above
            if i != 2:
                D[0, i-1] = (delta**2 / 3) * (
                    (i-2) * np.log((i-2)*delta) * f_fourier_coeff[N//2, i-2]
                    + 4*(i-1) * np.log((i-1)*delta) * f_fourier_coeff[N//2, i-1]
                    + i * np.log(i*delta) * f_fourier_coeff[N//2, i]
                )
        # Special endpoint cases for C, D (log terms etc)
        C[N//2, 0] = (delta**2 / 2) * f_fourier_coeff[N//2, 1]
        D[0, 1] = (delta**2 / 3) * (
            4*np.log(delta) * f_fourier_coeff[N//2, 1]
            + 2*np.log(2*delta) * f_fourier_coeff[N//2, 2]
        )
        D[0, 0] = (delta**2 / 2) * (
            (M-2)*np.log((M-2)*delta) * f_fourier_coeff[N//2, M-2]
            + (M-1)*np.log((M-1)*delta) * f_fourier_coeff[N//2, M-1]
        )

    else:
        raise ValueError("Unknown quad_rule; must be 1 (trapezoidal) or 2 (Simpson).")

    return C, D
