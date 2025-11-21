import numpy as np

def compute_C_D_uniform(r_m: np.ndarray, f_fourier_coeff: np.ndarray, quad_rule: int):
    """
    Compute the integrals C and D for a nonuniform mesh in the radial
    direction, using either Simpson's rule or trapezoidal rule.

    Parameters
    ----------
    r_m : np.ndarray
        Radial mesh points
    f_fourier_coeff : np.ndarray
        Fourier coefficients (N+1, M)
    quad_rule : int
        1 = Trapezoidal Rule, 2 = Simpson's Rule

    Returns
    -------
    C, D : np.ndarray
        Matrices of coefficients
    """
    M = len(r_m)
    N = f_fourier_coeff.shape[0] - 1

    # Declare matrices
    C = np.zeros((N // 2 + 1, M - 1), dtype=complex)
    D = np.zeros((N // 2 + 1, M - 1), dtype=complex)

    # Create the individual mesh widths.
    delta = r_m[1] - r_m[0]

    # Much of the index notation below is based around the
    # fact that the array indexing is always positive, whereas
    # the mathematics uses negative indexing.

    # Trapezoidal Rule %
    #------------------%
    if quad_rule == 1:
        for i in range(1, M):  # MATLAB 1:M-1 -> Python 1..M-1
            for n in range(1, N // 2 + 1):
                C[n-1, i-1] = (delta**2 / (4 * (-N/2 + n - 1))) * (
                    (i-1) * ((i-1)/i)**(-(-N/2 + n - 1)) * f_fourier_coeff[n-1, i-1]
                    + i * f_fourier_coeff[n-1, i]
                )
                D[n, i-1] = -(delta**2 / (4 * n)) * (
                    i * ((i-1)/i)**n * f_fourier_coeff[n+N//2, i]
                    + (i-1) * f_fourier_coeff[n+N//2, i-1]
                )
            C[N//2, i-1] = (delta**2 / 2) * (
                (i-1) * f_fourier_coeff[N//2, i-1]
                + i * f_fourier_coeff[N//2, i]
            )
            # We must compute D(1,1) separately, since there is a computation of
            # 0*log(0) in the algorithm.
            if i != 1:
                D[0, i-1] = (delta**2 / 2) * (
                    i * np.log(i * delta) * f_fourier_coeff[N//2, i]
                    + (i-1) * np.log((i-1) * delta) * f_fourier_coeff[N//2, i-1]
                )
        D[0, 0] = (delta**2 / 2) * (
            np.log(delta) * f_fourier_coeff[N//2, 1]
        )

    # With the Simpson's rule, we alot the first column of C 
    # and D to contain the values of C^(1,2) and D^(M-1,M).
    # 
    # Simpson's Rule %
    #----------------%
    elif quad_rule == 2:
        for i in range(2, M):  # MATLAB 2:M-1 -> Python 2..M-1
            for n in range(1, N // 2 + 1):
                C[n-1, i-1] = (delta**2 / (6 * (-N/2 + n - 1))) * (
                    (i-2) * ((i-2)/i)**(-(-N/2 + n - 1)) * f_fourier_coeff[n-1, i-2]
                    + 4*(i-1) * ((i-1)/i)**(-(-N/2 + n - 1)) * f_fourier_coeff[n-1, i-1]
                    + i * f_fourier_coeff[n-1, i]
                )
                D[n, i-1] = -(delta**2 / (6 * n)) * (
                    (i-2) * f_fourier_coeff[n+N//2, i-2]
                    + 4*(i-1) * ((i-2)/(i-1))**n * f_fourier_coeff[n+N//2, i-1]
                    + i * ((i-2)/i)**n * f_fourier_coeff[n+N//2, i]
                )
                if i == 2:  # Compute C^(1,2)_n and D^(M-1,M)_n
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
            # We must compute D(1,2) separately, since there is a computation of
            # 0*log(0) in the algorithm. That is why we exclude 'i=2'.
            if i != 2:
                D[0, i-1] = (delta**2 / 3) * (
                    (i-2) * np.log((i-2)*delta) * f_fourier_coeff[N//2, i-2]
                    + 4*(i-1) * np.log((i-1)*delta) * f_fourier_coeff[N//2, i-1]
                    + i * np.log(i*delta) * f_fourier_coeff[N//2, i]
                )
        # We must compute several more integrals separately since they are a 
        # bit different than those in the 'for loop' above.
        C[N//2, 0] = (delta**2 / 2) * f_fourier_coeff[N//2, 1]
        D[0, 1] = (delta**2 / 3) * (
            4*np.log(delta) * f_fourier_coeff[N//2, 1]
            + 2*np.log(2*delta) * f_fourier_coeff[N//2, 2]
        )
        D[0, 0] = (delta**2 / 2) * (
            (M-2)*np.log((M-2)*delta) * f_fourier_coeff[N//2, M-2]
            + (M-1)*np.log((M-1)*delta) * f_fourier_coeff[N//2, M-1]
        )

    return C, D
