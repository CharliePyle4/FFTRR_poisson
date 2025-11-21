import numpy as np
from .nonuniform_simps_rule import nonuniform_simps_rule

def compute_C_D_nonuniform(r_m: np.ndarray, f_fourier_coeff: np.ndarray, quad_rule: int):
    """
    Compute the integrals C and D for a nonuniform mesh in the radial
    direction, using either Simpson's rule or trapezoidal rule.

    Parameters
    ----------
    r_m : np.ndarray
        Radial mesh points
    f_fourier_coeff : np.ndarray
        Fourier coefficients
    quad_rule : int
        1 = Trapezoidal Rule, 2 = Simpson's Rule

    Returns
    -------
    C, D : np.ndarray
        Coefficient matrices
    """
    M = len(r_m)
    N = f_fourier_coeff.shape[0] - 1

    # Declare matrices
    C = np.zeros((N // 2 + 1, M - 1), dtype=complex)
    D = np.zeros((N // 2 + 1, M - 1), dtype=complex)

    # Create the individual mesh widths.
    delta = np.zeros(M - 1)
    for i in range(M - 1):
        delta[i] = r_m[i + 1] - r_m[i]

    # Much of the index notation below is based around the
    # fact that the array indexing is always positive, whereas
    # the mathematics uses negative indexing.

    # Trapezoidal Rule %
    #------------------%
    if quad_rule == 1:
        for i in range(M - 1):  # MATLAB 1:M-1 -> Python 0..M-2
            for n in range(1, N // 2 + 1):
                C[n - 1, i] = delta[i] / (4 * (-N / 2 + n - 1)) * (
                    r_m[i] * (r_m[i] / r_m[i + 1]) ** (-(-N / 2 + n - 1)) * f_fourier_coeff[n - 1, i]
                    + r_m[i + 1] * f_fourier_coeff[n - 1, i + 1]
                )
                D[n, i] = -(delta[i] / (4 * n)) * (
                    r_m[i + 1] * (r_m[i] / r_m[i + 1]) ** n * f_fourier_coeff[n + N // 2, i + 1]
                    + r_m[i] * f_fourier_coeff[n + N // 2, i]
                )
            C[N // 2, i] = delta[i] / 2 * (
                r_m[i] * f_fourier_coeff[N // 2, i]
                + r_m[i + 1] * f_fourier_coeff[N // 2, i + 1]
            )

            # We must compute D(1,1) separately, since there is a computation of
            # 0*log(0) in the algorithm. That is why we exclude 'i=1' from the
            # computation.
            if i != 0:
                D[0, i] = delta[i] / 2 * (
                    r_m[i + 1] * np.log(r_m[i + 1]) * f_fourier_coeff[N // 2, i + 1]
                    + r_m[i] * np.log(r_m[i]) * f_fourier_coeff[N // 2, i]
                )
        D[0, 0] = delta[0] / 2 * (r_m[1] * np.log(r_m[1]) * f_fourier_coeff[N // 2, 1])

    # With the Simpson's rule, we alot the first column of C 
    # and D to contain the values of C^(1,2) and D^(M-1,M).
    #
    # Simpson's Rule %
    #----------------%
    elif quad_rule == 2:
        for i in range(1, M - 1):  # MATLAB 2:M-1 -> Python 1..M-2
            # Need to use a nonuniform Simpson's rule.
            r_temp = np.array([r_m[i - 1], r_m[i], r_m[i + 1]])

            for n in range(1, N // 2 + 1):
                f_temp = np.array([
                    r_m[i - 1] / (2 * (-N / 2 + n - 1)) * (r_m[i + 1] / r_m[i - 1]) ** (-N / 2 + n - 1) * f_fourier_coeff[n - 1, i - 1],
                    r_m[i] / (2 * (-N / 2 + n - 1)) * (r_m[i + 1] / r_m[i]) ** (-N / 2 + n - 1) * f_fourier_coeff[n - 1, i],
                    r_m[i + 1] / (2 * (-N / 2 + n - 1)) * f_fourier_coeff[n - 1, i + 1]
                ])
                C[n - 1, i] = nonuniform_simps_rule(r_temp, f_temp)

                f_temp = np.array([
                    -r_m[i - 1] / (2 * n) * f_fourier_coeff[n + N // 2, i - 1],
                    -r_m[i] / (2 * n) * (r_m[i - 1] / r_m[i]) ** n * f_fourier_coeff[n + N // 2, i],
                    -r_m[i + 1] / (2 * n) * (r_m[i - 1] / r_m[i + 1]) ** n * f_fourier_coeff[n + N // 2, i + 1]
                ])
                D[n, i] = nonuniform_simps_rule(r_temp, f_temp)

                if i == 1:  # Compute C^(1,2)_n and D^(M-1,M)_n using trapezoidal rule.
                    C[n - 1, 0] = (delta[0] ** 2 / (4 * (-N / 2 + n - 1))) * f_fourier_coeff[n - 1, 1]
                    D[n, 0] = -(delta[M - 2] / (4 * n)) * (
                        r_m[M - 2] * f_fourier_coeff[n + N // 2, M - 2]
                        + r_m[M - 1] * (r_m[M - 2] / r_m[M - 1]) ** n * f_fourier_coeff[n + N // 2, M - 1]
                    )

            f_temp = np.array([
                r_m[i - 1] * f_fourier_coeff[N // 2, i - 1],
                r_m[i] * f_fourier_coeff[N // 2, i],
                r_m[i + 1] * f_fourier_coeff[N // 2, i + 1]
            ])
            C[N // 2, i] = nonuniform_simps_rule(r_temp, f_temp)

            # We must compute D(1,2) separately, since there is a computation of
            # 0*log(0) in the algorithm.
            if i != 1:
                f_temp = np.array([
                    r_m[i - 1] * np.log(r_m[i - 1]) * f_fourier_coeff[N // 2, i - 1],
                    r_m[i] * np.log(r_m[i]) * f_fourier_coeff[N // 2, i],
                    r_m[i + 1] * np.log(r_m[i + 1]) * f_fourier_coeff[N // 2, i + 1]
                ])
                D[0, i] = nonuniform_simps_rule(r_temp, f_temp)

        # We must compute several more integrals separately since they are a 
        # bit different than those in the 'for loop' above.
        C[N // 2, 0] = (r_m[1] ** 2 / 2) * f_fourier_coeff[N // 2, 1]  # Trapezoidal rule.
        r_temp = np.array([r_m[0], r_m[1], r_m[2]])
        f_temp = np.array([0,
                          r_m[1] * np.log(r_m[1]) * f_fourier_coeff[N // 2, 1],
                          r_m[2] * np.log(r_m[2]) * f_fourier_coeff[N // 2, 2]])
        D[0, 1] = nonuniform_simps_rule(r_temp, f_temp)

        D[0, 0] = delta[M - 2] / 2 * (
            r_m[M - 2] * np.log(r_m[M - 2]) * f_fourier_coeff[N // 2, M - 2]
            + r_m[M - 1] * np.log(r_m[M - 1]) * f_fourier_coeff[N // 2, M - 1]
        )  # Trapezoidal rule.

    return C, D
