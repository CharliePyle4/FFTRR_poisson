import numpy as np

def nonuniform_simps_rule(x: np.ndarray, f: np.ndarray) -> float:
    """
    Parameters
    ----------
    x : np.ndarray
        Array of three x-values (non-uniform grid points).
    f : np.ndarray
        Array of three function values corresponding to x.

    Returns
    -------
    result : float
        Approximation to the integral using the nonuniform Simpson’s rule.
    """

    # We use the method outlined in the paper "Numerically Integrating
    # Irregularly-spaced (x,y) Data" by B. Cameron Reed in /The Mathematics
    # Enthusiast/.

    # Make sure 'f' is a column vector.
    f = f.reshape(-1, 1) if f.ndim == 1 else f

    # Save some computational time.
    x_1 = x[0] ** 2
    x_2 = x[1] ** 2
    x_3 = x[2] ** 2

    # Compute matrix for finding coefficients.
    A = np.array([
        [x_1, x[0], 1],
        [x_2, x[1], 1],
        [x_3, x[2], 1]
    ], dtype=float)

    # Solve linear system for coefficients
    c = np.linalg.solve(A, f)

    result = (
        c[0] / 3 * (x[2] ** 3 - x[0] ** 3)
        + c[1] / 2 * (x_3 - x_1)
        + c[2] * (x[2] - x[0])
    )

    return float(result)
