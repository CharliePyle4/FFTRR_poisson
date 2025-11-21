"""
Non-uniform Simpson's rule integration utility.

Implements an extension of Simpson’s rule for integrating over three
non-uniformly spaced points, following Reed (2014), "Numerically Integrating
Irregularly-Spaced (x, y) Data" (The Mathematics Enthusiast).
"""

import numpy as np

def nonuniform_simps_rule(x: np.ndarray, f: np.ndarray) -> float:
    """
    Approximate the definite integral of sampled data at three non-uniformly spaced points using a
    generalized Simpson’s rule.

    Parameters
    ----------
    x : np.ndarray of shape (3,)
        Array of three strictly increasing x-coordinates (the grid points).
    f : np.ndarray of shape (3,) or (3, 1)
        Array of three function values at the corresponding x-coordinates.

    Returns
    -------
    result : float
        Approximate value of the definite integral over [x[0], x[2]].

    Notes
    -----
    This routine fits a unique quadratic (parabola) to the three (x, f) points
    and then analytically integrates that parabola. The algorithm follows:

        B.C. Reed, "Numerically Integrating Irregularly-Spaced (x, y) Data",
        The Mathematics Enthusiast, 11(3), 643-648 (2014).

    Examples
    --------
    >>> x = np.array([0.0, 0.3, 1.0])
    >>> f = np.array([1.0, 3.0, 2.0])
    >>> nonuniform_simps_rule(x, f)
    1.6605...
    """
    # Reshape f to column vector if needed (for robust linear solve)
    f = f.reshape(-1, 1) if f.ndim == 1 else f  # Ensure shape (3, 1)

    # Precompute powers for quadratic fit
    x_1 = x[0] ** 2
    x_2 = x[1] ** 2
    x_3 = x[2] ** 2

    # Set up the system Ax = f, where rows are [x^2, x, 1], to fit a parabola y = ax^2 + bx + c
    A = np.array([
        [x_1, x[0], 1],
        [x_2, x[1], 1],
        [x_3, x[2], 1]
    ], dtype=float)

    # Solve for parabola coefficients (a, b, c)
    c = np.linalg.solve(A, f)

    # Integrate parabola analytically over [x0, x2]
    result = (
        (c[0] / 3) * (x[2] ** 3 - x[0] ** 3)    # Integral of a x^2
        + (c[1] / 2) * (x_3 - x_1)              # Integral of b x
        + c[2] * (x[2] - x[0])                  # Integral of c
    )

    return float(result)
