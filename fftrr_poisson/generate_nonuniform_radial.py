import numpy as np

def generate_nonuniform_radial(M, R):
    """
    Generate a nonuniform radial mesh on [0, R].

    Parameters
    ----------
    M : int
        Number of radial points.
    R : float
        Radius of the disk.

    Returns
    -------
    iRadius : ndarray, shape (M,)
        Nonuniformly spaced radii between 0 and R.
    """

    r = np.linspace(0, R, M)
    iRadius = np.sqrt(R)*np.sqrt(r)


    # iRadius = R^(2/3) * r.^(1/3)
    # iRadius = R^(-1/5) * r.^(1/5 + 1)
    # iRadius = R/atan(R) * atan(r)
    # iRadius = r.^2 / R
    # iRadius = r

    return iRadius
