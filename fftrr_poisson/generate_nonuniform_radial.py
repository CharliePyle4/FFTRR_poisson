"""
Radial mesh utilities for the unit disk.

Provides routines to generate non-uniformly or randomly spaced radial coordinates 
on [0, R], for use in disk-based numerical schemes.

Options for mesh clustering are controlled by the 'mapping' parameter.
"""

import numpy as np

def generate_nonuniform_radial(M, R, mapping=None):
    """
    Generate a non-uniform or random radial mesh on the interval [0, R].

    Parameters
    ----------
    M : int
        Number of radial mesh points to generate.
    R : float
        Disk radius (end point of interval).
    mapping : {'sqrt', 'cubic_root', 'atan', 'squared', 'uniform', 'random'}, optional
        String identifier for the desired mapping:
        - 'sqrt' (default):       iRadius = sqrt(R) * sqrt(r)
        - 'cubic_root':           iRadius = R**(2/3) * r**(1/3)
        - 'atan':                 iRadius = R/atan(R) * atan(r)
        - 'squared':              iRadius = r**2 / R
        - 'uniform':              iRadius = r (equally spaced)
        - 'random':               iRadius = sorted random values in [0, R]

    Returns
    -------
    iRadius : ndarray, shape (M,)
        Computed radii between 0 and R.

    Examples
    --------
    >>> generate_nonuniform_radial(5, 1.0, mapping="random")
    array([...])
    >>> generate_nonuniform_radial(5, 1.0, mapping="atan")
    array([...])

    Notes
    -----
    See the `mapping` parameter for available choices and formulas.
    """
    if mapping is None or mapping == "sqrt":
        r = np.linspace(0, R, M)
        iRadius = np.sqrt(R) * np.sqrt(r)
    elif mapping == "cubic_root":
        r = np.linspace(0, R, M)
        iRadius = R ** (2/3) * r ** (1/3)
    elif mapping == "atan":
        r = np.linspace(0, R, M)
        iRadius = R / np.arctan(R) * np.arctan(r)
    elif mapping == "squared":
        r = np.linspace(0, R, M)
        iRadius = r ** 2 / R
    elif mapping == "uniform":
        iRadius = np.linspace(0, R, M)
    elif mapping == "random":
        iRadius = np.sort(np.random.uniform(0, R, M))
    else:
        raise ValueError(f"Unknown mapping: '{mapping}'. Valid options are "
            "'sqrt', 'cubic_root', 'atan', 'squared', 'uniform', 'random'.")
    return iRadius
