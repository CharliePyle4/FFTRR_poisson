import numpy as np

def pol2cart(rho, phi):
    """
    Convert Polar Coordinates to cartesian coordinates

    Parameters
    ----------
    rho : float or ndarray
        Radial coordinate(s).
    phi : float or ndarray
        Angular coordinate(s), in radians.

    Returns
    -------
    x : float or ndarray
        x-coordinate(s) corresponding to the input.
    y : float or ndarray
        y-coordinate(s) corresponding to the input.
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def generate_cartesian_grid_on_disk(iAngle, iRadius):
    """
    Generate Cartesian grid coordinates (x, y) on a disk
    from polar coordinates defined by iAngle and iRadius.

    Parameters
    ----------
    iAngle : ndarray, shape (N,)
        Azimuthal mesh points (angles in radians).
    iRadius : ndarray, shape (M,)
        Radial mesh points.

    Returns
    -------
    x_coord : ndarray, shape (N, M)
        x-coordinates of the grid (rows = angles, cols = radii).
    y_coord : ndarray, shape (N, M)
        y-coordinates of the grid (rows = angles, cols = radii).
    """

    #N describes the number of points azimuthally.
    #M describes the number of points radially.

    N = len(iAngle)
    M = len(iRadius)

    x_coord = np.zeros((N,M))
    y_coord = np.zeros((N,M))
    for j in range(M):
        for k in range(N):
            x_coord[k, j], y_coord[k, j] = pol2cart(iRadius[j], iAngle[k])

    return x_coord, y_coord