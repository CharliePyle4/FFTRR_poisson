"""
Cartesian grid generation for the unit disk.

This module provides routines for converting from polar (radius, angle)
to Cartesian (x, y) coordinates, as well as assembling 2D coordinate
meshes for use in numerical solutions on the unit disk.

Functions
---------
pol2cart : Convert (rho, phi) polar coordinates to (x, y) Cartesian.
generate_cartesian_grid_on_disk : Build Cartesian grid arrays from
                                  specified radial and angular coordinates.
"""
import numpy as np

def pol2cart(rho, phi):
    """
    Convert polar coordinates (rho, phi) to Cartesian coordinates (x, y).

    Parameters
    ----------
    rho : float or ndarray of shape (M,) or shape matching phi
        Radial coordinate(s). Can be a scalar or array.
    phi : float or ndarray of shape (N,) or shape matching rho
        Angular coordinate(s), in radians. Can be a scalar or array.

    Returns
    -------
    x : float or ndarray
        x-coordinate(s) corresponding to input (same shape as broadcasted inputs).
    y : float or ndarray
        y-coordinate(s) corresponding to input (same shape as broadcasted inputs).

    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def generate_cartesian_grid_on_disk(iAngle, iRadius):
    """
    Generate Cartesian (x, y) grid coordinates for the unit disk based on arrays of angles and radii.

    Parameters
    ----------
    iAngle : ndarray of shape (N,)
        Array of azimuthal angle mesh points, in radians.
    iRadius : ndarray of shape (M,)
        Array of radial mesh points (from 0 out to disk radius).

    Returns
    -------
    x_coord : ndarray of shape (N, M)
        x-coordinates at each (angle, radius) grid point,
        where `x_coord[k, j]` corresponds to iAngle[k], iRadius[j].
    y_coord : ndarray of shape (N, M)
        y-coordinates at each (angle, radius) grid point,
        where `y_coord[k, j]` corresponds to iAngle[k], iRadius[j].

    Examples
    --------
    >>> angles = np.linspace(0, 2*np.pi, 5)
    >>> radii = np.linspace(0, 1, 3)
    >>> x, y = generate_cartesian_grid_on_disk(angles, radii)
    >>> x.shape, y.shape
    ((5, 3), (5, 3))
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