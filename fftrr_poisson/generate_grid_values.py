"""
Grid evaluation utility functions for the FFTRR Poisson solver.

This module provides routines to evaluate a user-supplied function
f(x, y) over either a 2D Cartesian grid (mesh) or along
a 1D boundary (e.g., the boundary of the disk). It handles both mesh
arrays (shape (N, M)) and vector inputs (shape (N,)) and returns
values in matching shapes.
"""
import numpy as np

def generate_grid_values(f, x_coord, y_coord):
    """
    Evaluate a function of two variables, f(x, y), on a grid or vector of points.

    Supports both full 2D grid evaluation and 1D boundary evaluation
    (for Dirichlet or Neumann boundary data on a disk).

    Parameters
    ----------
    f : callable
        Function of two arguments, to evaluate at each input pair.
    x_coord : ndarray
        x-coordinates of the evaluation grid or vector.
        Shape can be (N, M) for a 2D grid, or (N,) for a 1D boundary.
    y_coord : ndarray
        y-coordinates of the evaluation grid or vector.
        Shape should match that of x_coord.

    Returns
    -------
    grid_values_f : ndarray
        Function values evaluated at (x_coord, y_coord) points.
        Shape matches input: (N, M) for mesh grids, (N,) for boundary vectors.

    Examples
    --------
    # On a 2D grid:
    >>> x = np.array([[0, 1],[0, 1]])
    >>> y = np.array([[0, 0],[1, 1]])
    >>> generate_grid_values(lambda x, y: x + y, x, y)
    array([[0., 1.],
           [1., 2.]])

    # On a 1D boundary vector:
    >>> x_edge = np.array([1.0, 0.0])
    >>> y_edge = np.array([0.0, 1.0])
    >>> generate_grid_values(np.hypot, x_edge, y_edge)
    array([1., 1.])

    Notes
    -----
    - For 1D inputs (boundary vectors), both x_coord and y_coord must be 1D,
      and of equal length.
    - For 2D mesh inputs, x_coord and y_coord should have the same shape,
      typically (N, M), where N is the number of angular points and
      M is the number of radial points in a polar-to-Cartesian mesh.
    """

    # For 1D boundary-style inputs
    if x_coord.ndim == 1 and y_coord.ndim == 1:
        N = x_coord.shape[0]
        grid_values_f = np.zeros(N)
        for k in range(N):
            grid_values_f[k] = f(x_coord[k], y_coord[k])
        return grid_values_f

    # For 2D grid inputs
    M = x_coord.shape[1]   
    N = x_coord.shape[0]   
    grid_values_f = np.zeros((N, M))
    for j in range(M):
        for k in range(N):
            grid_values_f[k, j] = f(x_coord[k,j], y_coord[k,j])

    return grid_values_f
