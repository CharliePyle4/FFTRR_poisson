import numpy as np

def generate_grid_values(f, x_coord, y_coord):
    """
    Evaluate a function f(x, y) on a polar grid converted to Cartesian coordinates.

    Parameters
    ----------
    f : callable
        Function of two variables (x, y).
    x_coord : ndarray
        x-coordinates of the grid. Can be shape (N, M) or (N,).
    y_coord : ndarray
        y-coordinates of the grid. Can be shape (N, M) or (N,).

    Returns
    -------
    grid_values_f : ndarray
        Values of f evaluated at each grid point.
        Shape will match input: (N, M) for grids, (N,) for boundary vectors.
    """

    #had to add this case for 1-d inputs to match up with matlab
    if x_coord.ndim == 1 and y_coord.ndim == 1:
        N = x_coord.shape[0]
        grid_values_f = np.zeros(N)
        for k in range(N):
            grid_values_f[k] = f(x_coord[k], y_coord[k])
        return grid_values_f


    M = x_coord.shape[1]   
    N = x_coord.shape[0]   
    
    grid_values_f = np.zeros((N, M))
    for j in range(M):
        for k in range(N):
            grid_values_f[k, j] = f(x_coord[k,j], y_coord[k,j])

    return grid_values_f
