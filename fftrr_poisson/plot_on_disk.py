"""
Basic visualization utility for plotting 2D disk mesh solutions in 3D.

Use this module to visualize computed or analytic solutions as surfaces on
the (x, y) domain of the disk.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_on_disk(x_coord: np.ndarray, y_coord: np.ndarray, u: np.ndarray) -> None:
    """
    Plot a 2D solution u(x, y) defined on a polar grid mapped to cartesian coordinates as a
    3D surface plot.

    Parameters
    ----------
    x_coord : ndarray of shape (N, M)
        X-coordinates of the mesh grid (typically from polar-to-Cartesian mapping).
    y_coord : ndarray of shape (N, M)
        Y-coordinates of the mesh grid (same shape as x_coord).
    u : ndarray of shape (N, M)
        Values of the function or solution at each grid point.
        Can be complex; only the real part is plotted.

    Returns
    -------
    None

    Examples
    --------
    >>> plot_on_disk(x_coord, y_coord, u)

    Notes
    -----
    - This function is intended for quick 3D visualization of a solution on the disk.
    - If `u` is complex, only the real part is displayed.
    - The color map is set to "cool" by default.
    """
    # Force data to be real for plotting stability
    u = np.real(u)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x_coord, y_coord, u, cmap="cool")
    ax.set_title("Solution on Disk")
    plt.show()
