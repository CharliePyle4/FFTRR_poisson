import numpy as np
import matplotlib.pyplot as plt

def plot_on_disk(x_coord: np.ndarray, y_coord: np.ndarray, u: np.ndarray) -> None:
    """
    Plot a solution u(x,y) on a disk in 3D.

    Parameters
    ----------
    x_coord : (N, M) array
        X-coordinates of the polar grid mapped to Cartesian.
    y_coord : (N, M) array
        Y-coordinates of the polar grid mapped to Cartesian.
    u : (N, M) array
        Values of the solution on the grid.
    """
    # Force data to be real for plotting stability
    u = np.real(u)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x_coord, y_coord, u, cmap="cool")
    ax.set_title("Solution on Disk")
    plt.show()
