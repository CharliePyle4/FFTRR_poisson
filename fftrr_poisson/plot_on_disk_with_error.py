"""
Visualization utilities for comparing numerical and analytical solutions on the disk.

- plot_on_disk_with_error: Aligns and compares an approximate solution to a reference solution, and plots both with pointwise errors.
"""

import numpy as np
import matplotlib.pyplot as plt

def _try_transforms(A, B):
    """
    Find the best alignment of array A to array B by exploring transpose,
    row-reversal, and row shifts along the 0th axis.

    The function tests all combinations of:
     - transpose (True/False)
     - row reversal (True/False) along the theta (angle) dimension
     - all possible row shifts (circular, 0 to N-1)

    The alignment that minimizes the maximum absolute difference (L∞ error)
    is selected.

    Parameters
    ----------
    A : ndarray, shape (N, M)
        Input array to align (the computed approximation).
    B : ndarray, shape (N, M)
        Reference array to align against (the true solution)

    Returns
    -------
    A_aligned : ndarray, shape (N, M)
        Array A, transformed to best align with B.
    details : tuple
        Tuple of (did_transpose, did_reverse, row_shift, min_inf_error) describing the transformation.

    Notes
    -----
    The algorithm is useful for comparing arrays when their orientations,
    angular conventions, or grid indexing might differ (as between MATLAB and Python).
    """
    candidates = []
    for do_T in (False, True):
        A1 = A.T if do_T else A
        B1 = B.T if do_T else B
        N = A1.shape[0]
        if B1.shape != A1.shape:
            continue
        for rev in (False, True):
            A2 = A1[::-1, :] if rev else A1
            best = (None, np.inf)
            for s in range(N):
                A3 = np.roll(A2, -s, axis=0)
                err = np.max(np.abs(A3 - B1))
                if err < best[1]:
                    best = (s, err)
            candidates.append((do_T, rev, best[0], best[1]))
    do_T, rev, shift, best_err = min(candidates, key=lambda t: t[3])

    # Apply best transform to A for output
    A1 = A.T if do_T else A
    A2 = A1[::-1, :] if rev else A1
    A_aligned = np.roll(A2, -shift, axis=0)
    return A_aligned, (do_T, rev, shift, best_err)

def plot_on_disk_with_error(
    x_coord: np.ndarray,
    y_coord: np.ndarray,
    u_approx: np.ndarray,
    u_true: np.ndarray
) -> None:
    """
    Align an approximate solution to a reference solution,
    then plot the true solution, the computed approximation, and the pointwise L∞ error.

    Alignment is performed by searching all transpose, theta-reverse, and row-shift
    combinations to minimize max-norm error, then *the same transformation* is
    applied to the coordinates for accurate 3D plotting.

    Parameters
    ----------
    x_coord : ndarray, shape (N, M)
        x-coordinates of the mesh grid.
    y_coord : ndarray, shape (N, M)
        y-coordinates of the mesh grid.
    u_approx : ndarray, shape (N, M)
        Approximate numerical solution (may be complex; real part used).
    u_true : ndarray, shape (N, M)
        Reference solution for comparison.

    Returns
    -------
    None

    Notes
    -----
    - Plots three 3D surfaces using matplotlib: true solution, approx solution, and error.
    - Useful for debugging index, orientation, or periodicity mismatches between
      computed and reference data.
    """
    # Take real parts for comparison
    uA0 = np.real(u_approx)
    uT0 = np.real(u_true)

    # Find minimal alignment between computed and true solution
    uA_aligned, (did_T, did_rev, row_shift, best_inf) = _try_transforms(uA0, uT0)
    print(f"[align] transpose={did_T}, reverse_theta={did_rev}, row_shift={row_shift}, "
          f"||Δ||_∞ after align = {best_inf:.3e}")

    # Apply same geometric transformation to coordinates
    X = x_coord.T if did_T else x_coord
    Y = y_coord.T if did_T else y_coord
    if did_rev:
        X = X[::-1, :]
        Y = Y[::-1, :]
    X = np.roll(X, -row_shift, axis=0)
    Y = np.roll(Y, -row_shift, axis=0)

    # Align u_true to coords if needed
    Utrue = uT0.T if did_T else uT0

    # For plotting, append a closure row (handles wrap-around for periodic data)
    Xp  = np.vstack([X,     X[0, :]])
    Yp  = np.vstack([Y,     Y[0, :]])
    UA  = np.vstack([uA_aligned, uA_aligned[0, :]])
    UT  = np.vstack([Utrue,      Utrue[0, :]])

    # Compute pointwise absolute error
    ptwise_error = np.abs(UT - UA)

    # Plot the true solution
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot_surface(Xp, Yp, UT, cmap='cool')
    ax1.set_title('True solution')

    # Plot the approximate solution
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(Xp, Yp, UA, cmap='cool')
    ax2.set_title('Approximate solution')

    # Plot the absolute error
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.plot_surface(Xp, Yp, ptwise_error, cmap='cool')
    ax3.set_title(r'$L^\infty$-Error')

    plt.show()
