import numpy as np
import matplotlib.pyplot as plt

def _try_transforms(A, B):
    """
    Find best alignment of A to B by trying:
    - transpose (False/True)
    - row-reversal (False/True)
    - row shift (0..N-1) on the (theta) row axis after transpose decision
    Returns aligned A and the chosen (transposed, reversed, shift, inf_err).
    """
    cand = []
    # two transpose options
    for do_T in (False, True):
        A1 = A.T if do_T else A
        B1 = B.T if do_T else B
        N = A1.shape[0]
        if B1.shape != A1.shape:
            continue
        for rev in (False, True):
            A2 = A1[::-1, :] if rev else A1
            # scan row shifts
            best = (None, np.inf)
            for s in range(N):
                A3 = np.roll(A2, -s, axis=0)
                err = np.max(np.abs(A3 - B1))
                if err < best[1]:
                    best = (s, err)
            cand.append((do_T, rev, best[0], best[1]))
    # pick best combo
    do_T, rev, shift, best_err = min(cand, key=lambda t: t[3])

    # rebuild aligned A using that combo
    A1 = A.T if do_T else A
    A2 = A1[::-1, :] if rev else A1
    A_aligned = np.roll(A2, -shift, axis=0)

    return A_aligned, (do_T, rev, shift, best_err)

def plot_on_disk_with_error(x_coord: np.ndarray,
                            y_coord: np.ndarray,
                            u_approx: np.ndarray,
                            u_true:   np.ndarray) -> None:
    """
    Auto-align Python's (x,y,u_approx) to u_true (MATLAB) by testing transpose,
    theta-reversal, and row shift. Then plot true, approx, and pointwise L∞ error.
    """

    # 0) Use real(u_approx) for comparison
    uA0 = np.real(u_approx)
    uT0 = np.real(u_true)

    # 1) Find best alignment to minimize sup-norm vs true
    uA_aligned, (did_T, did_rev, row_shift, best_inf) = _try_transforms(uA0, uT0)
    print(f"[align] transpose={did_T}, reverse_theta={did_rev}, row_shift={row_shift}, "
          f"||Δ||_∞ after align = {best_inf:.3e}")

    # 2) Apply the SAME transform to coords
    X = x_coord.T if did_T else x_coord
    Y = y_coord.T if did_T else y_coord
    if did_rev:
        X = X[::-1, :]
        Y = Y[::-1, :]
    X = np.roll(X, -row_shift, axis=0)
    Y = np.roll(Y, -row_shift, axis=0)

    # 3) Also transform u_true if we transposed (to keep shapes consistent)
    Utrue = uT0.T if did_T else uT0

    # 4) Append closure row AFTER alignment
    Xp  = np.vstack([X,      X[0, :]])
    Yp  = np.vstack([Y,      Y[0, :]])
    UA  = np.vstack([uA_aligned, uA_aligned[0, :]])
    UT  = np.vstack([Utrue,     Utrue[0, :]])

    # 5) Pointwise error
    ptwise_error = np.abs(UT - UA)

    # 6) Plots
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot_surface(Xp, Yp, UT, cmap='cool')
    ax1.set_title('True solution')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.plot_surface(Xp, Yp, UA, cmap='cool')
    ax2.set_title('Approximate solution')

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection='3d')
    ax3.plot_surface(Xp, Yp, ptwise_error, cmap='cool')
    ax3.set_title(r'$L^\infty$-Error')

    plt.show()
