"""
Disk Poisson solver using FFTRR method

This module provides a fast solver for the Poisson equation Δu=f
on the unit disk, supporting both Dirichlet and Neumann boundary
conditions.
"""
import numpy as np

# Import all required subroutines explicitly for clarity
from .generate_grid_values import generate_grid_values
from .generate_nonuniform_radial import generate_nonuniform_radial
from .generate_cartesian_grid_on_disk import generate_cartesian_grid_on_disk
from .compute_fourier_coeff_unif import compute_fourier_coeff_unif
from .compute_C_D_uniform import compute_C_D_uniform
from .compute_C_D_nonuniform import compute_C_D_nonuniform
from .nonuniform_simps_rule import nonuniform_simps_rule
from .trap_2d_on_disk import trap_2d_on_disk
from .plot_on_disk_with_error import plot_on_disk_with_error
from .plot_on_disk import plot_on_disk





def poisson_solver(
    f_values,
    g_values,
    u_fourier_0,
    N,
    M,
    r_m,
    theta_j,
    R,
    quad_rule,
    BC_choice,
    rad_unif
):
    """
    Solve the Poisson equation Δu = f in a disk using a hybrid Fourier-radial method.

    This routine implements the FFTRR algorithm for the disk
    with uniform or non-uniform radial meshes, supporting
    Dirichlet and Neumann boundary conditions.

    Parameters
    ----------
    f_values : ndarray of shape (N, M)
        Samples of the source term f(θ, r) on the polar grid (N angles × M radii).
    g_values : ndarray of shape (N,) or (N,1)
        Boundary data g(θ) at the disk's boundary (r=R).
    u_fourier_0 : ndarray of shape (M,) or (1,M)
        k=0 Fourier mode per radius (for Neumann BC; unused for Dirichlet).
    N : int
        Number of angular grid points (must be even).
    M : int
        Number of radial grid points.
    r_m : ndarray of shape (M,)
        Radial mesh, strictly increasing from 0 to R.
    theta_j : ndarray of shape (N,)
        Angular mesh points (radians; typically on [0, 2π)).
    R : float
        Disk radius.
    quad_rule : int
        Radial quadrature rule: 1 = trapezoidal, 2 = Simpson's.
    BC_choice : int
        1 = Dirichlet boundary conditions, 2 = Neumann boundary conditions.
    rad_unif : int
        Mesh flag: 1 = uniform radial mesh, 0 = non-uniform.

    Returns
    -------
    u_approx : ndarray of shape (N, M)
        Complex-valued approximate solution u(θ, r) on the polar mesh.

    Notes
    -----
    - The algorithm first expands f and boundary data in Fourier series,
      computes C, D integrals along radii, and solves for all Fourier modes.
    - Handles both uniform and non-uniform radial meshes via modular code.
    - Output shape matches input grid as (angle, radius).
    - For Neumann BC, the k=0 (DC) Fourier mode is handled specially via u_fourier_0.

    References
    ----------
    Borges, L. and Daripa, P. "A Fast Parallel Algorithm for the Poisson Equation on a Disk," J. Comput. Phys. 169(1):151–192, 2001.

    Examples
    --------
    >>> u = poisson_solver(f_grid, g_vec, None, N, M, r_m, theta_j, R, 1, 1, 1)
    >>> u.shape
    (N, M)
    """


    # --- 1. Compute Fourier coefficients (in angle θ) of f and BC data ---
    f_fourier_coeff = compute_fourier_coeff_unif(f_values)
    g_fourier_coeff = compute_fourier_coeff_unif(g_values) # Boundary Fourier coefficients.
    


    # --- 2. Compute radial integration matrices ---
    if(rad_unif == 1):
        C, D = compute_C_D_uniform(r_m, f_fourier_coeff, quad_rule)
    elif(rad_unif == 0):
        C, D = compute_C_D_nonuniform(r_m, f_fourier_coeff, quad_rule)
    else:
        raise ValueError('Incorrect index for "rad_unif"')


    # --- 3. Run recurrence for v^- (from center out) and v^+ (from edge in) ---
    v_neg = np.zeros((N // 2 + 1, M), dtype=complex)
    v_pos = np.zeros((N // 2 + 1, M), dtype=complex)


    if quad_rule == 1:
        # Trapezoidal rule
        for r in range(N // 2 + 1):
            v_neg[r, 1] = C[r, 0]
        for i in range(2, M):
            for r in range(N // 2 + 1):
                exp_neg = r - (N // 2)
                v_neg[r, i] = (r_m[i] / r_m[i - 1]) ** exp_neg * v_neg[r, i - 1] + C[r, i - 1]
        for i in range(M - 2, -1, -1):
            for r in range(N // 2 + 1):
                exp_pos = r
                v_pos[r, i] = (r_m[i] / r_m[i + 1]) ** exp_pos * v_pos[r, i + 1] + D[r, i]
    elif quad_rule == 2:
        # Simpson's rule
        for i in range(2, M):
            for r in range(N // 2 + 1):
                if i == 2:
                    v_neg[r, 1] = C[r, 0]
                    continue
                exp_neg = r - (N // 2)
                v_neg[r, i] = (r_m[i] / r_m[i - 2]) ** exp_neg * v_neg[r, i - 2] + C[r, i - 1]
        for i in range(M - 3, -1, -1):
            for r in range(N // 2 + 1):
                if i == M - 3:
                    v_pos[r, M - 2] = D[r, 0]
                exp_pos = r
                v_pos[r, i] = (r_m[i] / r_m[i + 2]) ** exp_pos * v_pos[r, i + 2] + D[r, i + 1]


    # --- 5. Adjust for boundary conditions per Fourier mode ---
    u_fourier_coeff = np.zeros((N + 1, M), dtype=complex)
    for i in range(M):
        if BC_choice == 1:
            u_fourier_coeff[N // 2, i] = v[N // 2, i] + (g_fourier_coeff[N // 2] - v[N // 2, M - 1])
        elif BC_choice == 2:
            u_fourier_coeff[N // 2, i] = v[N // 2, i] + (u_fourier_0[i] - v[N // 2, i])
        else:
            raise ValueError('Incorrect index for "BC_choice"')
        for n in range(N + 1):
            if n == N // 2:
                continue
            kabs = abs(n - (N // 2))
            if BC_choice == 1:
                B = (r_m[i] / R) ** kabs * (g_fourier_coeff[n] - v[n, M - 1])
                u_fourier_coeff[n, i] = v[n, i] + B
            elif BC_choice == 2:
                B = (r_m[i] / R) ** kabs * (R / kabs * g_fourier_coeff[n] + v[n, M - 1])
                u_fourier_coeff[n, i] = v[n, i] + B


    # --- 6. Rearrangement for IFFT: reorder rows for FFT convention ---
    u_fourier_coeff = np.vstack([u_fourier_coeff[N // 2 : N, :], u_fourier_coeff[0 : N // 2, :]])

    # --- 7. Inverse FFT in angle to recover solution ---
    u_approx = np.fft.ifft(u_fourier_coeff, axis=0) * N

    # --- 8. Rotate rows so theta=2π/N is first, matching grid convention ---
    u_approx = np.vstack([u_approx[1:, :], u_approx[:1, :]])
    
    
    
    return u_approx
