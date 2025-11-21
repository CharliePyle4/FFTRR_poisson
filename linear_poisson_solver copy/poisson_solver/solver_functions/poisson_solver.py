import numpy as np

# All helpers are sibling modules inside the same package
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





def poisson_solver(f_values, g_values, u_fourier_0,
                   N, M, r_m, theta_j, R,
                   quad_rule, BC_choice,
                   rad_unif):
    """
    Solve Δu = f on a disk of radius R in polar coords using Fourier-in-θ
    and radial integration (C, D). Direct translation of the MATLAB code.

    Parameters
    ----------
    f_values : (N, M) array
        Samples of f on the polar grid (rows=angles, cols=radii).
    g_values : (N,) or (N,1) array
        Boundary data on r=R over angles.
    u_fourier_0 : (M,) or (1,M) array
        k=0 Fourier parameter per radius for Neumann BC (unused for Dirichlet).
    N, M : int
        #angles (even) and #radii.
    r_m : (M,) array
        Radial mesh from 0 to R.
    theta_j : (N,) array
        Angular mesh (radians).
    R : float
    quad_rule : int
        1=trapezoidal, 2=Simpson.
    BC_choice : int
        1=Dirichlet, 2=Neumann.
    rad_unif : int
        Radial uniformity flag (1=uniform, 0=nonuniform).

    Returns
    -------
    u_approx : (N, M) complex array
        Approximate solution on the polar grid.
    """


    #Step 1:
    #Generate Fourier coefficients
    f_fourier_coeff = compute_fourier_coeff_unif(f_values)
    g_fourier_coeff = compute_fourier_coeff_unif(g_values) # Boundary Fourier coefficients.
    


    #Step 2:
    #Compute the integrals for C_n and D_n 

    if(rad_unif == 1):
        C, D = compute_C_D_uniform(r_m, f_fourier_coeff, quad_rule)
    elif(rad_unif == 0):
        C, D = compute_C_D_nonuniform(r_m, f_fourier_coeff, quad_rule)
    else:
        raise ValueError('Incorrect index for "rad_unif"')


    #Step 3 and 4:
    # Compute v^- and v^+ 

    #Declare placeholders: need dtype complex for complex fourier coefficients
    v_neg = np.zeros((N // 2 + 1, M), dtype=complex)
    v_pos = np.zeros((N // 2 + 1, M), dtype=complex)


    #Algorithm for v^- and v^+ with trapezoidal rule
    if quad_rule == 1:
        # seed v^- at i=1 to avoid r=0 division
        for r in range(N // 2 + 1):
            v_neg[r, 1] = C[r, 0]

        # i >= 2
        for i in range(2, M):
            for r in range(N // 2 + 1):
                exp_neg = (r) - (N // 2)          # (n - N/2 - 1) with n=r+1
                v_neg[r, i] = (r_m[i] / r_m[i - 1]) ** exp_neg * v_neg[r, i - 1] + C[r, i - 1]

        # v^+ (no r=0 division since denom is r_{i+1})
        for i in range(M - 2, -1, -1):
            for r in range(N // 2 + 1):
                exp_pos = r                       # (n - 1) with n=r+1
                v_pos[r, i] = (r_m[i] / r_m[i + 1]) ** exp_pos * v_pos[r, i + 1] + D[r, i]


    elif quad_rule == 2:
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

    ## Step 5:
    # Combine coefficients to form 'v'
    # --------------------------------
    # IMPORTANT: handle DC (k=0) / Nyquist (k=N/2) correctly and enforce
    # Hermitian symmetry by pairing k with N-k for k=1..N/2-1.

    v = np.zeros((N + 1, M), dtype=complex)

    for i in range(M):
        # Nyquist / "central" mode (k = 0 in the MATLAB sense):
        # MATLAB (1-based): v(N/2+1,i) = log(r_i)*v_neg(N/2+1,i) + v_pos(1,i) for i>1,
        # and v(N/2+1,1) = v_neg(N/2+1,1) + v_pos(1,1).
        if i != 0:
            v[N // 2, i] = np.log(r_m[i]) * v_neg[N // 2, i] + v_pos[0, i]
        else:
            v[N // 2, 0] = v_neg[N // 2, 0] + v_pos[0, 0]

        # Pair the remaining harmonics: k = 1..N/2-1
        # (exclude the Nyquist index N//2 which is handled above)
        for k in range(1, N // 2):
            # Positive side k
            # NOTE the corrected conjugate index for v_pos:
            #   MATLAB: v(n,i) = v_neg(n,i) + conj(v_pos(N/2 - n + 2, i))
            #   Python (0-based): v[k,i] = v_neg[k,i] + conj(v_pos[N//2 - k, i])
            v[k, i] = v_neg[k, i] + np.conj(v_pos[N // 2 - k, i])

            # Mirror to enforce Hermitian symmetry
            # (MATLAB: v(N-n+2, i) = conj(v(n,i))
            #  Python: index N-k)
            v[N - k, i] = np.conj(v[k, i])

        # Optional: explicitly zero out any residual imag at the central bin
        # v[N // 2, i] = np.real(v[N // 2, i])





        ## Step 6:
    # Compute the Fourier coefficients for 'u'
    # ---------------------------------------
    # Use 0-based indexing: rows n=0..N, central (k=0) at n = N//2.
    # Exponent must be |n - N//2|.

    u_fourier_coeff = np.zeros((N + 1, M), dtype=complex)

    for i in range(M):
        # --- central bin (k = 0) ---
        if BC_choice == 1:  # Dirichlet
            # (r/R)^0 = 1
            u_fourier_coeff[N // 2, i] = v[N // 2, i] + (g_fourier_coeff[N // 2] - v[N // 2, M - 1])
        elif BC_choice == 2:  # Neumann
            # provided u_fourier_0 gives the correct central coefficient
            u_fourier_coeff[N // 2, i] = v[N // 2, i] + (u_fourier_0[i] - v[N // 2, i])
        else:
            raise ValueError('Incorrect index for "BC_choice"')

        # --- all other rows ---
        for n in range(N + 1):
            if n == N // 2:
                continue  # handled above

            kabs = abs(n - (N // 2))  # <-- CORRECT exponent

            if BC_choice == 1:  # Dirichlet
                B = (r_m[i] / R) ** kabs * (g_fourier_coeff[n] - v[n, M - 1])
                u_fourier_coeff[n, i] = v[n, i] + B

            elif BC_choice == 2:  # Neumann
                # denominator uses |n - N//2| and is never zero here
                B = (r_m[i] / R) ** kabs * (R / kabs * g_fourier_coeff[n] + v[n, M - 1])
                u_fourier_coeff[n, i] = v[n, i] + B


        
    ## Step 7:
    # Compute the approximate solution #
    #----------------------------------#
    # Rearrange the Fourier coefficients in the order of 0 to N so that we can apply the IFFT.
    u_fourier_coeff = np.vstack([u_fourier_coeff[N // 2 : N, :],u_fourier_coeff[0 : N // 2, :]])

        
    # The IFFT in numpy divides by N (see the FFT documentation)
    # so we multiply by N to compensate.
    u_approx = np.fft.ifft(u_fourier_coeff, axis=0) * N
        
    # Rearrange u so that the first row starts at 'theta = 2*pi/N' instead of 'theta = 0'.
    u_approx = np.vstack([u_approx[1:, :], u_approx[:1, :]])


    
    
            
    
    return u_approx
