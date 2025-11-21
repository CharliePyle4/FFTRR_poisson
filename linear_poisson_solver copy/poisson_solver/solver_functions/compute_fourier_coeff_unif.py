import numpy as np

def compute_fourier_coeff_unif(f_values: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    f_values : np.ndarray
        Input array of function values

    Returns
    -------
    f_fourier_coeff : np.ndarray
        Fourier coefficients
    """
    N = f_values.shape[0]

    if f_values.ndim > 1 and f_values.shape[1] > 1:
        # The first row should correspond to values on the angle
        # 'theta = 0', since recall we computed the grid values
        # starting at the angle 'theta = 2*pi/N'.
        f_values = np.vstack([f_values[N-1, :], f_values[0:N-1, :]])
        
        # The FFT computes the discrete Fourier transform for
        # each fixed radius; that is, it computes the DFT of each
        # column in 'grid_values_f'
        f_fourier_coeff = np.fft.fft(f_values, axis=0) / N
        
        # In order to work with the Fourier coefficients, we
        # change the indexing from '0 <= n <= N-1' to
        # '-N/2 <= n <= N/2'.
        # We do this using the fact that for a real valued function
        # f, the Fourier coefficients have the property that,
        # f_n = conj(f_{-n}).
        f_fourier_coeff = np.vstack([f_fourier_coeff[N//2:N, :], f_fourier_coeff[0:N//2+1, :]])
        
        # There is a small technical detail we need to correct.
        # That is, f_{N/2} = f_{-N/2} because of the periodicity
        # of exp(-2*pi*k*n/N). Long story short, we need to
        # compensate for double counting the last Fourier coefficient.
        f_fourier_coeff[0, :] = f_fourier_coeff[0, :] / 2  # Divide by 2 for the double count.
        f_fourier_coeff[N, :] = f_fourier_coeff[N, :] / 2

    else:
        # Special case for when 'f_values' is a column vector. Specifically,
        # this is for comuting the boundary values.
        f_values = np.hstack([f_values[N-1], f_values[0:N-1]])
        f_fourier_coeff = np.fft.fft(f_values) / N
        f_fourier_coeff = np.hstack([f_fourier_coeff[N//2:N], f_fourier_coeff[0:N//2+1]])
        f_fourier_coeff[0] = f_fourier_coeff[0] / 2 
        f_fourier_coeff[N] = f_fourier_coeff[N] / 2

    return f_fourier_coeff 
