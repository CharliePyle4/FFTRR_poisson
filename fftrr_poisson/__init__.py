"""
FFTRR algorithm for solving Poisson equations on the unit disk, along with helper functions
"""
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
from .poisson_solver import poisson_solver

__version__ = "0.1.0"

__all__ = [
    "generate_grid_values",
    "generate_nonuniform_radial",
    "generate_cartesian_grid_on_disk",
    "compute_fourier_coeff_unif",
    "compute_C_D_uniform",
    "compute_C_D_nonuniform",
    "nonuniform_simps_rule",
    "trap_2d_on_disk",
    "plot_on_disk_with_error",
    "plot_on_disk",
    "poisson_solver",
]
