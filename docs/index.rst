fftrr_poisson: Fast Poisson Solver on the Disk
==============================================

A Fast Fourier Transform Recursive Relation Python solver for the Poisson equation on the unit disk, supporting Dirichlet and Neumann boundary conditions, with built-in error diagnostics and plotting.

**Source code:** https://github.com/CharliePyle4/FFTRR_poisson  
**Documentation:** https://fftrr-poisson.readthedocs.io

Welcome!
--------

Solve the Poisson equation efficiently in Python using the FFTRR algorithm developed by Dr. Daripa.

**Features:**
- FFTRR-algorithm solution on the disk
- Uniform or non-uniform radial meshes
- Error analysis and visualization 

Quickstart
----------

.. code-block:: python

   import numpy as np
   from fftrr_poisson import poisson_solver
   # Define your f(x, y), boundary conditions, grid, then call poisson_solver

Installation
------------

.. code-block:: bash

   pip install fftrr_poisson

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Navigation

   api_reference