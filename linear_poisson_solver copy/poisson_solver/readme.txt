#######################
# Fast Poisson Solver #
#######################

Ver. 1.0.0

Table of Contents:
:0f: Summary
:1f: Getting Started
:2f: How to Use Your Own Data (Program Specifics) 
:3f: Known Bugs
:4f: Future Updates
:5f: References


--:0f:--
Summary
---------------------
This program uses a fast algorithm, developed by Dr. Daripa [1], to solve the 2D Poisson equation, ( Delta*u = f ),
on a disk with either Dirichlet or Nuemann boundary conditions. The program works best on a uniform mesh 
(uniform in the radial and azimuthal direction), but can work for nonuniform meshes.




--:2f:--
How to Use YOur Own Data (Program Specifics)
------------------------------------------------
The parameters are all listed in the file "poisson_example.py" and short comments are given about the different
parameters. Changes of the parameters should be intuitive, but more detailed information is listed here.

*Quadrature Rule:
You can use either Trapezoidal or Simpson's Rule for the numerical integration. Simpson's rule is more
accurate and has better convergence, but is slower. 

*Boundary Conditions:
Specify the boundary condition first by setting "bc_choice = 1" for Dirichlet or "bc_choice = 2" for Nuemann.
In addition, make sure to modifiy the boundary function, "g". For the Nuemann boundary condition, a summation
constant must be specified in order to "pin down" the function. Recall that solutions to the Poisson equation
with the Nuemann boundary condition are unique /up to/ a constant. In the program, we take this constant to be
the Fourier coefficients for the zero-th mode.

*Number of Annuli:
"M" represents the number of annuli, in other words, increase "M" to refine in the radial direction.
Increasing "M" will also improve the accuracy, as this affects the accuracy of the integration methods.

*Number of Angular Slices:
"N" Represents the number of angular slices. For best results, "N" should be a power of 2 as the index is
used in the Fast Fourier Transform. Increasing "N" should not increase accuracy except in very special 
circumstances.

*Radius of Disk:
Should be obvious.

*Nonuniform Radial Mesh
Set "rad_unif = 0" to elect a non-uniform radial mesh. We only include one specific non-uniform mesh, but 
different meshes can be implemented in the "Generate Data" section of "poisson_example.py". Note also that
special version of Simpson's rule is used for non-uniform intervals of integration (see [2]).
Different example meshes can be found in the file "generate_nonuniform_radial.py" and can simply be
uncommented for use. 

*Nonuniform Azimuthal Mesh
The nonuniform mesh in the azimuthal direction is a bit more involved as it requires the use of the 
Non-Uniform Fast Fourier Transform by Dutt and Rokhlin [3]. Results tend to be poorer if the source term,
"f", is sharply varying and if the mesh points are not uniformly distributed. Note: we specify that
"uniformly distributed" does /not/ mean the points are equispaced, but are rather distributed throughout
the whole interval. To see more details about this topic, see [4]. 

Other Comments:
+ If you are unsure of the proper syntax for writing the functions, "u", "f", and "g", the you should put
  a dot, '.', at the beggining of each multiplicative operation, '*' and '/'. 
+ If you do not know the true solution, then the "Error Computation" section should be commented out, then
  comment out the "plot_on_disk_with_error" function and uncomment the "plot_on_disk" function in the 
  "Graphing" section.


--:3f:-- 
Known Bugs
----------------------
*No known bugs.


--:4f:--
Future updates
----------------------
+ Update the solver to be compatible with the Poisson equation with variable coefficient. 
+ Improve the nonuniform azimuthal computation by using the method outlined in "Fast Fourier Transforms 
  for Nonequispaced Data II"; involves the Fast Multipole Method.


--:5f:--
References
----------------------
[1] L. Borges and P. Daripa.  A fast parallel algorithm for the Poisson equation on a disk. 
    J. Comput. Phys., 169(1):151–192, 2001

[2] Reed,  B.  C.  (2014).  Numerically  Integrating  Irregularly-spaced  (x,  y)  Data. 
    The Mathematics Enthusiast, 11(3), 643-648. 

[3] A.  Dutt  and  V.  Rokhlin.   Fast  Fourier  Transforms  for  Nonequispaced  Data.
    SIAM  J.  Sci. Comput., 14(6):1368–1393, 1993






