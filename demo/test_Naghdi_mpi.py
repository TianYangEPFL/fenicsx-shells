# %%
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import ufl
from dolfinx.fem import Function, FunctionSpace, dirichletbc, Constant, Expression, locate_dofs_topological
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary
from ufl import (FiniteElement, MixedElement, VectorElement, grad, inner,
                 split)
import pyvista
from dolfinx import plot, log
from dolfinx.nls.petsc import NewtonSolver


from tqdm import tqdm
import matplotlib.pyplot as plt

# %% [markdown]
# ## Parameters of the geometry and the material

# %%
# the radius of the cylinder
r = 1.016 

# the length of the cylinder
L = 3.048 

# the Young's modulus and Poisson's ratio
E, nu = 2.0685E7, 0.3 

# the shear modulus
mu = E/(2.0*(1.0 + nu)) 

# the lame parameter
lmbda = 2.0*mu*nu/(1.0 - 2.0*nu) 

# the thickness of the cylinder
t = 0.03 

# %% [markdown]
# # Generate the mesh in the parameter space 
# $x_1 \in [- \pi / 2, \pi /2], x_2 \in [0, L]$

# %%
mesh = create_rectangle(MPI.COMM_WORLD, np.array([[-np.pi / 2, 0], [np.pi / 2, L]]), [20, 20], CellType.triangle)
tdim = mesh.topology.dim

# %% [markdown]
# # Define the initial shape

# %%
x = ufl.SpatialCoordinate(mesh)
phi0_ufl = ufl.as_vector([r * ufl.sin(x[0]), x[1], r * ufl.cos(x[0])])


# %% [markdown]
# # Define the initial local orthonormal basis
# 

# %% [markdown]
# ## Unit normal basis 
# $$
#  \vec{n}  = \frac{\partial_1 \phi_0 \times \partial_2 \phi_0}{\| \partial_1 \phi_0 \times \partial_2 \phi_0 \|}
# $$

# %%
def unit_normal(phi):
    n = ufl.cross(phi.dx(0), phi.dx(1))
    return n/ufl.sqrt(inner(n, n))

# the analytical expression of n0
n0_ufl = unit_normal(phi0_ufl)

# %% [markdown]
# ## two tangent basis
# $$
# \vec{t}_{0i} = \mathbf{R}_0 \vec{e}_i \\
# \vec{n} = \vec{t}_{03} \\
# $$
# 
# Define $\vec{t}_{01}$ and $\vec{t}_{02}$ with $\vec{e}_1$ and $\vec{e}_2$ 
# $$
# \vec{t}_{01} = \frac{\vec{e}_2 \times \vec{n}}{\| \vec{e}_2 \times \vec{n}\|} \\
# \vec{t}_{02} =   \vec{n} \times \vec{t}_{01}
# $$
# 
# The corresponding rotation matrix $\mathbf{R}_0$:
# $$
# \mathbf{R}_0 = [\vec{t}_{01}; \vec{t}_{02}; \vec{n}]
# $$
# 
# 

# %%
def tangent_1(n):
    e2 = ufl.as_vector([0, 1, 0])
    t1 = ufl.cross(e2, n)
    t1 = t1/ufl.sqrt(inner(t1, t1))
    return t1

def tangent_2(n, t1):
    t2 = ufl.cross(n, t1)
    t2 = t2/ufl.sqrt(inner(t2, t2))
    return t2

# the analytical expression of t1 and t2
t1_ufl = tangent_1(n0_ufl)
t2_ufl = tangent_2(n0_ufl, t1_ufl)

# the analytical expression of R0
def rotation_matrix(t1, t2, n):
    R = ufl.as_matrix([[t1[0], t2[0], n[0]], 
                       [t1[1], t2[1], n[1]], 
                       [t1[2], t2[2], n[2]]])
    return R

R0_ufl = rotation_matrix(t1_ufl, t2_ufl, n0_ufl)

# %% [markdown]
# # The parameterization of director 
# Update the director with two successive elementary rotations
# $$
#     \vec{t}_3 = \mathbf{R}_0 \cdot \vec{\Lambda_3}
# $$
# 
# $$
# \vec{\Lambda_3} = [\sin(\theta_2)\cos(\theta_1), -\sin(\theta_1), \cos(\theta_2)\cos(\theta_1)]^\text{T}
# $$
# 
# Derivation: 
# 
# 
# $\theta_1$, $\theta_2$ are the rotation angles about the fixed axis $\vec{e}_1$ and $\vec{e}_2$ or follower axes $\mathbf{t}_1$ and $\mathbf{t}_2$
# 
# $$
# \vec{t}_i = \mathbf{R} \vec{e}_i \\
# 
# \mathbf{R}  = \text{exp}[\theta_1 \hat{\mathbf{t}}_1] \text{exp}[\theta_2 \hat{\mathbf{t}}_{02}] \mathbf{R}_0 
# $$
# 
# where $\mathbf{t}_{02} = \mathbf{R}_0 \cdot \vec{e}_2 $, $\mathbf{t}_1 = \text{exp}[\theta_2 \hat{\mathbf{t}}_{02}] \cdot \mathbf{t}_{01} $
# 
# $$
# \mathbf{R} = \mathbf{R}_0 \text{exp}[\theta_2 \hat{\mathbf{e}}_{2}] \text{exp}[\theta_1 \hat{\mathbf{e}}_1]
# 
# $$

# %%
# Update the director with two successive elementary rotations

def director(R0, theta):
    Lm3 = ufl.as_vector([ufl.sin(theta[1])*ufl.cos(theta[0]), -ufl.sin(theta[0]), ufl.cos(theta[1])*ufl.cos(theta[0])])
    d = ufl.dot(R0, Lm3)
    return d

# %% [markdown]
# # Nonlinear Naghdi Shell elements

# %% [markdown]
# ## Define element
# 
# For 3 translations $u_x$, $u_y$, and $u_z$, we use 2nd order Langrange elements enriched with 3rd order Bubble elements
# 
# For 2 rotations $\theta_1$ and $\theta_2$, we use 2nd order Langrange elements.

# %%
# for the 3 translation DOFs, we use the P2 + B3 enriched element
P2 = FiniteElement("Lagrange", ufl.triangle, degree = 2)
B3 = FiniteElement("Bubble", ufl.triangle, degree = 3)

# Enriched
P2B3 = P2 + B3

# for 2 rotation DOFs, we use P2 element

# mixed element for the nonlinear Naghdi shell
naghdi_shell_element = MixedElement([VectorElement(P2B3, dim = 3), VectorElement(P2, dim=2)])
naghdi_shell_FS = FunctionSpace(mesh, naghdi_shell_element)

# %% [markdown]
# ## Define `Function`

# %%
q_func = Function(naghdi_shell_FS)
q_trial = ufl.TrialFunction(naghdi_shell_FS)
q_test = ufl.TestFunction(naghdi_shell_FS)

u_func, theta_func = split(q_func)

# %% [markdown]
# ## Define metric and curvature tensor
# 
# Deformation gradient
# $$
# \mathbf{F} = \nabla \vec{\phi} \quad  (F_{ij} = \frac{\partial \phi_i}{\partial \xi_j}); \quad \vec{\phi} = \vec{\phi}_0 + 
# \vec{u}
# $$
# 
# Metric tensor $\mathbf{a} \in \mathbb{S}^2_+$ and curvature tensor $\mathbf{b} \in \mathbb{S}^2$ (First and second fundamental form)
# $$
# \mathbf{a} = {\nabla \vec{\phi}} ^{T} \nabla \vec{\phi} \\
# \mathbf{b} = -\frac{1}{2}({\nabla \vec{\phi}} ^{T} \nabla \vec{d} + {\nabla \vec{d}} ^{T} \nabla \vec{\phi})
# 
# $$
# 
# Initial configuration, $\vec{d} = \vec{n}_0$, $\vec{\phi} = \vec{\phi}_0$, conresponding initial tensors $\mathbf{a}_0$, $\mathbf{b}_0$

# %%
# current deformation gradient 
F = grad(u_func) + grad(phi0_ufl) 

# current director
d = director(R0_ufl, theta_func)

# initial metric and curvature tensor a0 and b0
a0_ufl = grad(phi0_ufl).T * grad(phi0_ufl)
b0_ufl = -0.5*( grad(phi0_ufl).T * grad(n0_ufl) + grad(n0_ufl).T * grad(phi0_ufl) )

# %% [markdown]
# ## Define strain measures
# 
# - Membrane strain tensor $\boldsymbol{\varepsilon}(\vec{u})$
# 
# $$
# \boldsymbol{\varepsilon} (\vec{u})= \frac{1}{2} \left ( \mathbf{a}(\vec{u}) - \mathbf{a}_0 \right)
# $$
# 
# - Bending strain tensor $\boldsymbol{\kappa}(\vec{u}, \vec{\theta})$ 
# 
# $$
# \boldsymbol{\kappa}(\vec{u}, \vec{\theta}) = \mathbf{b}(\vec{u}, \vec{\theta}) - \mathbf{b}_0
# $$
# 
# - transverse shear strain vector $\vec{\gamma}(\vec{u}, \vec{\theta})$ 
# 
# $$
# \begin{aligned}
# \vec{\gamma}(\vec{u}, \vec{\theta}) & = {\nabla \vec{\phi}(\vec{u})}^T \vec{d}(\vec{\theta}) - {\nabla\vec{\phi}_0}^T \vec{n}_0 \\
# & = {\nabla \vec{\phi}(\vec{u})}^T \vec{d}(\vec{\theta}) \quad \text{if zero initial shears}
# \end{aligned}
# $$
# 
# 

# %%
# membrane strain
epsilon = lambda F: 0.5*(F.T * F - a0_ufl)

# bending strain
kappa = lambda F, d: -0.5 * (F.T * grad(d) + grad(d).T * F) - b0_ufl

# transverse shear strain (zero initial shear strain)
gamma = lambda F, d: F.T * d

# %% [markdown]
# ## Define isotropic linear material model
# 
# - Membrane stiffness modulus $A^{\alpha\beta\sigma\tau}$, $D^{\alpha\beta\sigma\tau}$ (__contravariant__ components)
# 
# $$
# \frac{A^{\alpha\beta\sigma\tau}}t=12\frac{D^{\alpha\beta\sigma\tau}}{t^3}=\frac{2\lambda\mu}{\lambda+2\mu}a_0^{\alpha\beta}a_0^{\sigma\tau}+\mu(a_0^{\alpha\sigma}a_0^{\beta\tau}+a_0^{\alpha\tau}a_0^{\beta\sigma})
# $$
# 
# - Shear stiffness modulus $S^{\alpha\beta}$
# 
# $$
# \frac{S^{\alpha\beta}}t = \mu a_0^{\alpha\beta}
# $$
# 

# %%
a0_contra_ufl = ufl.inv(a0_ufl)
j0_ufl = ufl.det(a0_ufl)

i,j,l,m = ufl.indices(4)
A_contra_ufl = ufl.as_tensor( ( ((2.0*lmbda*mu) / (lmbda + 2.0*mu)) * a0_contra_ufl[i,j]*a0_contra_ufl[l,m]
                + 1.0*mu* (a0_contra_ufl[i,l]*a0_contra_ufl[j,m] + a0_contra_ufl[i,m]*a0_contra_ufl[j,l]) )
                ,[i,j,l,m])


# %% [markdown]
# ## Define stress measures
# 
# - Membrane stress tensor $\mathbf{N}$
# 
# $$
# \mathbf{N} = \mathbf{A} : \boldsymbol{\varepsilon}
# $$
# 
# - Bending stress tensor $\mathbf{M}$
# 
# $$
# \mathbf{M} = \mathbf{D} : \boldsymbol{\kappa}
# $$
# 
# - Shear stress vector $\vec{T}$
# 
# $$
# \vec{T} = \mathbf{S} \cdot \vec{\gamma}
# $$
# 

# %%
N = ufl.as_tensor(t * A_contra_ufl[i,j,l,m] * epsilon(F)[l,m], [i,j])

M = ufl.as_tensor( (t**3 / 12.0) * A_contra_ufl[i,j,l,m]*kappa(F, d)[l,m], [i,j])

T = ufl.as_tensor( (t * mu) * a0_contra_ufl[i, j] * gamma(F, d)[j], [i])

# %% [markdown]
# ## Define elastic strain energy density
# $\psi_{m}$, $\psi_{b}$, $\psi_{s}$ for membrane, bending and shear, respectively.
# 
# $$
# \psi_m = \frac{1}{2} \mathbf{N} : \boldsymbol{\varepsilon}; \quad
# \psi_b = \frac{1}{2} \mathbf{M} : \boldsymbol{\kappa}; \quad
# \psi_s = \frac{1}{2} \vec{T} \cdot \vec{\gamma}
# $$

# %%
psi_m = 0.5*inner(N, epsilon(F))

psi_b = 0.5*inner(M, kappa(F, d))

psi_s = 0.5*inner(T, gamma(F, d))

# %% [markdown]
# ## Partial selective reduced integration (PSRI)
# 
# We introduce a parameter $\alpha \in \mathbb{R}$ that splits the membrane and shear energy in the energy functional into a weighted sum of two parts:
# 
# $$
# \begin{aligned}\Pi_{N}(u,\theta)&=\Pi^b(u_h,\theta_h)+\alpha\Pi^m(u_h)+(1-\alpha)\Pi^m(u_h)\\&+\alpha\Pi^s(u_h,\theta_h)+(1-\alpha)\Pi^s(u_h,\theta_h)-W_{\mathrm{ext}},\end{aligned}
# $$
# 
# We apply reduced integration to the parts weighted by the factor $(1-\alpha)$
# 
# - Optimal choice $\alpha = \frac{t^2}{h^2}$, $h$ is the diameter of the cell
# - Full integration : Gauss quadrature of degree 4 (6 integral points for triangle)
# - Reduced integration : Gauss quadrature of degree 2 (3 integral points for triangle)
# 
# __More on lockings__:
# 
# As we can see in the definition of bending, membrane and shear stiffness modulus:
# $$
# \Pi ^ b \propto t^3, \quad \Pi^m \propto t, \quad \Pi^s \propto t
# $$
# 
# when thickness $t$ is very small, $\Pi^b$ is much smaller than $\Pi^m$ and $\Pi^s$. In order to find the minimum of total energy functional, it is almost equivalent to require  $\Pi^m = \Pi^s = 0$. 
# 
# However, with common FEM discretizations, this requirement is too strong, the resulting solution of $\vec{u}_h, \vec{\theta}_h$ is very close to zero even at strong external forces, unless the mesh size is small compared with the thickness.
# 
# For example, we consider shear strain $\gamma = \nabla w - \theta$, with 1st order piecewise polynomal approximations $w_h$ and $\theta_h$. The only solution for $\gamma_h = \nabla w_h - \theta_h = 0$ is $w_h = 0, \theta_h = 0$
# 
# __More on reduced integration__:
# 
# With reduced integration, the new constraint is far less restrictive. It is now possible to have nonzero solutions $u_h, \theta_h $ for $\Pi^{m,r} = \Pi^{s,r} = 0$.
# 
# How reduced integration may threaten the stability of FEM, because of the existence of spurious zero-stiffness modes. In other words, there may exist nonzero solutions $\vec{q}_{nz}$ for all the three kinds of energy equal to zero: $\Pi^{b,r}(\vec{q}_{nz}) = \Pi^{m,r}(\vec{q}_{nz}) = \Pi^{s,r}(\vec{q}_{nz}) = 0$

# %%
dx_f = ufl.Measure('dx', domain=mesh, metadata={"quadrature_degree": 4})
dx_r = ufl.Measure('dx', domain=mesh, metadata={"quadrature_degree": 2})

h = ufl.CellDiameter(mesh)

alpha_FS = FunctionSpace(mesh, FiniteElement("DG", ufl.triangle, 0))
alpha_expr = Expression(t**2 / h**2, alpha_FS.element.interpolation_points())

alpha = Function(alpha_FS)
alpha.interpolate(alpha_expr)

# Full integration part
Pi_PSRI = psi_b * ufl.sqrt(j0_ufl) * dx_f 
Pi_PSRI += alpha * psi_m * ufl.sqrt(j0_ufl) * dx_f
Pi_PSRI += alpha * psi_s * ufl.sqrt(j0_ufl) * dx_f

# Reduced integration part
Pi_PSRI += (1.0 - alpha) * psi_m * ufl.sqrt(j0_ufl) * dx_r
Pi_PSRI += (1.0 - alpha) * psi_s * ufl.sqrt(j0_ufl) * dx_r

# %% [markdown]
# ## Internal Force vector and Jacobian matrix

# %%
Residual = ufl.derivative(Pi_PSRI, q_func, q_test)
Jacobian = ufl.derivative(Residual, q_func, q_trial)

# %% [markdown]
# # Boundary conditions

# %% [markdown]
# ## Dirichlet BC
# 

# %% [markdown]
# 
# - Clamped boundary conditions at $X_1 = 0$

# %%
def clamped_boundary(x):
    return np.isclose(x[1], 0.0)

fdim = tdim - 1

clamped_facets = locate_entities_boundary(mesh, fdim, clamped_boundary)


u_FS, _ = naghdi_shell_FS.sub(0).collapse()
theta_FS, _ = naghdi_shell_FS.sub(1).collapse()

# u1, u2, u3 = 0 on the clamped boundary
u_clamped = Function(u_FS)
clamped_dofs_u = locate_dofs_topological((naghdi_shell_FS.sub(0), u_FS), fdim, clamped_facets)
bc_clamped_u = dirichletbc(u_clamped, clamped_dofs_u, naghdi_shell_FS.sub(0))

# theta1, theta2 = 0 on the clamped boundary
theta_clamped = Function(theta_FS)
clamped_dofs_theta = locate_dofs_topological((naghdi_shell_FS.sub(1), theta_FS), fdim, clamped_facets)
bc_clamped_theta = dirichletbc(theta_clamped, clamped_dofs_theta, naghdi_shell_FS.sub(1))


# %% [markdown]
# - Symmetric boundary conditions on left and right side ($u_3 = 0, \theta_2 = 0$)

# %%
def symm_boundary(x):
    return np.isclose(abs(x[0]), np.pi/2)

symm_facets = locate_entities_boundary(mesh, fdim, symm_boundary)

symm_dofs_u = locate_dofs_topological((naghdi_shell_FS.sub(0).sub(2), u_FS.sub(2)), fdim, symm_facets)
bc_symm_u = dirichletbc(u_clamped, symm_dofs_u, naghdi_shell_FS.sub(0).sub(2))

symm_dofs_theta = locate_dofs_topological((naghdi_shell_FS.sub(1).sub(1), theta_FS.sub(1)), fdim, symm_facets)
bc_symm_theta = dirichletbc(theta_clamped, symm_dofs_theta, naghdi_shell_FS.sub(1).sub(1))


# %%
bcs = [bc_clamped_u, bc_clamped_theta, bc_symm_u, bc_symm_theta]

# %% [markdown]
# # Point Force 

# %%
def compute_cell_contributions(V, points):
    # Determine what process owns a point and what cells it lies within
    mesh = V.mesh
    _, _, owning_points, cells = dolfinx.cpp.geometry.determine_point_ownership(
        mesh._cpp_object, points, 1e-6)
    owning_points = np.asarray(owning_points).reshape(-1, 3)

    # Pull owning points back to reference cell
    mesh_nodes = mesh.geometry.x
    cmap = mesh.geometry.cmaps[0]
    ref_x = np.zeros((len(cells), mesh.geometry.dim),
                     dtype=mesh.geometry.x.dtype)
    for i, (point, cell) in enumerate(zip(owning_points, cells)):
        geom_dofs = mesh.geometry.dofmap[cell]
        ref_x[i] = cmap.pull_back(point.reshape(-1, 3), mesh_nodes[geom_dofs])

    # Create expression evaluating a trial function (i.e. just the basis function)
    u = ufl.TrialFunction(V.sub(0).sub(2))
    num_dofs = V.sub(0).sub(2).dofmap.dof_layout.num_dofs * V.sub(0).sub(2).dofmap.bs
    if len(cells) > 0:
        # NOTE: Expression lives on only this communicator rank
        expr = dolfinx.fem.Expression(u, ref_x, comm=MPI.COMM_SELF)
        values = expr.eval(mesh, np.asarray(cells, dtype=np.int32))

        # Strip out basis function values per cell
        basis_values = values[:num_dofs:num_dofs*len(cells)]
    else:
        basis_values = np.zeros(
            (0, num_dofs), dtype=dolfinx.default_scalar_type)
    return cells, basis_values

# %%
# Point source
if mesh.comm.rank == 0:
    points = np.array([[0.0, L, 0.0]], dtype=mesh.geometry.x.dtype)
else:
    points = np.zeros((0, 3), dtype=mesh.geometry.x.dtype)

cells, basis_values = compute_cell_contributions(naghdi_shell_FS, points)

# %% [markdown]
# # Nonlinear problem plus point source

# %%
import typing
from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.function import Function as _Function
from dolfinx.fem.petsc import assemble_vector, apply_lifting, set_bc

class NonlinearProblemPointSource(NonlinearProblem):
    def __init__(self, F: ufl.form.Form, u: _Function, bcs: typing.List[DirichletBC] = [],
                 J: ufl.form.Form = None, cells = [], basis_values = [], PS: float = 0.0):
        
        super().__init__(F, u, bcs, J)
        
        self.PS = PS
        self.cells = cells
        self.basis_values = basis_values
        self.function_space = u.function_space
        
    def F(self, x: PETSc.Vec, b: PETSc.Vec) -> None:
        # Reset the residual vector
        with b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(b, self._L)
    
        # Add point source
        if len(self.cells) > 0:
            for cell, basis_value in zip(self.cells, self.basis_values):
                dofs = self.function_space.sub(0).sub(2).dofmap.cell_dofs(cell)
                with b.localForm() as b_local:
                    b_local.setValuesLocal(dofs, basis_value * self.PS, addv=PETSc.InsertMode.ADD_VALUES)
    

        # Apply boundary condition
        apply_lifting(b, [self._a], bcs=[self.bcs], x0=[x], scale=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, self.bcs, x, -1.0)

# %%
problem = NonlinearProblemPointSource(Residual, q_func, bcs, Jacobian, cells, basis_values)

solver = NewtonSolver(mesh.comm, problem)

# Set Newton solver options
solver.rtol = 1e-6
solver.atol = 1e-6
solver.max_it = 30
solver.convergence_criterion = "incremental"
solver.report = True

# Modify the linear solver in each Newton iteration
ksp = solver.krylov_solver
opts = PETSc.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "cg"
#opts[f"{option_prefix}pc_type"] = "gamg"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

# %%


# target point


# bb_tree = dolfinx.geometry.bb_tree(mesh, 2)
# bb_point = np.array([[0.0, L, 0.0]], dtype=np.float64)

# # Find the leaf that the target point is in
# bb_cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, bb_point)

# # Find the cell from the leaf that contains the target point
# bb_cells = dolfinx.geometry.compute_colliding_cells(
#          mesh, bb_cell_candidates, bb_point)


# %%
#log.set_log_level(log.LogLevel.INFO)

PS_diff = 50.0
n_step = 40

if mesh.comm.rank == 0:
    u3_list = np.zeros(n_step + 1)
    PS_list = np.arange(0, PS_diff * (n_step + 1), PS_diff)

q_func.x.array[:] = 0.0

bb_point = np.array([[0.0, L, 0.0]], dtype=np.float64)

for i in range(1, n_step + 1):
    problem.PS = PS_diff * i
    n, converged = solver.solve(q_func)
    assert (converged)
    q_func.x.scatter_forward()
    if mesh.comm.rank == 0:
        print(f"Load step {i:d}, Number of iterations: {n:d}, Load: {problem.PS:.2f}", flush=True)
    
    u3_bb = None
    u3_func = q_func.sub(0).sub(2).collapse()
    if len(cells) > 0:
        u3_bb = u3_func.eval(bb_point, cells[0])[0]
        
    
    u3_bb = mesh.comm.gather(u3_bb, root=0)

    if mesh.comm.rank == 0:
        for u3 in u3_bb:
            if u3 is not None:
                u3_list[i] = u3
                break

# %% [markdown]
# ## Plot the deformed shape

# %%
# interpolate phi_ufl into CG2 Space
#log.set_log_level(log.LogLevel.OFF)

u_P2B3 = q_func.sub(0).collapse()
theta_P2 = q_func.sub(1).collapse()

phi_FS = FunctionSpace(mesh, VectorElement("Lagrange", ufl.triangle, degree = 2, dim = 3))


phi_expr = Expression(phi0_ufl + u_P2B3, phi_FS.element.interpolation_points())

phi_func = Function(phi_FS)
phi_func.interpolate(phi_expr)

u_P2 = Function(phi_FS)
u_P2.interpolate(u_P2B3)

with dolfinx.io.VTXWriter(mesh.comm, "u_naghdi.bp", [u_P2]) as vtx:
     vtx.write(0)

with dolfinx.io.VTXWriter(mesh.comm, "theta_naghdi.bp", [theta_P2]) as vtx:
     vtx.write(0)

with dolfinx.io.VTXWriter(mesh.comm, "phi_naghdi.bp", [phi_func]) as vtx:
     vtx.write(0)


if mesh.comm.rank == 0:
    np.savetxt("u3_naghdi.txt", u3_list)
    fig = plt.figure(figsize = (8, 6))
    
    reference_u3 = 1.e-2*np.array([0., 5.421, 16.1, 22.195, 27.657, 32.7, 37.582, 42.633,
            48.537, 56.355, 66.410, 79.810, 94.669, 113.704, 124.751, 132.653,
            138.920, 144.185, 148.770, 152.863, 156.584, 160.015, 163.211,
            166.200, 168.973, 171.505])
    
    reference_P = 2000.*np.array([0., .05, .1, .125, .15, .175, .2, .225, .25, .275, .3,
            .325, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .9, .95, 1.])

    l1 = plt.plot(-u3_list, PS_list, label='FEniCS')
    l2 = plt.plot(reference_u3, reference_P, "or", label='Sze (Abaqus S4R)')
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Load (N)")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("naghdi_shell.png")
