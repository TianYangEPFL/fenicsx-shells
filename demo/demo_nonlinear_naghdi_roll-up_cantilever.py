# %% [markdown]
# # A non-linear Naghdi roll-up cantilever

# %%
import numpy as np

# %%
import dolfinx
import ufl
from dolfinx.fem import Function, FunctionSpace, dirichletbc, Expression, locate_dofs_topological,Constant
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.mesh import CellType, create_rectangle, locate_entities_boundary, meshtags
from ufl import (FiniteElement, MixedElement, VectorElement, grad, inner,
                 split)
from dolfinx import plot, log, default_scalar_type
from dolfinx.nls.petsc import NewtonSolver
import matplotlib.pyplot as plt
# %%
from mpi4py import MPI
from petsc4py import PETSc


# %%
length = 12.0
width = 1.0
mesh = create_rectangle(MPI.COMM_WORLD, np.array([[0.0, -width/2.0], [length, width/2.0]]), [48, 4], CellType.triangle, 
                        diagonal=dolfinx.cpp.mesh.DiagonalType.crossed)
tdim = mesh.topology.dim
fdim = tdim - 1

# %%
from pathlib import Path

results_folder = Path("results/nonlinear_Naghdi/roll-up_cantilever")
results_folder.mkdir(exist_ok=True, parents=True)

# %%
E, nu = Constant(mesh, default_scalar_type(1.2E6)), Constant(mesh, default_scalar_type(0.0))
mu = E/(2.0*(1.0 + nu))
lmbda = 2.0*mu*nu/(1.0 - 2.0*nu)
t = Constant(mesh, default_scalar_type(1E-1))

# %% [markdown]
# ## shell model

# %% [markdown]
# initial shape

# %%
x = ufl.SpatialCoordinate(mesh)
phi0_ufl = ufl.as_vector([x[0], x[1], 0])

def unit_normal(phi):
    n = ufl.cross(phi.dx(0), phi.dx(1))
    return n/ufl.sqrt(inner(n, n))


n0_ufl = unit_normal(phi0_ufl)

# %% [markdown]
# Initial rotation matrix

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
# update director 

# %%
# Update the director with two successive elementary rotations

def director(R0, theta):
    Lm3 = ufl.as_vector([ufl.sin(theta[1])*ufl.cos(theta[0]), -ufl.sin(theta[0]), ufl.cos(theta[1])*ufl.cos(theta[0])])
    d = ufl.dot(R0, Lm3)
    return d

# %% [markdown]
# Shell element

# %%
# for the 3 translation DOFs, we use the P2 + B3 enriched element
P2 = FiniteElement("Lagrange", ufl.triangle, degree = 2)
B3 = FiniteElement("Bubble", ufl.triangle, degree = 3)

# Enriched
P2B3 = P2 + B3

# for 2 rotation DOFs, we use P2 element
# mixed element for u and theta
naghdi_shell_element = MixedElement([VectorElement(P2B3, dim = 3), VectorElement(P2, dim=2)])
naghdi_shell_FS = FunctionSpace(mesh, naghdi_shell_element)

# %% [markdown]
# Trial, test functions

# %%
q_func = Function(naghdi_shell_FS) # current configuration
q_trial = ufl.TrialFunction(naghdi_shell_FS)
q_test = ufl.TestFunction(naghdi_shell_FS)

u_func, theta_func = split(q_func) # current displacement and rotation

# %% [markdown]
# strain

# %%
# current deformation gradient 
F = grad(u_func) + grad(phi0_ufl) 

# current director
d = director(R0_ufl, theta_func)

# initial metric and curvature tensor a0 and b0
a0_ufl = grad(phi0_ufl).T * grad(phi0_ufl)
b0_ufl = -0.5*( grad(phi0_ufl).T * grad(n0_ufl) + grad(n0_ufl).T * grad(phi0_ufl))

# membrane strain
epsilon = lambda F: 0.5*(F.T * F - a0_ufl)

# bending strain
kappa = lambda F, d: -0.5 * (F.T * grad(d) + grad(d).T * F) - b0_ufl

# transverse shear strain (zero initial shear strain)
gamma = lambda F, d: F.T * d

# %% [markdown]
# constitution law

# %%
a0_contra_ufl = ufl.inv(a0_ufl)
j0_ufl = ufl.det(a0_ufl)

i,j,l,m = ufl.indices(4)
A_contra_ufl = ufl.as_tensor( ( ((2.0*lmbda*mu) / (lmbda + 2.0*mu)) * a0_contra_ufl[i,j]*a0_contra_ufl[l,m]
                + 1.0*mu* (a0_contra_ufl[i,l]*a0_contra_ufl[j,m] + a0_contra_ufl[i,m]*a0_contra_ufl[j,l]) )
                ,[i,j,l,m])

# %% [markdown]
# stress, and elastic energy density

# %%
N = ufl.as_tensor(t * A_contra_ufl[i,j,l,m] * epsilon(F)[l,m], [i,j])

M = ufl.as_tensor( (t**3 / 12.0) * A_contra_ufl[i,j,l,m]*kappa(F, d)[l,m], [i,j])

T = ufl.as_tensor( (t * mu *5.0 / 6.0) * a0_contra_ufl[i, j] * gamma(F, d)[j], [i])

psi_m = 0.5*inner(N, epsilon(F))

psi_b = 0.5*inner(M, kappa(F, d))

psi_s = 0.5*inner(T, gamma(F, d))

# %% [markdown]
# Locate left and right boundaries and create tags for them

# %%
def left(x):
    return np.isclose(x[0], 0)


def right(x):
    return np.isclose(x[0], length)

left_facets = locate_entities_boundary(mesh, fdim, left)
right_facets = locate_entities_boundary(mesh, fdim, right)

marked_facets = np.hstack([left_facets, right_facets])
marked_values = np.hstack([np.full_like(left_facets, 1), np.full_like(right_facets, 2)])
sorted_facets = np.argsort(marked_facets)
facet_tag = meshtags(mesh, fdim, marked_facets[sorted_facets], marked_values[sorted_facets])

# %% [markdown]
# external work on right boundary

# %%
M_right = Constant(mesh, default_scalar_type(0.0))

ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag, metadata={"quadrature_degree": 2})

W_ext = M_right * theta_func[1] * ds(2)

# %% [markdown]
# Potential energy with PSRI

# %%
# Full integration of order 4
dx_f = ufl.Measure('dx', domain=mesh, metadata={"quadrature_degree": 4})

# Reduced integration of order 2
dx_r = ufl.Measure('dx', domain=mesh, metadata={"quadrature_degree": 2})

# Calculate the factor alpha as a function of the mesh size h
h = ufl.CellDiameter(mesh)
alpha_FS = FunctionSpace(mesh, FiniteElement("DG", ufl.triangle, 0))
alpha_expr = Expression(t**2 / h**2, alpha_FS.element.interpolation_points())
alpha = Function(alpha_FS)
alpha.interpolate(alpha_expr)

# Full integration part of the total elastic energy
Pi_PSRI = psi_b * ufl.sqrt(j0_ufl) * dx_f 
Pi_PSRI += alpha * psi_m * ufl.sqrt(j0_ufl) * dx_f
Pi_PSRI += alpha * psi_s * ufl.sqrt(j0_ufl) * dx_f

# Reduced integration part of the total elastic energy
Pi_PSRI += (1.0 - alpha) * psi_m * ufl.sqrt(j0_ufl) * dx_r
Pi_PSRI += (1.0 - alpha) * psi_s * ufl.sqrt(j0_ufl) * dx_r

# external work part (zero in this case)
Pi_PSRI -= W_ext

# %% [markdown]
# Residual and Jacobian

# %%
Residual = ufl.derivative(Pi_PSRI, q_func, q_test)
Jacobian = ufl.derivative(Residual, q_func, q_trial)

# %% [markdown]
# clamped left boundary condtions

# %%
u_FS, _ = naghdi_shell_FS.sub(0).collapse()
theta_FS, _ = naghdi_shell_FS.sub(1).collapse()

# u1, u2, u3 = 0 on the clamped boundary
u_clamped = Function(u_FS) # default value is 0
clamped_dofs_u = locate_dofs_topological((naghdi_shell_FS.sub(0), u_FS), fdim, facet_tag.find(1))
bc_clamped_u = dirichletbc(u_clamped, clamped_dofs_u, naghdi_shell_FS.sub(0))

# theta1, theta2 = 0 on the clamped boundary
theta_clamped = Function(theta_FS) # default value is 0
clamped_dofs_theta = locate_dofs_topological((naghdi_shell_FS.sub(1), theta_FS), fdim, facet_tag.find(1))
bc_clamped_theta = dirichletbc(theta_clamped, clamped_dofs_theta, naghdi_shell_FS.sub(1))

bcs = [bc_clamped_u, bc_clamped_theta]

# %% [markdown]
# ## Newton solver

# %% [markdown]
# target point

# %%

bb_tree = dolfinx.geometry.bb_tree(mesh, 2)
bb_point = np.array([[length, 0.0, 0.0]], dtype=np.float64)

# Find the leaf that the target point is in
bb_cell_candidates = dolfinx.geometry.compute_collisions_points(bb_tree, bb_point)

# Find the cell from the leaf that contains the target point
bb_cells = dolfinx.geometry.compute_colliding_cells(
    mesh, bb_cell_candidates, bb_point)

# %% [markdown]
# set up of solver

# %%
problem = NonlinearProblem(Residual, q_func, bcs, Jacobian)
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
opts[f"{option_prefix}ksp_type"] = "preonly"
#opts[f"{option_prefix}ksp_type"] = "cg"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"


ksp.setFromOptions()

# %%
M_max = 2.0*np.pi*E.value*t.value**3/(12.0*length)
nstep = 20
Ms = np.linspace(0.0, M_max, nstep)

if mesh.comm.rank == 0:
    u1_list = np.zeros(nstep)
    u3_list = np.zeros(nstep)
    
    
    
q_func.x.array[:] = 0.0
#log.set_log_level(log.LogLevel.INFO)
#log.set_log_level(log.LogLevel.OFF)

for i, M_curr in enumerate(Ms):
    M_right.value = M_curr
    n, converged = solver.solve(q_func)
    assert (converged)
    q_func.x.scatter_forward()
    if mesh.comm.rank == 0:
        print(f"Load step {i:d}, Number of iterations: {n:d}, Load: {M_curr:.2f}", flush=True)
    
    # calculate u3 at the point load
    u1_bb = None
    u3_bb = None
    u1_func = q_func.sub(0).sub(0).collapse()
    u3_func = q_func.sub(0).sub(2).collapse()
    if len(bb_cells.array) > 0:
        u1_bb = u1_func.eval(bb_point, bb_cells.array[0])[0]
        u3_bb = u3_func.eval(bb_point, bb_cells.array[0])[0]
    u1_bb = mesh.comm.gather(u1_bb, root=0)
    u3_bb = mesh.comm.gather(u3_bb, root=0)
    if mesh.comm.rank == 0:
        for u1 in u1_bb:
            if u1 is not None:
                u1_list[i] = u1
                break
            
        for u3 in u3_bb:
            if u3 is not None:
                u3_list[i] = u3
                break

# %% [markdown]
# Write outputs

# %%
u_P2B3 = q_func.sub(0).collapse()
theta_P2 = q_func.sub(1).collapse()

# Interpolate phi in the [P2]³ Space
phi_FS = FunctionSpace(mesh, VectorElement("Lagrange", ufl.triangle, degree = 2, dim = 3))
phi_expr = Expression(phi0_ufl + u_P2B3, phi_FS.element.interpolation_points())
phi_func = Function(phi_FS)
phi_func.interpolate(phi_expr)

# Interpolate u in the [P2]³ Space
u_P2 = Function(phi_FS)
u_P2.interpolate(u_P2B3)


with dolfinx.io.VTXWriter(mesh.comm, results_folder/"u_naghdi.bp", [u_P2]) as vtx:
     vtx.write(0)

with dolfinx.io.VTXWriter(mesh.comm, results_folder/"theta_naghdi.bp", [theta_P2]) as vtx:
     vtx.write(0)

with dolfinx.io.VTXWriter(mesh.comm, results_folder/"phi_naghdi.bp", [phi_func]) as vtx:
     vtx.write(0)

if mesh.comm.rank == 0:
    Ms_analytical = np.linspace(1E-3, 1.0, 100)
    vs = 12.0*(np.sin(2.0*np.pi*Ms_analytical)/(2.0*np.pi*Ms_analytical) - 1.0)
    ws = -12.0*(1.0 - np.cos(2.0*np.pi*Ms_analytical))/(2.0*np.pi*Ms_analytical)

    fig = plt.figure(figsize=(5.0, 5.0/1.648))
    plt.plot(Ms_analytical, vs/length, "-", label="$v/L$")
    plt.plot(Ms/M_max, u1_list/length, "x", label="$v_h/L$")
    plt.plot(Ms_analytical, ws/length, "--", label="$w/L$")
    plt.plot(Ms/M_max, u3_list/length, "o", label="$w_h/L$")
    plt.xlabel(r"$M/M_{\mathrm{max}}$")
    plt.ylabel("normalised displacement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_folder/"comparisons.png")


