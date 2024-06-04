from __future__ import annotations

import typing

import dolfinx_mpc.cpp
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import ufl
from dolfinx import la as _la
from dolfinx import fem as _fem

from dolfinx.fem import Constant
from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.forms import Form
from dolfinx.fem.forms import form as _create_form
from dolfinx.fem.function import Function as _Function
import dolfinx.fem.petsc

from dolfinx_mpc import MultiPointConstraint
import dolfinx_mpc

class ArclengthMPCProblem:
    def __init__(
        self,
        Fint: ufl.form.Form,
        Fext: ufl.form.Form,
        u: _Function,
        mpc: MultiPointConstraint,
        lmbda0: float = 0.0,
        bcs: list[DirichletBC] = [],
        J: ufl.form.Form = None,
        petsc_options: typing.Optional[dict] = None,
        ps_cells: list[list[int]] = [],
        ps_basis_values: list[np.ndarray] = [],
        ps_dirs: list[int] = 2,
        ps_scales: list[float] = 1.0,
        form_compiler_options: typing.Optional[dict] = None,
        jit_options: typing.Optional[dict] = None,
    ):
        # Solution functions (original FS) in i, i+1 and i+2 steps
        self.u0 = u.copy()
        self.u1 = u.copy()
        self.u = u
        self._x = _la.create_petsc_vector_wrap(self.u.x)
        
        # load parameter in i, i+1 and i+2 steps
        self._lmbda0 = Constant(self.u.function_space.mesh, lmbda0)
        self._lmbda1 = Constant(self.u.function_space.mesh, lmbda0)
        self._lmbda = Constant(self.u.function_space.mesh, lmbda0)

        # Point source cell and basis values
        self.ps_cells = ps_cells
        self.ps_basis_values = ps_basis_values
        self.ps_dirs = ps_dirs
        self.ps_scales = ps_scales
        
        # Compile the forms
        self._Lext = _create_form(Fext, form_compiler_options=form_compiler_options, jit_options=jit_options)
        self._L = _create_form(Fint-self._lmbda*Fext, form_compiler_options=form_compiler_options, jit_options=jit_options)
        
        if J is None:
            V = u.function_space
            du = ufl.TrialFunction(V)
            J = ufl.derivative(Fint, u, du)
        self._a = _create_form(J, form_compiler_options=form_compiler_options, jit_options=jit_options)
        
        # MPC 
        if not mpc.finalized:
            raise RuntimeError("The multi point constraint has to be finalized before calling initializer")
        self.mpc = mpc
        self.bcs = bcs
        self.u_mpc = _Function(mpc.function_space)
        
        # Create MPC matrix and vector
        self._A = dolfinx_mpc.cpp.mpc.create_matrix(self._a._cpp_object, self.mpc._cpp_object)
        self._b = _la.create_petsc_vector(self.mpc.function_space.dofmap.index_map, self.mpc.function_space.dofmap.index_map_bs)
        self._bext = _la.create_petsc_vector(self.mpc.function_space.dofmap.index_map, self.mpc.function_space.dofmap.index_map_bs)
        
        # Tolerance and maximum iteration for solvers 
        self.tol = 1e-8
        self.max_it = 100
        
        # Arc length control parameter
        self.eta = 1.0
        self.ds = 0.0
        self.s0 = 0.0
        self.s1 = 0.0
        self.s = 0.0
        
        # Create the PETSc solver
        self._solver = PETSc.KSP().create(self.u.function_space.mesh.comm)
        self._solver.setOperators(self._A)
        
        # Give PETSc solver options a unique prefix
        problem_prefix = f"dolfinx_mpc_solve_{id(self)}"
        self._solver.setOptionsPrefix(problem_prefix)
        
        # Set PETSc options
        opts = PETSc.Options()
        opts.prefixPush(problem_prefix)
        if petsc_options is not None:
            for k, v in petsc_options.items():
                opts[k] = v
        opts.prefixPop()
        self._solver.setFromOptions()
        
        # Set matrix and vector PETSc options
        self._A.setOptionsPrefix(problem_prefix)
        self._A.setFromOptions()
        self._b.setOptionsPrefix(problem_prefix)
        self._b.setFromOptions()
        self._bext.setOptionsPrefix(problem_prefix)
        self._bext.setFromOptions()
        
    def __del__(self):
        self._solver.destroy()
        self._A.destroy()
        self._b.destroy()
        self._bext.destroy()
        self._x.destroy()
    
    @property
    def Lext(self) -> Form:
        """Compiled linear form (the external force form)"""
        return self._Lext

    @property
    def L(self) -> Form:
        """Compiled linear form (the residual form)"""
        return self._L
    
    @property
    def a(self) -> Form:
        """Compiled bilinear form (the Jacobian form)"""
        return self._a
    
    @property
    def A(self) -> PETSc.Mat:
        """Jacobian Matrix"""
        return self._A
    
    @property
    def b(self) -> PETSc.Vec:
        """Residual vector"""
        return self._b
    
    @property
    def bext(self) -> PETSc.Vec:
        """External force vector"""
        return self._bext
    
    @property
    def lmbda(self) -> float:
        """Initial value of the load parameter"""
        return self._lmbda.value
    
    @property
    def solver(self) -> PETSc.KSP:
        """Linear solver object"""
        return self._solver
    
    def assemble_Residual(self, x: PETSc.Vec, b: PETSc.Vec) -> None:
        """Assemble the residual F into the vector b.

        Args:
            x: The vector containing the latest solution
            b: Vector to assemble the residual into

        """
        # Reset the residual vector
        with b.localForm() as b_local:
            b_local.set(0.0)
        dolfinx_mpc.assemble_vector(self._L, self.mpc, b)

        # Add point source term
        for ps_cell, ps_basis_value, ps_dir, ps_scale in zip(self.ps_cells, self.ps_basis_values, self.ps_dirs, self.ps_scales):
            if len(ps_cell) > 0:
                dofs = self.mpc.function_space.sub(0).sub(ps_dir).dofmap.cell_dofs(ps_cell[0])
                with b.localForm() as b_local:
                    b_local.setValuesLocal(dofs, -ps_basis_value*self.lmbda*ps_scale,
                                           addv=PETSc.InsertMode.ADD_VALUES)
        
        # Apply boundary condition
        dolfinx_mpc.apply_lifting(b, [self._a], bcs=[self.bcs], constraint=self.mpc, x0=[x], scale=-1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, self.bcs, x, -1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
        
    def assemble_Fext(self, bext: PETSc.Vec) -> None:
        """Assemble the external force into vector bext.

        Args:
            x: The vector containing the latest solution
            b: Vector to assemble the external force into

        """
        # Reset the Fext vector
        with bext.localForm() as bext_local:
            bext_local.set(0.0)
            
        dolfinx_mpc.assemble_vector(self._Lext, self.mpc, bext)
        
        # Add point source term
        for ps_cell, ps_basis_value, ps_dir, ps_scale in zip(self.ps_cells, self.ps_basis_values, self.ps_dirs, self.ps_scales):
            if len(ps_cell) > 0:
                dofs = self.mpc.function_space.sub(0).sub(ps_dir).dofmap.cell_dofs(ps_cell[0])
                with bext.localForm() as bext_local:
                    bext_local.setValuesLocal(dofs, ps_basis_value*ps_scale,
                                           addv=PETSc.InsertMode.ADD_VALUES)
                    
                    
        # Apply homogeneous boundary condition
        dolfinx_mpc.apply_lifting(bext, [self._a], bcs=[self.bcs], constraint=self.mpc, scale=0.0)
        bext.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(bext, self.bcs, scale=0.0)
        bext.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)
    
    def assemble_Jacobian(self, x: PETSc.Vec, A: PETSc.Mat) -> None:
        """Assemble the Jacobian matrix.

        Args:
            x: The vector containing the latest solution

        """
        A.zeroEntries()
        dolfinx_mpc.assemble_matrix(self._a, self.mpc, bcs=self.bcs, A=A)
        A.assemble()

    def reset(self) -> None:
        """Reset the arclength parameter"""
        self.s0 = 0.0
        self.s1 = 0.0
        self.s = 0.0
        self.ds = 0.0
        self._lmbda0.value = 0.0
        self._lmbda1.value = 0.0
        self._lmbda.value = 0.0
        self.u0.x.array[:] = 0.0
        self.u1.x.array[:] = 0.0
        self.u.x.array[:] = 0.0
    
    def update_mpc(self, dx: PETSc.Vec, x: PETSc.Vec) -> None:
        self.u_mpc.x.petsc_vec.array = x.array_r
        self.u_mpc.x.petsc_vec.axpy(-1.0, dx)
        self.u_mpc.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,  
            mode=PETSc.ScatterMode.FORWARD,  
        ) 
        self.mpc.homogenize(self.u_mpc)
        self.mpc.backsubstitution(self.u_mpc)
        x.array = self.u_mpc.x.petsc_vec.array_r
        x.ghostUpdate(
            addv=PETSc.InsertMode.INSERT,  
            mode=PETSc.ScatterMode.FORWARD,  
        ) 
        
    def NewtonStep(self, Pred_control = False, lmbda2: float = 0.0) -> typing.Tuple[int, bool]:
        """Perform a Newton step.

        Args:
            lmbda2: The next value of the arclength parameter
            Pred_control: with Prediction or not

        """      
        # Increment of the new solution and backups of the last solution
        du = _Function(self.u.function_space)
        u_backup = self.u.copy()
        lmbda_backup = self._lmbda.value.copy()
        
        # With Prediction:
        if Pred_control:
            lmbda_pred, u_pred = self.Second_order_prediction(self.s + self.ds)
            self.u.x.array[:] = u_pred.x.array
            self.u.x.scatter_forward()
            self._lmbda.value = lmbda_pred
        else:
            self._lmbda.value = lmbda2
            
        i = int(0)
        while i < self.max_it:
            # Assemble the residual and Jacobian
            self.assemble_Jacobian(self._x, self._A)
            self.assemble_Residual(self._x, self._b)
            
            # Solve the linear system and update the ghost values in the solution vector
            self._solver.solve(self._b, du.x.petsc_vec)
            du.x.scatter_forward()
            
            # update the solution with mpc 
            self.update_mpc(du.x.petsc_vec, self._x)
            
            # Check the convergence
            du_norm_l2 = _la.norm(du.x, _la.Norm.l2)
            if du_norm_l2 < self.tol:
                converged = True
                break
            else:
                i += 1
                if i == self.max_it:
                    converged = False
                
        if converged:
            self._lmbda0.value = self._lmbda1.value
            self._lmbda1.value = lmbda_backup
            self.u0.x.array[:] = self.u1.x.array 
            self.u1.x.array[:] = u_backup.x.array
            
            # update the arclength parameter
            ds0 = self.compute_arclength_increment()
            if not Pred_control:
                self.ds = ds0
                
            self.s0 = self.s1
            self.s1 = self.s
            self.s += ds0
        else:
            self._lmbda.value = lmbda_backup
            self.u.x.array[:] = u_backup.x.array
            print("Warning: max iterations reached in Newton step", flush=True)
        
        return i+1, converged
    
    def compute_arclength_increment(self) -> float:
        """Compute the arclength increment from the last step"""
        self.assemble_Fext(self._bext)
        Du = _Function(self.u.function_space)
        Du.x.array[:] = self.u.x.array - self.u1.x.array
        Du.x.scatter_forward()
        Dlmbda = self._lmbda.value - self._lmbda1.value
        ds = np.sqrt(Du.x.petsc_vec.dot(Du.x.petsc_vec) + 
                     self.eta * Dlmbda**2 * self._bext.dot(self._bext))
        return ds
    
    def Second_order_prediction(self, s_pred) -> typing.Tuple[np.ndarray, _Function]:
        """Predict the solution in the next step using the second order extrapolation"""
        # Lagrange basis functions
        c0 = (s_pred - self.s1) * (s_pred - self.s) / ((self.s0 - self.s1) * (self.s0 - self.s))
        c1 = (s_pred - self.s0) * (s_pred - self.s) / ((self.s1 - self.s0) * (self.s1 - self.s))
        c2 = (s_pred - self.s0) * (s_pred - self.s1) / ((self.s - self.s0) * (self.s - self.s1))
        
        # Predict the lmbda value
        lmbda_pred = c0 * self._lmbda0.value + c1 * self._lmbda1.value + c2 * self._lmbda.value
        u_pred = _Function(self.u.function_space)
        u_pred.x.array[:] = c0 * self.u0.x.array + c1 * self.u1.x.array + c2 * self.u.x.array
        
        return lmbda_pred, u_pred