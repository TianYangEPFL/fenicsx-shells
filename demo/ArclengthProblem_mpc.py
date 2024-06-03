from __future__ import annotations

import typing

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np

import ufl
from dolfinx import la as _la
from dolfinx import cpp as _cpp
from dolfinx import fem as _fem

from dolfinx.fem import Constant
from dolfinx.fem.bcs import DirichletBC
from dolfinx.fem.forms import Form
from dolfinx.fem.forms import form as _create_form
from dolfinx.fem.function import Function as _Function
import dolfinx.fem.petsc

from dolfinx_mpc import (MultiPointConstraint, assemble_matrix, assemble_vector,
                         apply_lifting, create_sparsity_pattern)
import dolfinx_mpc

class ArclengthProblem:
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
        ps_cells: list[int] = [],
        ps_basis_values: np.ndarray = [],
        ps_direction: int = 2,
        ps_cells_B: list[int] = [],
        ps_basis_values_B: np.ndarray = [],
        ps_direction_B: int = 2,
        form_compiler_options: typing.Optional[dict] = None,
        jit_options: typing.Optional[dict] = None,
    ):
        pass