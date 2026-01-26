"""
p-Laplacian Equation: Aronsson Example (Homogeneous, Dirichlet)
Using NVIDIA PhysicsNeMo-sym

This script solves the p-Laplacian equation with parameterized p:
    Δ_p u = div(|∇u|^(p-2) ∇u) = 0

Expanded form:
    |∇u|^(p-4) * [(p-1)*(u_x² u_xx + u_y² u_yy) + 2(p-2)*u_x*u_y*u_xy + u_x² u_yy + u_y² u_xx] = 0

Problem Setup:
- Domain: 2D square [-1, 1] × [-1, 1]
- Boundary condition: u(x,y) = |x|^(4/3) - |y|^(4/3) on ∂Ω
- Parameter: p ∈ [2, 100] as a training variable
- Validation: Compare p=100 solution with infinity-Laplacian solution

Note: The Aronsson solution u(x,y) = |x|^(4/3) - |y|^(4/3) is the exact solution
for the infinity-Laplacian and serves as an approximation for large p.

Installation:
    pip install nvidia-physicsnemo
    pip install Cython
    pip install nvidia-physicsnemo-sym --no-build-isolation
"""

import numpy as np
from sympy import Symbol, Function, Abs, Rational, sqrt

import physicsnemo.sym
from physicsnemo.sym.hydra import instantiate_arch
from physicsnemo.sym.hydra.config import PhysicsNeMoConfig

from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain
from physicsnemo.sym.geometry.primitives_2d import Rectangle
from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.domain.validator import PointwiseValidator
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pde import PDE
from physicsnemo.sym.utils.io import ValidatorPlotter

import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'jet'

print("Using physicsnemo.sym for p-Laplacian")


# ============================================================================
# Custom p-Laplacian PDE with parameterized p
# ============================================================================

class PLaplacian(PDE):
    name = "PLaplacian"
    
    def __init__(self):
        # Define coordinates and parameter
        x = Symbol("x")
        y = Symbol("y")
        p = Symbol("p")
        
        # Define input variables (including p as parameter)
        input_variables = {"x": x, "y": y, "p": p}
        
        # Define the solution function u(x, y, p)
        u = Function("u")(*input_variables)
        
        # First derivatives
        u_x = u.diff(x)
        u_y = u.diff(y)
        
        # Second derivatives
        u_xx = u.diff(x, x)
        u_yy = u.diff(y, y)
        u_xy = u.diff(x, y)
        
        # For numerical stability, we use the normalized form (without |∇u|^(p-4) factor)
        p_laplacian = (
            (p - 1) * u_x**2 * u_xx + 
            (p - 1) * u_y**2 * u_yy + 
            2 * (p - 2) * u_x * u_y * u_xy + 
            u_x**2 * u_yy + 
            u_y**2 * u_xx
        )
        
        # Set up equations
        self.equations = {}
        self.equations["p_laplacian"] = p_laplacian


# ============================================================================
# Exact Solutions
# ============================================================================

def aronsson_solution_sympy(x_sym, y_sym):
    """Get the Aronsson solution as a SymPy expression for boundary conditions."""
    return Abs(x_sym)**Rational(4, 3) - Abs(y_sym)**Rational(4, 3)


def aronsson_solution_numpy(X, Y):
    """Get the Aronsson solution as numpy function for validation."""
    return np.abs(X)**(4/3) - np.abs(Y)**(4/3)


# ============================================================================
# Main Solver Function
# ============================================================================

@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    # Initialize p-Laplacian PDE
    p_lap = PLaplacian()
    
    # Create neural network with p as input
    net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("p")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    
    # Create nodes for the solver
    nodes = p_lap.make_nodes() + [net.make_node(name="p_laplacian_network")]

    # Create geometry: square [-1, 1] × [-1, 1]
    geo = Rectangle((-1.0, -1.0), (1.0, 1.0))
    
    # Define coordinates and parameter symbols
    x_sym = Symbol("x")
    y_sym = Symbol("y")
    p_sym = Symbol("p")
    sdf_sym = Symbol("sdf")
    
    # Define p parameterization for sampling
    p_range = {
        p_sym: lambda batch_size: np.random.uniform(2, 10, (batch_size, 1))
    }
    # Create domain
    domain = Domain()

    exact_bc =Abs(x_sym)**Rational(4, 3) - Abs(y_sym)**Rational(4, 3)
    
    # Add boundary constraint: u = |x|^(4/3) - |y|^(4/3)
    BC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": exact_bc},
        lambda_weighting={"u": 1.0},
        batch_size=cfg.batch_size.BC,
        parameterization=p_range,
    )
    domain.add_constraint(BC, "BC")
    
    # Add interior constraint: p_laplacian = 0
    # Use SDF-based weighting to reduce influence near x=0 and y=0
    # where the Aronsson solution has singular gradients
    eps = 0.01
    dist_x = sqrt(x_sym**2 + eps**2)
    dist_y = sqrt(y_sym**2 + eps**2)
    axis_weight = dist_x * dist_y  # Reduces weight near x=0 and y=0
    
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"p_laplacian": 0},
        batch_size=cfg.batch_size.interior,
        lambda_weighting={
            "p_laplacian": sdf_sym * axis_weight,
        },
        parameterization=p_range,
    )
    domain.add_constraint(interior, "interior")
    
    # Add validation data
    n_val = 50
    x_vals = np.linspace(-0.98, 0.98, n_val)
    y_vals = np.linspace(-0.98, 0.98, n_val)
    X, Y = np.meshgrid(x_vals, y_vals)
    X = np.expand_dims(X.flatten(), axis=-1)
    Y = np.expand_dims(Y.flatten(), axis=-1)
    
    u_exact = np.abs(X)**(4/3) - np.abs(Y)**(4/3)
    
    # Add validators for different p values
    for p_val in [2, 5, 10]:
        P_test = np.full_like(X, float(p_val))
        invar_p = {"x": X, "y": Y, "p": P_test}
        # Note: The exact solution for finite p is not known analytically,
        # so we use Aronsson as reference
        validator_p = PointwiseValidator(
            nodes=nodes,
            invar=invar_p,
            true_outvar={"u": u_exact},  # Using Aronsson as reference
            batch_size=128,
            plotter=ValidatorPlotter(),
        )
        domain.add_validator(validator_p, f"validation_p{p_val}")
    
    # Create and run solver
    slv = Solver(cfg, domain)
    slv.solve()


if __name__ == "__main__":
    run()

