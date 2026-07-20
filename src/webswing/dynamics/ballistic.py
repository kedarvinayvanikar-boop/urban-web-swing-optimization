"""Unconstrained projectile dynamics for post-release ballistic motion.

Implements dq/dt for q = [x, y, vx, vy] under constant gravitational
acceleration with no aerodynamic drag:

    x_dot = vx,  y_dot = vy,  vx_dot = 0,  vy_dot = -g.

The closed-form solution
    x(t)  = x0 + vx0*t
    y(t)  = y0 + vy0*t - (1/2)*g*t^2
    vx(t) = vx0
    vy(t) = vy0 - g*t
is the primary oracle for validating this ODE right-hand side; it is not
used at runtime, since the numerical integration path (`solve_ivp`) is the
one exercised by event detection and trajectory assembly.
"""

from __future__ import annotations

import numpy as np

from webswing.config import PhysicalParameters


def ballistic_state_derivative(t: float, q: np.ndarray, params: PhysicalParameters) -> np.ndarray:
    """Evaluate the state derivative for unconstrained projectile motion.

    Parameters
    ----------
    t : float
        Simulation time, in seconds. Unused; retained so the function
        matches the `solve_ivp` callable signature `fun(t, y)` and the
        signature of `swing_state_derivative`.
    q : np.ndarray, shape (4,)
        State vector [x, y, vx, vy], in meters and meters/second.
    params : PhysicalParameters
        Supplies the gravitational acceleration magnitude.

    Returns
    -------
    np.ndarray, shape (4,)
        Derivative [vx, vy, 0.0, -g].
    """
    _, _, vx, vy = q
    return np.array([vx, vy, 0.0, -params.gravity], dtype=float)
