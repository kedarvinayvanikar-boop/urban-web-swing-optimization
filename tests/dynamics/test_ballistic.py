"""Tests for unconstrained ballistic (projectile) dynamics.

Validates the state-derivative form directly, agreement of the integrated
trajectory with the closed-form projectile solution, and horizontal
velocity conservation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from webswing.config import PhysicalParameters
from webswing.dynamics.ballistic import ballistic_state_derivative

PARAMS = PhysicalParameters(mass=1.0, gravity=9.80665)


def closed_form(t: np.ndarray, q0: np.ndarray) -> np.ndarray:
    x0, y0, vx0, vy0 = q0
    g = PARAMS.gravity
    x = x0 + vx0 * t
    y = y0 + vy0 * t - 0.5 * g * t**2
    vx = np.full_like(t, vx0)
    vy = vy0 - g * t
    return np.vstack([x, y, vx, vy])


def test_derivative_matches_expected_form() -> None:
    q = np.array([1.0, 2.0, 3.0, -4.0])
    dq = ballistic_state_derivative(0.0, q, PARAMS)
    assert dq[0] == 3.0
    assert dq[1] == -4.0
    assert dq[2] == 0.0
    assert dq[3] == pytest.approx(-PARAMS.gravity, rel=1e-15)


def test_integration_matches_closed_form_trajectory() -> None:
    q0 = np.array([0.0, 50.0, 8.0, 6.0])
    t_eval = np.linspace(0.0, 3.0, 31)
    sol = solve_ivp(
        ballistic_state_derivative,
        (0.0, 3.0),
        q0,
        args=(PARAMS,),
        t_eval=t_eval,
        rtol=1e-12,
        atol=1e-12,
        method="RK45",
    )
    assert sol.success
    expected = closed_form(t_eval, q0)
    # RK45 is exact for this constant-coefficient linear system up to solver
    # tolerance, so the comparison tolerance tracks rtol/atol above rather
    # than a loosely chosen number.
    np.testing.assert_allclose(sol.y, expected, rtol=1e-9, atol=1e-9)


def test_horizontal_velocity_conserved() -> None:
    q0 = np.array([0.0, 100.0, 12.5, -3.0])
    sol = solve_ivp(
        ballistic_state_derivative,
        (0.0, 4.0),
        q0,
        args=(PARAMS,),
        rtol=1e-12,
        atol=1e-12,
        method="RK45",
    )
    assert sol.success
    np.testing.assert_allclose(sol.y[2], q0[2], rtol=0.0, atol=1e-9)
