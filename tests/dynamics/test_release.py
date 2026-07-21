"""Tests for the attached-to-ballistic release transition.

Validates the position and velocity mappings against their defining
formulas, checks the analytic velocity against a finite-difference
derivative of the position mapping along a real integrated swing
trajectory, and confirms the packaged release state matches both.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from webswing.config import PhysicalParameters
from webswing.dynamics.release import (
    attached_position,
    attached_velocity,
    release_to_ballistic_state,
)
from webswing.dynamics.swing import swing_state_derivative
from webswing.exceptions import NonPositiveWebLengthError

PARAMS = PhysicalParameters(mass=1.0, gravity=9.80665)
X_ANCHOR, Y_ANCHOR = 10.0, 20.0


def test_position_matches_defining_formula() -> None:
    theta, ell = 0.6, 3.0
    x, y = attached_position(theta, ell, X_ANCHOR, Y_ANCHOR)
    assert x == pytest.approx(X_ANCHOR + ell * math.sin(theta), rel=1e-12)
    assert y == pytest.approx(Y_ANCHOR - ell * math.cos(theta), rel=1e-12)


def test_velocity_matches_defining_formula() -> None:
    theta, omega, ell, nu = 0.6, 0.4, 3.0, 0.2
    vx, vy = attached_velocity(theta, omega, ell, nu)
    expected_vx = nu * math.sin(theta) + ell * omega * math.cos(theta)
    expected_vy = -nu * math.cos(theta) + ell * omega * math.sin(theta)
    assert vx == pytest.approx(expected_vx, rel=1e-12)
    assert vy == pytest.approx(expected_vy, rel=1e-12)


def test_velocity_matches_finite_difference_of_position_along_trajectory() -> None:
    # Nonzero, non-constant control so l_dot and l_ddot both contribute to
    # the position derivative, exercising the full product-rule expansion
    # rather than only the fixed-length special case.
    def control(t: float) -> float:
        return 0.05 * math.sin(t)

    z0 = np.array([0.5, 0.3, 2.0, -0.1])
    t_span = (0.0, 1.0)
    t_probe = np.linspace(0.2, 0.8, 5)
    dt = 1.0e-6

    sol = solve_ivp(
        swing_state_derivative,
        t_span,
        z0,
        args=(PARAMS, control),
        dense_output=True,
        rtol=1e-12,
        atol=1e-13,
        method="RK45",
    )
    assert sol.success

    for t in t_probe:
        theta_m, omega_m, ell_m, nu_m = sol.sol(t - dt)
        theta_p, omega_p, ell_p, nu_p = sol.sol(t + dt)
        theta_0, omega_0, ell_0, nu_0 = sol.sol(t)

        x_m, y_m = attached_position(theta_m, ell_m, X_ANCHOR, Y_ANCHOR)
        x_p, y_p = attached_position(theta_p, ell_p, X_ANCHOR, Y_ANCHOR)
        vx_fd = (x_p - x_m) / (2.0 * dt)
        vy_fd = (y_p - y_m) / (2.0 * dt)

        vx_analytic, vy_analytic = attached_velocity(theta_0, omega_0, ell_0, nu_0)

        # Central-difference truncation error is O(dt^2); at dt=1e-6 this is
        # far below 1e-6 in magnitude, so a 1e-6 absolute tolerance is a
        # genuine check of the analytic formula, not a vacuous one.
        assert vx_fd == pytest.approx(vx_analytic, abs=1e-6)
        assert vy_fd == pytest.approx(vy_analytic, abs=1e-6)


def test_release_to_ballistic_state_matches_component_mappings() -> None:
    z = np.array([0.6, 0.4, 3.0, 0.2])
    q = release_to_ballistic_state(z, X_ANCHOR, Y_ANCHOR)
    x, y = attached_position(z[0], z[2], X_ANCHOR, Y_ANCHOR)
    vx, vy = attached_velocity(z[0], z[1], z[2], z[3])
    np.testing.assert_allclose(q, [x, y, vx, vy], rtol=0.0, atol=0.0)


def test_release_rejects_nonpositive_web_length() -> None:
    z = np.array([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(NonPositiveWebLengthError):
        release_to_ballistic_state(z, X_ANCHOR, Y_ANCHOR)
