"""Tests for the controlled variable-length pendulum dynamics.

Validates state-derivative structure, reduction to the fixed-length simple
pendulum, small-angle linearization, mechanical-energy conservation without
control, the web-tension force balance against hand-computable states, and
rejection of a degenerate (non-positive) web length.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from webswing.config import PhysicalParameters
from webswing.dynamics.swing import swing_state_derivative, web_tension
from webswing.exceptions import NonPositiveWebLengthError

PARAMS = PhysicalParameters(mass=1.0, gravity=9.80665)


def zero_control(t: float) -> float:
    return 0.0


def test_state_derivative_shape_and_dtype() -> None:
    z = np.array([0.3, 0.1, 2.0, 0.05])
    dz = swing_state_derivative(0.0, z, PARAMS, zero_control)
    assert dz.shape == (4,)
    assert dz.dtype == np.float64


def test_fixed_length_reduces_to_simple_pendulum() -> None:
    theta, ell = 0.4, 2.0
    z = np.array([theta, 0.0, ell, 0.0])
    dz = swing_state_derivative(0.0, z, PARAMS, zero_control)
    expected_omega_dot = -(PARAMS.gravity / ell) * math.sin(theta)
    assert dz[1] == pytest.approx(expected_omega_dot, rel=1e-12)
    assert dz[2] == 0.0
    assert dz[3] == 0.0


def test_small_angle_matches_linear_pendulum() -> None:
    theta0, ell = 1.0e-3, 3.0
    omega_n = math.sqrt(PARAMS.gravity / ell)
    z0 = np.array([theta0, 0.0, ell, 0.0])
    t_eval = np.linspace(0.0, 2.0, 50)
    sol = solve_ivp(
        swing_state_derivative,
        (0.0, 2.0),
        z0,
        args=(PARAMS, zero_control),
        t_eval=t_eval,
        rtol=1e-10,
        atol=1e-12,
        method="RK45",
    )
    assert sol.success
    # Nonlinear correction to the linear pendulum is O(theta0^2) ~ 1e-6 relative,
    # so an absolute tolerance well above that and well below the amplitude is
    # a meaningful check rather than a vacuous one.
    theta_linear = theta0 * np.cos(omega_n * t_eval)
    np.testing.assert_allclose(sol.y[0], theta_linear, rtol=0.0, atol=1e-6)


def test_energy_conservation_without_control() -> None:
    theta0, omega0, ell = 0.8, 0.5, 1.5

    def energy(theta: float, omega: float) -> float:
        return 0.5 * PARAMS.mass * ell**2 * omega**2 - PARAMS.mass * PARAMS.gravity * ell * math.cos(theta)

    z0 = np.array([theta0, omega0, ell, 0.0])
    e0 = energy(theta0, omega0)
    sol = solve_ivp(
        swing_state_derivative,
        (0.0, 5.0),
        z0,
        args=(PARAMS, zero_control),
        rtol=1e-12,
        atol=1e-13,
        method="RK45",
    )
    assert sol.success
    theta_f, omega_f = sol.y[0, -1], sol.y[1, -1]
    e_f = energy(theta_f, omega_f)
    # Tolerance reflects RK45 truncation error at rtol=1e-12 over ~2 swing
    # periods, not a relaxed bound chosen to force a pass.
    assert e_f == pytest.approx(e0, rel=1e-6, abs=1e-8)


def test_nonpositive_web_length_raises() -> None:
    z = np.array([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(NonPositiveWebLengthError):
        swing_state_derivative(0.0, z, PARAMS, zero_control)


def test_tension_static_hang_equals_weight() -> None:
    T = web_tension(theta=0.0, omega=0.0, ell=2.0, ell_ddot=0.0, params=PARAMS)
    assert T == pytest.approx(PARAMS.mass * PARAMS.gravity, rel=1e-12)


def test_tension_includes_centripetal_and_control_terms() -> None:
    ell, omega, ell_ddot = 2.0, 1.5, 0.7
    T = web_tension(theta=0.0, omega=omega, ell=ell, ell_ddot=ell_ddot, params=PARAMS)
    expected = PARAMS.mass * (ell * omega**2 + PARAMS.gravity - ell_ddot)
    assert T == pytest.approx(expected, rel=1e-12)


def test_tension_nonpositive_length_raises() -> None:
    with pytest.raises(NonPositiveWebLengthError):
        web_tension(theta=0.0, omega=0.0, ell=-1.0, ell_ddot=0.0, params=PARAMS)
