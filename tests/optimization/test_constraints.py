"""Tests for swing-trajectory feasibility margin evaluation.

Validates the fully-feasible case, an isolated violation for each margin
category (tension, load factor, web length min/max, radial speed) checked
against its exact analytic value, shape-mismatch rejection, and a direct
integration with `solve_ivp` cross-checked against the `web_tension` oracle
used elsewhere in the test suite.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from webswing.config import PhysicalParameters, SwingConstraints
from webswing.dynamics.swing import swing_state_derivative, web_tension
from webswing.optimization.constraints import swing_constraint_margins

PARAMS = PhysicalParameters(mass=1.0, gravity=9.80665)


def zero_control(t: float) -> float:
    return 0.0


def make_constraints(**overrides) -> SwingConstraints:
    defaults = dict(tension_max=1000.0, load_factor_max=1000.0, ell_min=1.0, ell_max=10.0, nu_max=4.0)
    defaults.update(overrides)
    return SwingConstraints(**defaults)


# --- fully feasible ------------------------------------------------------------


def test_fully_feasible_trajectory_reports_zero_violation() -> None:
    times = np.array([0.0, 0.5, 1.0])
    states = np.array(
        [
            [0.1, 0.0, 2.0, 0.0],
            [0.2, 0.1, 2.0, 0.0],
            [0.3, 0.2, 2.0, 0.0],
        ]
    )
    margins = swing_constraint_margins(times, states, PARAMS, zero_control, make_constraints())
    assert margins.is_feasible is True
    assert margins.max_violation == 0.0


# --- isolated tension violation --------------------------------------------------


def test_tension_upper_violation_matches_analytic_value() -> None:
    constraints = make_constraints(tension_max=50.0, load_factor_max=10.0)
    theta, omega, ell = 0.0, 5.0, 2.0
    states = np.array([[theta, omega, ell, 0.0]])
    times = np.array([0.0])

    margins = swing_constraint_margins(times, states, PARAMS, zero_control, constraints)

    expected_tension = web_tension(theta, omega, ell, 0.0, PARAMS)
    assert margins.tension_upper[0] == pytest.approx(50.0 - expected_tension, rel=1e-12)
    assert margins.is_feasible is False
    assert margins.max_violation == pytest.approx(expected_tension - 50.0, rel=1e-12)


def test_tension_lower_margin_equals_tension_value() -> None:
    constraints = make_constraints()
    theta, omega, ell = 0.3, 0.5, 3.0
    states = np.array([[theta, omega, ell, 0.0]])
    times = np.array([0.0])
    margins = swing_constraint_margins(times, states, PARAMS, zero_control, constraints)
    expected_tension = web_tension(theta, omega, ell, 0.0, PARAMS)
    assert margins.tension_lower[0] == pytest.approx(expected_tension, rel=1e-12)


# --- isolated load-factor violation ----------------------------------------------


def test_load_factor_violation_matches_analytic_value() -> None:
    constraints = make_constraints(tension_max=1000.0, load_factor_max=0.5)
    theta, omega, ell = 0.0, 0.0, 2.0
    states = np.array([[theta, omega, ell, 0.0]])
    times = np.array([0.0])

    margins = swing_constraint_margins(times, states, PARAMS, zero_control, constraints)

    expected_tension = web_tension(theta, omega, ell, 0.0, PARAMS)
    expected_load_factor = expected_tension / (PARAMS.mass * PARAMS.gravity)
    assert margins.load_factor[0] == pytest.approx(0.5 - expected_load_factor, rel=1e-12)
    assert margins.is_feasible is False


# --- isolated web-length violations ----------------------------------------------


def test_length_lower_violation() -> None:
    constraints = make_constraints()
    states = np.array([[0.0, 0.0, 0.5, 0.0]])
    times = np.array([0.0])
    margins = swing_constraint_margins(times, states, PARAMS, zero_control, constraints)
    assert margins.length_lower[0] == pytest.approx(0.5 - 1.0, rel=1e-12)
    assert margins.is_feasible is False
    assert margins.max_violation == pytest.approx(0.5, rel=1e-12)


def test_length_upper_violation() -> None:
    constraints = make_constraints()
    states = np.array([[0.0, 0.0, 11.0, 0.0]])
    times = np.array([0.0])
    margins = swing_constraint_margins(times, states, PARAMS, zero_control, constraints)
    assert margins.length_upper[0] == pytest.approx(10.0 - 11.0, rel=1e-12)
    assert margins.is_feasible is False
    assert margins.max_violation == pytest.approx(1.0, rel=1e-12)


# --- isolated radial-speed violation ---------------------------------------------


def test_radial_speed_violation_uses_absolute_value() -> None:
    constraints = make_constraints()
    states_pos = np.array([[0.0, 0.0, 2.0, 6.0]])
    states_neg = np.array([[0.0, 0.0, 2.0, -6.0]])
    times = np.array([0.0])
    margins_pos = swing_constraint_margins(times, states_pos, PARAMS, zero_control, constraints)
    margins_neg = swing_constraint_margins(times, states_neg, PARAMS, zero_control, constraints)
    assert margins_pos.radial_speed[0] == pytest.approx(4.0 - 6.0, rel=1e-12)
    assert margins_neg.radial_speed[0] == pytest.approx(4.0 - 6.0, rel=1e-12)
    assert margins_pos.is_feasible is False
    assert margins_neg.is_feasible is False


# --- shape validation -------------------------------------------------------------


def test_rejects_mismatched_leading_dimensions() -> None:
    times = np.array([0.0, 1.0])
    states = np.array([[0.0, 0.0, 2.0, 0.0]])
    with pytest.raises(ValueError):
        swing_constraint_margins(times, states, PARAMS, zero_control, make_constraints())


def test_rejects_wrong_state_width() -> None:
    times = np.array([0.0])
    states = np.array([[0.0, 0.0, 2.0]])
    with pytest.raises(ValueError):
        swing_constraint_margins(times, states, PARAMS, zero_control, make_constraints())


# --- integration with a real solve_ivp trajectory ---------------------------------


def test_margins_over_integrated_trajectory_match_web_tension_oracle() -> None:
    constraints = make_constraints()
    z0 = np.array([0.4, 0.0, 2.0, 0.0])
    t_eval = np.linspace(0.0, 1.0, 11)
    sol = solve_ivp(
        swing_state_derivative,
        (0.0, 1.0),
        z0,
        args=(PARAMS, zero_control),
        t_eval=t_eval,
        rtol=1e-12,
        atol=1e-13,
        method="RK45",
    )
    assert sol.success
    states = sol.y.T

    margins = swing_constraint_margins(t_eval, states, PARAMS, zero_control, constraints)

    for i, t in enumerate(t_eval):
        theta, omega, ell, _nu = states[i]
        expected_tension = web_tension(theta, omega, ell, 0.0, PARAMS)
        assert margins.tension_upper[i] == pytest.approx(constraints.tension_max - expected_tension, rel=1e-10)
        assert margins.tension_lower[i] == pytest.approx(expected_tension, rel=1e-10)
