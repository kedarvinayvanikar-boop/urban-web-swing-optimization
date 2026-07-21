"""Tests for swing and ballistic terminal event functions.

Each event is checked directly against its defining formula and its
`.terminal`/`.direction` attributes. Ground-contact events are additionally
exercised inside a real `solve_ivp` run to confirm they trigger integration
termination at the physically correct location/time, since a value-only
check cannot prove the zero-crossing search actually stops the solver.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from webswing.config import BallisticDomain, PhysicalParameters, SwingConstraints
from webswing.dynamics.ballistic import ballistic_state_derivative
from webswing.dynamics.events import (
    assert_state_finite,
    make_ballistic_capture_region_event,
    make_ballistic_ground_impact_event,
    make_ballistic_max_duration_event,
    make_ballistic_x_max_event,
    make_ballistic_x_min_event,
    make_ballistic_y_max_event,
    make_swing_ground_contact_event,
    make_swing_length_max_event,
    make_swing_length_min_event,
    make_swing_load_factor_max_event,
    make_swing_radial_speed_max_event,
    make_swing_release_time_event,
    make_swing_tension_max_event,
    make_swing_tension_nonpositive_event,
)
from webswing.dynamics.swing import swing_state_derivative, web_tension
from webswing.exceptions import NonFiniteStateError

PARAMS = PhysicalParameters(mass=1.0, gravity=9.80665)
CONSTRAINTS = SwingConstraints(tension_max=50.0, load_factor_max=3.0, ell_min=1.0, ell_max=10.0, nu_max=4.0)
DOMAIN = BallisticDomain(x_min=-50.0, x_max=50.0, y_max=100.0, max_duration=20.0)


def zero_control(t: float) -> float:
    return 0.0


# --- Swing events ------------------------------------------------------------


def test_swing_ground_contact_event_matches_position_formula() -> None:
    event = make_swing_ground_contact_event(x_anchor=0.0, y_anchor=5.0)
    theta, ell = 0.3, 4.0
    z = np.array([theta, 0.0, ell, 0.0])
    expected_y = 5.0 - ell * math.cos(theta)
    assert event(0.0, z, PARAMS, zero_control) == pytest.approx(expected_y, rel=1e-12)
    assert event.terminal is True
    assert event.direction == -1.0


def test_swing_ground_contact_event_triggers_in_integration() -> None:
    x_anchor, y_anchor = 0.0, 1.0
    ell = 2.0
    # theta0 chosen so y0 > 0 but the bottom of the swing (theta=0) has
    # y = y_anchor - ell = -1.0 < 0, guaranteeing a crossing en route.
    z0 = np.array([1.2, 0.0, ell, 0.0])
    event = make_swing_ground_contact_event(x_anchor, y_anchor)

    sol = solve_ivp(
        swing_state_derivative,
        (0.0, 5.0),
        z0,
        args=(PARAMS, zero_control),
        events=event,
        rtol=1e-10,
        atol=1e-12,
        method="RK45",
    )
    assert sol.success
    assert len(sol.t_events[0]) == 1
    theta_hit, _omega_hit, ell_hit, _nu_hit = sol.y_events[0][0]
    y_hit = y_anchor - ell_hit * math.cos(theta_hit)
    assert y_hit == pytest.approx(0.0, abs=1e-8)


def test_swing_tension_max_event_matches_web_tension() -> None:
    event = make_swing_tension_max_event(CONSTRAINTS)
    theta, omega, ell = 0.0, 3.0, 2.0
    z = np.array([theta, omega, ell, 0.0])
    tension = web_tension(theta, omega, ell, 0.0, PARAMS)
    assert event(0.0, z, PARAMS, zero_control) == pytest.approx(tension - CONSTRAINTS.tension_max, rel=1e-12)
    assert event.terminal is True
    assert event.direction == 1.0


def test_swing_tension_nonpositive_event_matches_web_tension() -> None:
    event = make_swing_tension_nonpositive_event()
    theta, omega, ell = 0.0, 0.1, 3.0
    z = np.array([theta, omega, ell, 0.0])
    tension = web_tension(theta, omega, ell, 0.0, PARAMS)
    assert event(0.0, z, PARAMS, zero_control) == pytest.approx(tension, rel=1e-12)
    assert event.terminal is True
    assert event.direction == -1.0


def test_swing_load_factor_max_event_matches_tension_over_weight() -> None:
    event = make_swing_load_factor_max_event(CONSTRAINTS)
    theta, omega, ell = 0.0, 2.0, 2.0
    z = np.array([theta, omega, ell, 0.0])
    tension = web_tension(theta, omega, ell, 0.0, PARAMS)
    expected = tension / (PARAMS.mass * PARAMS.gravity) - CONSTRAINTS.load_factor_max
    assert event(0.0, z, PARAMS, zero_control) == pytest.approx(expected, rel=1e-12)
    assert event.terminal is True
    assert event.direction == 1.0


def test_swing_length_min_event_matches_ell_minus_threshold() -> None:
    event = make_swing_length_min_event(CONSTRAINTS)
    z = np.array([0.0, 0.0, 1.5, 0.0])
    assert event(0.0, z, PARAMS, zero_control) == pytest.approx(1.5 - CONSTRAINTS.ell_min, rel=1e-12)
    assert event.terminal is True
    assert event.direction == -1.0


def test_swing_length_max_event_matches_ell_minus_threshold() -> None:
    event = make_swing_length_max_event(CONSTRAINTS)
    z = np.array([0.0, 0.0, 8.0, 0.0])
    assert event(0.0, z, PARAMS, zero_control) == pytest.approx(8.0 - CONSTRAINTS.ell_max, rel=1e-12)
    assert event.terminal is True
    assert event.direction == 1.0


def test_swing_radial_speed_max_event_uses_absolute_value() -> None:
    event = make_swing_radial_speed_max_event(CONSTRAINTS)
    z_pos = np.array([0.0, 0.0, 3.0, 2.0])
    z_neg = np.array([0.0, 0.0, 3.0, -2.0])
    expected = abs(2.0) - CONSTRAINTS.nu_max
    assert event(0.0, z_pos, PARAMS, zero_control) == pytest.approx(expected, rel=1e-12)
    assert event(0.0, z_neg, PARAMS, zero_control) == pytest.approx(expected, rel=1e-12)
    assert event.terminal is True
    assert event.direction == 1.0


def test_swing_release_time_event_matches_t_minus_release_time() -> None:
    event = make_swing_release_time_event(release_time=2.5)
    z = np.array([0.0, 0.0, 3.0, 0.0])
    assert event(2.5, z, PARAMS, zero_control) == 0.0
    assert event(1.0, z, PARAMS, zero_control) == pytest.approx(-1.5, rel=1e-12)
    assert event.terminal is True
    assert event.direction == 1.0


# --- Ballistic events ---------------------------------------------------------


def test_ballistic_ground_impact_event_matches_y() -> None:
    event = make_ballistic_ground_impact_event()
    q = np.array([1.0, 3.5, 2.0, -1.0])
    assert event(0.0, q, PARAMS) == 3.5
    assert event.terminal is True
    assert event.direction == -1.0


def test_ballistic_ground_impact_event_triggers_at_closed_form_landing_time() -> None:
    q0 = np.array([0.0, 50.0, 8.0, 6.0])
    event = make_ballistic_ground_impact_event()

    sol = solve_ivp(
        ballistic_state_derivative,
        (0.0, 20.0),
        q0,
        args=(PARAMS,),
        events=event,
        rtol=1e-12,
        atol=1e-12,
        method="RK45",
    )
    assert sol.success
    assert len(sol.t_events[0]) == 1

    g = PARAMS.gravity
    x0, y0, vx0, vy0 = q0
    t_land = (vy0 + math.sqrt(vy0**2 + 2.0 * g * y0)) / g
    assert sol.t_events[0][0] == pytest.approx(t_land, rel=1e-6)
    assert sol.y_events[0][0][1] == pytest.approx(0.0, abs=1e-8)


def test_ballistic_max_duration_event_matches_t_minus_threshold() -> None:
    event = make_ballistic_max_duration_event(max_duration=10.0)
    q = np.array([0.0, 0.0, 0.0, 0.0])
    assert event(10.0, q, PARAMS) == 0.0
    assert event(3.0, q, PARAMS) == pytest.approx(-7.0, rel=1e-12)
    assert event.terminal is True
    assert event.direction == 1.0


def test_ballistic_capture_region_event_matches_distance_minus_radius() -> None:
    event = make_ballistic_capture_region_event(x_anchor=3.0, y_anchor=4.0, capture_radius=0.5)
    q = np.array([0.0, 0.0, 0.0, 0.0])
    distance = math.hypot(3.0, 4.0)
    assert event(0.0, q, PARAMS) == pytest.approx(distance - 0.5, rel=1e-12)
    assert event.terminal is True
    assert event.direction == -1.0


def test_ballistic_domain_events_match_bounds() -> None:
    x_min_event = make_ballistic_x_min_event(DOMAIN)
    x_max_event = make_ballistic_x_max_event(DOMAIN)
    y_max_event = make_ballistic_y_max_event(DOMAIN)
    q = np.array([10.0, 20.0, 0.0, 0.0])

    assert x_min_event(0.0, q, PARAMS) == pytest.approx(10.0 - DOMAIN.x_min, rel=1e-12)
    assert x_max_event(0.0, q, PARAMS) == pytest.approx(10.0 - DOMAIN.x_max, rel=1e-12)
    assert y_max_event(0.0, q, PARAMS) == pytest.approx(20.0 - DOMAIN.y_max, rel=1e-12)

    assert x_min_event.direction == -1.0
    assert x_max_event.direction == 1.0
    assert y_max_event.direction == 1.0
    assert x_min_event.terminal is True
    assert x_max_event.terminal is True
    assert y_max_event.terminal is True


# --- Numerical validity -------------------------------------------------------


def test_assert_state_finite_accepts_finite_state() -> None:
    assert_state_finite(np.array([0.1, 0.2, 3.0, -1.0]))


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_assert_state_finite_rejects_nonfinite_state(bad_value: float) -> None:
    state = np.array([0.1, bad_value, 3.0, -1.0])
    with pytest.raises(NonFiniteStateError):
        assert_state_finite(state, label="swing state")
