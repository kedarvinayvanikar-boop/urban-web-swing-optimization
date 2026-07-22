"""Tests for radial control profile parameterizations.

Validates `ConstantControl` and `PiecewiseConstantControl` against their
defining evaluation rules (including the extrapolation policy outside the
breakpoint range and the half-open segment convention at exact breakpoint
values), every construction-time rejection branch, `equal_interval_control`
as the optimizer-facing bridge, and a direct integration with
`swing_state_derivative` confirming these controls satisfy the
`RadialControl` interface used elsewhere in the package.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from webswing.config import PhysicalParameters
from webswing.dynamics.swing import swing_state_derivative
from webswing.exceptions import InvalidControlProfileError
from webswing.optimization.controls import (
    ConstantControl,
    PiecewiseConstantControl,
    equal_interval_control,
)

PARAMS = PhysicalParameters(mass=1.0, gravity=9.80665)


# --- ConstantControl -------------------------------------------------------------


def test_constant_control_returns_value_everywhere() -> None:
    control = ConstantControl(value=0.5, u_min=-1.0, u_max=1.0)
    assert control(0.0) == 0.5
    assert control(100.0) == 0.5
    assert control(-100.0) == 0.5


def test_constant_control_rejects_value_outside_bounds() -> None:
    with pytest.raises(InvalidControlProfileError):
        ConstantControl(value=2.0, u_min=-1.0, u_max=1.0)


def test_constant_control_rejects_nonfinite_value() -> None:
    with pytest.raises(InvalidControlProfileError):
        ConstantControl(value=float("nan"), u_min=-1.0, u_max=1.0)


def test_constant_control_rejects_invalid_bounds() -> None:
    with pytest.raises(InvalidControlProfileError):
        ConstantControl(value=0.0, u_min=1.0, u_max=1.0)
    with pytest.raises(InvalidControlProfileError):
        ConstantControl(value=0.0, u_min=1.0, u_max=-1.0)


# --- PiecewiseConstantControl ------------------------------------------------------


def make_piecewise() -> PiecewiseConstantControl:
    return PiecewiseConstantControl(
        breakpoints=np.array([0.0, 1.0, 2.0]),
        values=np.array([3.0, -2.0]),
        u_min=-5.0,
        u_max=5.0,
    )


def test_piecewise_control_evaluates_correct_segment() -> None:
    control = make_piecewise()
    assert control(0.5) == 3.0
    assert control(1.5) == -2.0


def test_piecewise_control_half_open_segment_at_breakpoint() -> None:
    control = make_piecewise()
    assert control(0.0) == 3.0
    assert control(1.0) == -2.0  # segment boundary belongs to the next segment


def test_piecewise_control_extrapolates_before_and_after_range() -> None:
    control = make_piecewise()
    assert control(-1.0) == 3.0
    assert control(3.0) == -2.0


def test_piecewise_control_rejects_mismatched_value_count() -> None:
    with pytest.raises(InvalidControlProfileError):
        PiecewiseConstantControl(
            breakpoints=np.array([0.0, 1.0, 2.0]),
            values=np.array([3.0, -2.0, 1.0]),
            u_min=-5.0,
            u_max=5.0,
        )


def test_piecewise_control_rejects_non_increasing_breakpoints() -> None:
    with pytest.raises(InvalidControlProfileError):
        PiecewiseConstantControl(
            breakpoints=np.array([0.0, 1.0, 1.0]),
            values=np.array([3.0, -2.0]),
            u_min=-5.0,
            u_max=5.0,
        )


def test_piecewise_control_rejects_too_few_breakpoints() -> None:
    with pytest.raises(InvalidControlProfileError):
        PiecewiseConstantControl(
            breakpoints=np.array([0.0]), values=np.array([1.0]), u_min=-5.0, u_max=5.0
        )


def test_piecewise_control_rejects_value_outside_bounds() -> None:
    with pytest.raises(InvalidControlProfileError):
        PiecewiseConstantControl(
            breakpoints=np.array([0.0, 1.0, 2.0]),
            values=np.array([3.0, 100.0]),
            u_min=-5.0,
            u_max=5.0,
        )


def test_piecewise_control_arrays_are_read_only() -> None:
    control = make_piecewise()
    with pytest.raises(ValueError):
        control.values[0] = 999.0
    with pytest.raises(ValueError):
        control.breakpoints[0] = 999.0


# --- equal_interval_control --------------------------------------------------------


def test_equal_interval_control_builds_expected_segments() -> None:
    control = equal_interval_control(
        parameters=np.array([1.0, 2.0, 3.0]), duration=3.0, u_min=-5.0, u_max=5.0
    )
    assert control(0.5) == 1.0
    assert control(1.5) == 2.0
    assert control(2.5) == 3.0
    np.testing.assert_allclose(control.breakpoints, [0.0, 1.0, 2.0, 3.0])


def test_equal_interval_control_rejects_nonpositive_duration() -> None:
    with pytest.raises(InvalidControlProfileError):
        equal_interval_control(parameters=np.array([1.0]), duration=0.0, u_min=-5.0, u_max=5.0)


def test_equal_interval_control_rejects_empty_parameters() -> None:
    with pytest.raises(InvalidControlProfileError):
        equal_interval_control(parameters=np.array([]), duration=1.0, u_min=-5.0, u_max=5.0)


def test_equal_interval_control_propagates_bound_violation() -> None:
    with pytest.raises(InvalidControlProfileError):
        equal_interval_control(parameters=np.array([100.0]), duration=1.0, u_min=-5.0, u_max=5.0)


# --- Integration with swing_state_derivative ----------------------------------------


def test_constant_control_integrates_with_swing_dynamics() -> None:
    control = ConstantControl(value=0.2, u_min=-1.0, u_max=1.0)
    z0 = np.array([0.3, 0.0, 2.0, 0.0])
    sol = solve_ivp(
        swing_state_derivative,
        (0.0, 1.0),
        z0,
        args=(PARAMS, control),
        rtol=1e-10,
        atol=1e-12,
        method="RK45",
    )
    assert sol.success
    # nu_dot = u = 0.2 exactly (constant), so nu(t) = nu0 + 0.2*t.
    assert sol.y[3, -1] == pytest.approx(0.2 * 1.0, rel=1e-6)


def test_piecewise_control_integrates_with_swing_dynamics_across_a_switch() -> None:
    control = PiecewiseConstantControl(
        breakpoints=np.array([0.0, 0.5, 1.0]), values=np.array([0.4, -0.4]), u_min=-1.0, u_max=1.0
    )
    z0 = np.array([0.3, 0.0, 2.0, 0.0])
    sol = solve_ivp(
        swing_state_derivative,
        (0.0, 1.0),
        z0,
        args=(PARAMS, control),
        max_step=0.01,
        rtol=1e-10,
        atol=1e-12,
        method="RK45",
    )
    assert sol.success
    # nu increases at 0.4 m/s^2 for 0.5 s then decreases at 0.4 m/s^2 for
    # 0.5 s, returning to its starting value.
    assert sol.y[3, -1] == pytest.approx(0.0, abs=1e-6)
