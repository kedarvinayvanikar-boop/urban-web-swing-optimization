"""Tests for the A* planning heuristics.

Validates `distance_to_region` against hand-computed cases (inside, on
boundary, directly outside one axis, diagonally outside both axes),
`node_position` against a direct `attached_position` computation using the
same bin-center values, `zero_heuristic`'s unconditional 0.0 return (the
Dijkstra-equivalence property), and `speed_bound_heuristic`'s exact
distance/v_max value plus its v_max validation.
"""

from __future__ import annotations

import math

import pytest

from webswing.dynamics.release import attached_position
from webswing.geometry.buildings import DestinationRegion
from webswing.planning.heuristic import (
    distance_to_region,
    node_position,
    speed_bound_heuristic,
    zero_heuristic,
)
from webswing.planning.state import PlanningState, PlanningStateResolution

RESOLUTION = PlanningStateResolution(
    theta_bin_width=0.1, omega_bin_width=0.2, ell_bin_width=0.5, nu_bin_width=0.1
)
ANCHOR = (10.0, 40.0)
DESTINATION = DestinationRegion(x_min=50.0, x_max=60.0, y_min=0.0, y_max=5.0)


# --- distance_to_region -----------------------------------------------------------


def test_distance_to_region_inside_is_zero() -> None:
    assert distance_to_region((55.0, 2.0), DESTINATION) == 0.0


def test_distance_to_region_on_boundary_is_zero() -> None:
    assert distance_to_region((50.0, 0.0), DESTINATION) == 0.0


def test_distance_to_region_directly_outside_one_axis() -> None:
    d = distance_to_region((45.0, 2.0), DESTINATION)
    assert d == pytest.approx(5.0, rel=1e-12)


def test_distance_to_region_diagonally_outside_both_axes() -> None:
    d = distance_to_region((45.0, -3.0), DESTINATION)
    assert d == pytest.approx(math.hypot(5.0, 3.0), rel=1e-12)


# --- node_position -----------------------------------------------------------------


def test_node_position_matches_attached_position_at_bin_center() -> None:
    state = PlanningState(anchor_id="A", theta_bin=0, omega_bin=0, ell_bin=3, nu_bin=0)
    position = node_position(state, RESOLUTION, ANCHOR)

    theta_center = 0.5 * RESOLUTION.theta_bin_width
    ell_center = (3 + 0.5) * RESOLUTION.ell_bin_width
    expected = attached_position(theta_center, ell_center, ANCHOR[0], ANCHOR[1])

    assert position == pytest.approx(expected, rel=1e-12)


def test_node_position_differs_for_different_anchor_positions() -> None:
    state = PlanningState(anchor_id="A", theta_bin=0, omega_bin=0, ell_bin=3, nu_bin=0)
    pos_a = node_position(state, RESOLUTION, (0.0, 0.0))
    pos_b = node_position(state, RESOLUTION, (100.0, 100.0))
    assert pos_a != pos_b


# --- zero_heuristic ----------------------------------------------------------------


def test_zero_heuristic_always_returns_zero() -> None:
    state_near = PlanningState(anchor_id="A", theta_bin=0, omega_bin=0, ell_bin=0, nu_bin=0)
    state_far = PlanningState(anchor_id="B", theta_bin=100, omega_bin=-50, ell_bin=200, nu_bin=10)
    assert zero_heuristic(state_near, RESOLUTION, ANCHOR, DESTINATION) == 0.0
    assert zero_heuristic(state_far, RESOLUTION, (-500.0, 500.0), DESTINATION) == 0.0


# --- speed_bound_heuristic ----------------------------------------------------------


def test_speed_bound_heuristic_matches_distance_over_v_max() -> None:
    state = PlanningState(anchor_id="A", theta_bin=0, omega_bin=0, ell_bin=3, nu_bin=0)
    v_max = 8.0
    h = speed_bound_heuristic(state, RESOLUTION, ANCHOR, DESTINATION, v_max)

    position = node_position(state, RESOLUTION, ANCHOR)
    expected = distance_to_region(position, DESTINATION) / v_max
    assert h == pytest.approx(expected, rel=1e-12)


def test_speed_bound_heuristic_rejects_nonpositive_v_max() -> None:
    state = PlanningState(anchor_id="A", theta_bin=0, omega_bin=0, ell_bin=3, nu_bin=0)
    with pytest.raises(ValueError):
        speed_bound_heuristic(state, RESOLUTION, ANCHOR, DESTINATION, 0.0)
    with pytest.raises(ValueError):
        speed_bound_heuristic(state, RESOLUTION, ANCHOR, DESTINATION, -1.0)


def test_speed_bound_heuristic_rejects_nonfinite_v_max() -> None:
    state = PlanningState(anchor_id="A", theta_bin=0, omega_bin=0, ell_bin=3, nu_bin=0)
    with pytest.raises(ValueError):
        speed_bound_heuristic(state, RESOLUTION, ANCHOR, DESTINATION, float("inf"))
    with pytest.raises(ValueError):
        speed_bound_heuristic(state, RESOLUTION, ANCHOR, DESTINATION, float("nan"))
