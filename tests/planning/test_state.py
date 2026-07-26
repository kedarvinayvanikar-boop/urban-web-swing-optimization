"""Tests for the discretized planning-graph state model.

Validates `PlanningStateResolution` bin-width validation, the half-open
binning convention on both sides of zero, that two continuous states
falling in the same bin discretize to an equal (and equally-hashed)
`PlanningState` -- the property A*'s duplicate-state detection and closed
set depend on -- and the `representative_state` round trip.
"""

from __future__ import annotations

import numpy as np
import pytest

from webswing.planning.state import (
    PlanningState,
    PlanningStateResolution,
    discretize_state,
    representative_state,
)
from webswing.exceptions import InvalidPhysicalParameterError, NonFiniteStateError

RESOLUTION = PlanningStateResolution(
    theta_bin_width=0.1, omega_bin_width=0.2, ell_bin_width=0.5, nu_bin_width=0.1
)


# --- PlanningStateResolution validation -----------------------------------------


@pytest.mark.parametrize(
    "field", ["theta_bin_width", "omega_bin_width", "ell_bin_width", "nu_bin_width"]
)
@pytest.mark.parametrize("bad_value", [0.0, -1.0, float("nan"), float("inf")])
def test_resolution_rejects_nonpositive_or_nonfinite_widths(field: str, bad_value: float) -> None:
    kwargs = dict(theta_bin_width=0.1, omega_bin_width=0.2, ell_bin_width=0.5, nu_bin_width=0.1)
    kwargs[field] = bad_value
    with pytest.raises(InvalidPhysicalParameterError):
        PlanningStateResolution(**kwargs)


# --- discretize_state: binning convention ---------------------------------------


def test_discretize_state_positive_and_negative_bins() -> None:
    state = np.array([0.05, -0.05, 1.2, -0.05])
    node = discretize_state("A", state, RESOLUTION)
    assert node.theta_bin == 0  # [0.0, 0.1)
    assert node.omega_bin == -1  # [-0.2, 0.0)
    assert node.ell_bin == 2  # [1.0, 1.5)
    assert node.nu_bin == -1  # [-0.1, 0.0)


def test_discretize_state_half_open_at_bin_edge() -> None:
    state = np.array([0.1, 0.0, 0.5, 0.0])
    node = discretize_state("A", state, RESOLUTION)
    assert node.theta_bin == 1  # exactly at the edge belongs to the next bin
    assert node.ell_bin == 1


def test_discretize_state_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        discretize_state("A", np.array([0.0, 0.0, 1.0]), RESOLUTION)


def test_discretize_state_rejects_nonfinite_state() -> None:
    with pytest.raises(NonFiniteStateError):
        discretize_state("A", np.array([0.0, float("nan"), 1.0, 0.0]), RESOLUTION)


# --- duplicate-state detection property -----------------------------------------


def test_two_states_in_the_same_bin_are_equal_and_equally_hashed() -> None:
    node_a = discretize_state("A", np.array([0.02, 0.0, 1.1, 0.0]), RESOLUTION)
    node_b = discretize_state("A", np.array([0.08, 0.19, 1.4, 0.09]), RESOLUTION)
    assert node_a == node_b
    assert hash(node_a) == hash(node_b)


def test_two_states_in_different_bins_are_not_equal() -> None:
    node_a = discretize_state("A", np.array([0.02, 0.0, 1.1, 0.0]), RESOLUTION)
    node_b = discretize_state("A", np.array([0.25, 0.0, 1.1, 0.0]), RESOLUTION)
    assert node_a != node_b


def test_same_bin_different_anchor_is_not_equal() -> None:
    node_a = discretize_state("A", np.array([0.02, 0.0, 1.1, 0.0]), RESOLUTION)
    node_b = discretize_state("B", np.array([0.02, 0.0, 1.1, 0.0]), RESOLUTION)
    assert node_a != node_b


def test_planning_state_usable_in_a_set() -> None:
    nodes = [
        discretize_state("A", np.array([0.02, 0.0, 1.1, 0.0]), RESOLUTION),
        discretize_state("A", np.array([0.08, 0.0, 1.1, 0.0]), RESOLUTION),  # duplicate of above
        discretize_state("A", np.array([0.5, 0.0, 1.1, 0.0]), RESOLUTION),  # distinct
    ]
    assert len(set(nodes)) == 2


# --- representative_state ---------------------------------------------------------


def test_representative_state_round_trips_through_discretize() -> None:
    node = PlanningState(anchor_id="A", theta_bin=-1, omega_bin=2, ell_bin=3, nu_bin=-2)
    center = representative_state(node, RESOLUTION)
    round_tripped = discretize_state("A", center, RESOLUTION)
    assert round_tripped == node


def test_representative_state_is_bin_center() -> None:
    node = PlanningState(anchor_id="A", theta_bin=0, omega_bin=0, ell_bin=0, nu_bin=0)
    center = representative_state(node, RESOLUTION)
    expected = np.array(
        [0.5 * RESOLUTION.theta_bin_width, 0.5 * RESOLUTION.omega_bin_width, 0.5 * RESOLUTION.ell_bin_width, 0.5 * RESOLUTION.nu_bin_width]
    )
    np.testing.assert_allclose(center, expected)
