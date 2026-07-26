"""Tests for stitching a planned route into one continuous trajectory.

Scenarios reuse the same small city as `test_astar.py` (derived by
simulating the physics first, not by asserting arbitrary numbers). Checks
focus on invariants: the assembled trajectory's total duration matches
`SearchResult.total_cost` exactly (not merely approximately, despite the
fixed-grid ballistic sampling used internally), the trivial empty-path case
matches the analytic `attached_position`/`attached_velocity` formulas
directly, mode/anchor labeling is consistent with swing vs. ballistic
samples, and a mis-typed edge label is rejected.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from webswing.config import BallisticDomain, PhysicalParameters, SwingConstraints
from webswing.dynamics.release import attached_position, attached_velocity
from webswing.geometry.buildings import Building, City, DestinationRegion
from webswing.geometry.collision import point_in_destination
from webswing.planning.astar import GOAL_NODE, plan_route
from webswing.planning.state import PlanningStateResolution
from webswing.simulation.trajectory import Trajectory, assemble_trajectory

PARAMS = PhysicalParameters(mass=1.0, gravity=9.80665)
CONSTRAINTS = SwingConstraints(
    tension_max=200.0, load_factor_max=20.0, ell_min=1.0, ell_max=30.0, nu_max=5.0
)
DOMAIN = BallisticDomain(x_min=-200.0, x_max=200.0, y_max=200.0, max_duration=10.0)
RESOLUTION = PlanningStateResolution(
    theta_bin_width=0.1, omega_bin_width=0.2, ell_bin_width=0.5, nu_bin_width=0.1
)


def make_city() -> City:
    verts = np.array([[20.0, 0.0], [24.0, 0.0], [24.0, 27.0], [20.0, 27.0]])
    building = Building(
        building_id="B1",
        vertices=verts,
        width=4.0,
        height=27.0,
        roof_elevation=27.0,
        candidate_anchors=((20.0, 27.0), (24.0, 27.0)),
    )
    destination = DestinationRegion(x_min=18.0, x_max=26.0, y_min=24.0, y_max=30.0)
    return City(buildings=(building,), destination=destination)


def run_plan_route(**overrides):
    defaults = dict(
        start_anchor_id="START",
        start_anchor_position=(0.0, 50.0),
        start_state=np.array([-0.3, 1.0, 10.0, 0.0]),
        city=make_city(),
        params=PARAMS,
        constraints=CONSTRAINTS,
        resolution=RESOLUTION,
        u_min=-0.5,
        u_max=0.5,
        n_control_segments=3,
        t_release_min=0.1,
        t_release_max=3.0,
        capture_radius=4.0,
        ballistic_domain=DOMAIN,
        max_expansions=50,
    )
    defaults.update(overrides)
    return plan_route(**defaults)


# --- trivial (empty-path) case -----------------------------------------------------


def test_empty_path_falls_back_to_analytic_start_state() -> None:
    start_state = np.array([0.0, 0.0, 1.0, 0.0])
    trajectory = assemble_trajectory((), "START", (20.0, 27.0), start_state)

    assert len(trajectory.times) == 1
    assert trajectory.times[0] == 0.0
    assert trajectory.modes == ("swing",)
    assert trajectory.active_anchor_ids == ("START",)

    expected_position = attached_position(0.0, 1.0, 20.0, 27.0)
    expected_velocity = attached_velocity(0.0, 0.0, 1.0, 0.0)
    np.testing.assert_allclose(trajectory.positions[0], expected_position)
    np.testing.assert_allclose(trajectory.velocities[0], expected_velocity)


# --- real planned route -------------------------------------------------------------


def test_assembled_trajectory_total_time_matches_search_total_cost() -> None:
    result = run_plan_route()
    assert result.success is True

    trajectory = assemble_trajectory(
        result.path, "START", (0.0, 50.0), np.array([-0.3, 1.0, 10.0, 0.0])
    )

    assert trajectory.times[0] == 0.0
    assert trajectory.times[-1] == pytest.approx(result.total_cost, rel=0.0, abs=1e-9)
    assert np.all(np.diff(trajectory.times) >= 0.0)


def test_trajectory_starts_swinging_at_the_start_anchor() -> None:
    result = run_plan_route()
    trajectory = assemble_trajectory(
        result.path, "START", (0.0, 50.0), np.array([-0.3, 1.0, 10.0, 0.0])
    )
    assert trajectory.modes[0] == "swing"
    assert trajectory.active_anchor_ids[0] == "START"
    assert not math.isnan(trajectory.web_lengths[0])


def test_ballistic_samples_have_no_anchor_and_nan_swing_quantities() -> None:
    result = run_plan_route()
    trajectory = assemble_trajectory(
        result.path, "START", (0.0, 50.0), np.array([-0.3, 1.0, 10.0, 0.0])
    )
    ballistic_indices = [i for i, m in enumerate(trajectory.modes) if m == "ballistic"]
    assert len(ballistic_indices) > 0
    for i in ballistic_indices:
        assert trajectory.active_anchor_ids[i] is None
        assert math.isnan(trajectory.web_lengths[i])
        assert math.isnan(trajectory.angular_rates[i])
        assert math.isnan(trajectory.radial_rates[i])


def test_interception_truncated_trajectory_ends_inside_destination() -> None:
    result = run_plan_route()
    assert result.path[-1].to_node == GOAL_NODE  # this scenario reaches goal by interception

    trajectory = assemble_trajectory(
        result.path, "START", (0.0, 50.0), np.array([-0.3, 1.0, 10.0, 0.0])
    )
    last_point = (float(trajectory.positions[-1][0]), float(trajectory.positions[-1][1]))
    assert point_in_destination(last_point, make_city().destination) is True
    assert trajectory.modes[-1] == "ballistic"


def test_rejects_edge_with_non_planned_transfer_label() -> None:
    result = run_plan_route()
    bad_path = (replace(result.path[0], label="not-a-planned-transfer"),)
    with pytest.raises(TypeError):
        assemble_trajectory(bad_path, "START", (0.0, 50.0), np.array([-0.3, 1.0, 10.0, 0.0]))


# --- Trajectory validation ----------------------------------------------------------


def test_trajectory_rejects_inconsistent_lengths() -> None:
    with pytest.raises(ValueError):
        Trajectory(
            times=np.array([0.0, 1.0]),
            positions=np.array([[0.0, 0.0]]),
            velocities=np.array([[0.0, 0.0], [0.0, 0.0]]),
            modes=("swing", "swing"),
            active_anchor_ids=("A", "A"),
            web_lengths=np.array([1.0, 1.0]),
            angular_rates=np.array([0.0, 0.0]),
            radial_rates=np.array([0.0, 0.0]),
        )
