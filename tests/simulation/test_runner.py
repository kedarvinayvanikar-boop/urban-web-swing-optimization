"""Tests for the top-level plan-assemble-evaluate orchestration.

`run_simulation` introduces no new physics or search logic, so these tests
focus on correct orchestration: a successful run's `trajectory`/`evaluation`
match what calling `plan_route` + `assemble_trajectory` +
`evaluate_trajectory` directly would produce, and a failed run reports
`trajectory`/`evaluation` as None with the search's own failure reason
propagated, using the same small city and scenario verified in
`test_astar.py`/`test_trajectory.py`/`test_evaluator.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from webswing.config import BallisticDomain, PhysicalParameters, SwingConstraints
from webswing.geometry.buildings import Building, City, DestinationRegion
from webswing.planning.astar import plan_route
from webswing.planning.state import PlanningStateResolution
from webswing.simulation.evaluator import evaluate_trajectory
from webswing.simulation.runner import SimulationRun, run_simulation
from webswing.simulation.trajectory import assemble_trajectory

PARAMS = PhysicalParameters(mass=1.0, gravity=9.80665)
CONSTRAINTS = SwingConstraints(
    tension_max=200.0, load_factor_max=20.0, ell_min=1.0, ell_max=30.0, nu_max=5.0
)
DOMAIN = BallisticDomain(x_min=-200.0, x_max=200.0, y_max=200.0, max_duration=10.0)
RESOLUTION = PlanningStateResolution(
    theta_bin_width=0.1, omega_bin_width=0.2, ell_bin_width=0.5, nu_bin_width=0.1
)
START_ANCHOR_ID = "START"
START_ANCHOR_POSITION = (0.0, 50.0)
START_STATE = np.array([-0.3, 1.0, 10.0, 0.0])


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


def run(**overrides):
    defaults = dict(
        start_anchor_id=START_ANCHOR_ID,
        start_anchor_position=START_ANCHOR_POSITION,
        start_state=START_STATE,
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
    return run_simulation(**defaults)


def test_successful_run_matches_calling_each_stage_directly() -> None:
    simulation_run = run()

    assert isinstance(simulation_run, SimulationRun)
    assert simulation_run.success is True
    assert simulation_run.failure_reason is None
    assert simulation_run.search_result.success is True

    expected_trajectory = assemble_trajectory(
        simulation_run.search_result.path, START_ANCHOR_ID, START_ANCHOR_POSITION, START_STATE
    )
    expected_evaluation = evaluate_trajectory(
        simulation_run.search_result.path, START_STATE, PARAMS, CONSTRAINTS
    )

    np.testing.assert_array_equal(simulation_run.trajectory.times, expected_trajectory.times)
    np.testing.assert_array_equal(simulation_run.trajectory.positions, expected_trajectory.positions)
    np.testing.assert_array_equal(simulation_run.evaluation.tensions, expected_evaluation.tensions)


def test_successful_run_total_time_matches_search_cost() -> None:
    simulation_run = run()
    assert simulation_run.trajectory.times[-1] == pytest.approx(
        simulation_run.search_result.total_cost, rel=0.0, abs=1e-9
    )


def test_failed_run_reports_no_trajectory_or_evaluation() -> None:
    simulation_run = run(max_expansions=1)
    assert simulation_run.success is False
    assert simulation_run.trajectory is None
    assert simulation_run.evaluation is None
    assert simulation_run.failure_reason == simulation_run.search_result.failure_reason
    assert simulation_run.failure_reason is not None


def test_failed_run_failure_reason_matches_direct_plan_route_call() -> None:
    simulation_run = run(max_expansions=1)
    direct_result = plan_route(
        start_anchor_id=START_ANCHOR_ID,
        start_anchor_position=START_ANCHOR_POSITION,
        start_state=START_STATE,
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
        max_expansions=1,
    )
    assert simulation_run.failure_reason == direct_result.failure_reason
