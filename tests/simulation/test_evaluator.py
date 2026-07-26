"""Tests for force-derived trajectory evaluation (tension, load factor).

Reuses the same small city as `test_astar.py`/`test_trajectory.py`. Checks
index-alignment with `assemble_trajectory`'s output on the same path, exact
cross-checked tension against an independently reconstructed control and
`web_tension` call, NaN during ballistic samples, the trivial single-point
case, and rejection of a mis-typed edge label.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from webswing.config import BallisticDomain, PhysicalParameters, SwingConstraints
from webswing.dynamics.swing import web_tension
from webswing.geometry.buildings import Building, City, DestinationRegion
from webswing.optimization.controls import equal_interval_control
from webswing.optimization.local_transfer import simulate_transfer
from webswing.planning.astar import plan_route
from webswing.planning.state import PlanningStateResolution
from webswing.simulation.evaluator import TrajectoryEvaluation, evaluate_trajectory
from webswing.simulation.trajectory import assemble_trajectory

PARAMS = PhysicalParameters(mass=1.0, gravity=9.80665)
CONSTRAINTS = SwingConstraints(
    tension_max=200.0, load_factor_max=20.0, ell_min=1.0, ell_max=30.0, nu_max=5.0
)
DOMAIN = BallisticDomain(x_min=-200.0, x_max=200.0, y_max=200.0, max_duration=10.0)
RESOLUTION = PlanningStateResolution(
    theta_bin_width=0.1, omega_bin_width=0.2, ell_bin_width=0.5, nu_bin_width=0.1
)
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


def run_plan_route(**overrides):
    defaults = dict(
        start_anchor_id="START",
        start_anchor_position=(0.0, 50.0),
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
    return plan_route(**defaults)


# --- trivial (empty-path) case -----------------------------------------------------


def test_empty_path_uses_zero_control_convention() -> None:
    start_state = np.array([0.0, 0.5, 2.0, 0.0])
    evaluation = evaluate_trajectory((), start_state, PARAMS, CONSTRAINTS)

    expected_tension = web_tension(0.0, 0.5, 2.0, 0.0, PARAMS)
    assert evaluation.tensions[0] == pytest.approx(expected_tension, rel=1e-12)
    assert evaluation.load_factors[0] == pytest.approx(
        expected_tension / (PARAMS.mass * PARAMS.gravity), rel=1e-12
    )
    assert evaluation.tension_margins[0] == pytest.approx(
        CONSTRAINTS.tension_max - expected_tension, rel=1e-12
    )


# --- real planned route -------------------------------------------------------------


def test_evaluation_length_matches_assembled_trajectory() -> None:
    result = run_plan_route()
    trajectory = assemble_trajectory(result.path, "START", (0.0, 50.0), START_STATE)
    evaluation = evaluate_trajectory(result.path, START_STATE, PARAMS, CONSTRAINTS)
    assert len(evaluation.tensions) == len(trajectory.times)


def test_first_swing_sample_tension_matches_independent_computation() -> None:
    result = run_plan_route()
    evaluation = evaluate_trajectory(result.path, START_STATE, PARAMS, CONSTRAINTS)

    edge = result.path[0]
    problem = edge.label.problem
    transfer_result = edge.label.result
    control = equal_interval_control(
        transfer_result.solution[1:], transfer_result.release_time, problem.u_min, problem.u_max
    )
    sim = simulate_transfer(transfer_result.solution, problem)
    theta0, omega0, ell0, _nu0 = sim.swing_states[0]
    t0 = float(sim.swing_times[0])
    expected_tension = web_tension(theta0, omega0, ell0, control(t0), PARAMS)

    assert evaluation.tensions[0] == pytest.approx(expected_tension, rel=1e-12)


def test_ballistic_samples_have_nan_tension_and_load_factor() -> None:
    result = run_plan_route()
    trajectory = assemble_trajectory(result.path, "START", (0.0, 50.0), START_STATE)
    evaluation = evaluate_trajectory(result.path, START_STATE, PARAMS, CONSTRAINTS)

    ballistic_indices = [i for i, m in enumerate(trajectory.modes) if m == "ballistic"]
    assert len(ballistic_indices) > 0
    for i in ballistic_indices:
        assert math.isnan(evaluation.tensions[i])
        assert math.isnan(evaluation.load_factors[i])
        assert math.isnan(evaluation.tension_margins[i])
        assert math.isnan(evaluation.load_factor_margins[i])


def test_swing_samples_have_finite_tension_within_loose_constraints() -> None:
    result = run_plan_route()
    trajectory = assemble_trajectory(result.path, "START", (0.0, 50.0), START_STATE)
    evaluation = evaluate_trajectory(result.path, START_STATE, PARAMS, CONSTRAINTS)

    swing_indices = [i for i, m in enumerate(trajectory.modes) if m == "swing"]
    for i in swing_indices:
        assert math.isfinite(evaluation.tensions[i])
        assert evaluation.tension_margins[i] > 0.0  # constraints are loose in this scenario


def test_rejects_edge_with_non_planned_transfer_label() -> None:
    from dataclasses import replace

    result = run_plan_route()
    bad_path = (replace(result.path[0], label="not-a-planned-transfer"),)
    with pytest.raises(TypeError):
        evaluate_trajectory(bad_path, START_STATE, PARAMS, CONSTRAINTS)


# --- TrajectoryEvaluation validation -------------------------------------------------


def test_trajectory_evaluation_rejects_inconsistent_lengths() -> None:
    with pytest.raises(ValueError):
        TrajectoryEvaluation(
            tensions=np.array([1.0, 2.0]),
            load_factors=np.array([1.0]),
            tension_margins=np.array([1.0, 2.0]),
            load_factor_margins=np.array([1.0, 2.0]),
        )
