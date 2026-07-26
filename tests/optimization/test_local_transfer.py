"""Tests for the constrained local swing-to-anchor transfer optimization.

Scenarios were derived by first simulating the passive (near-zero-control)
swing-release-ballistic path numerically to locate a target anchor genuinely
reachable by the dynamics, rather than asserting arbitrary hand-picked
numbers. Assertions check invariants (captured state within the capture
radius, objective consistency, bound satisfaction, reproducibility) rather
than pinning exact floating-point outputs, since the specific solver
iterate is not itself a physical invariant.

A real numerical issue surfaced during this derivation and is covered
explicitly here: `solve_ivp`'s adaptive stepping can take a step so large
that a brief capture-region entry/exit happens entirely inside it, invisible
to the endpoint-based terminal-event sign check. `simulate_transfer` bounds
`max_step` to prevent this; `test_capture_of_a_reachable_target_succeeds`
is a regression test for that fix (the target/capture_radius combination
here reproduces the missed-event failure without the fix).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from webswing.config import BallisticDomain, PhysicalParameters, SwingConstraints
from webswing.geometry.buildings import Building, City, DestinationRegion
from webswing.optimization.local_transfer import (
    LocalTransferProblem,
    LocalTransferResult,
    solve_local_transfer,
)

PARAMS = PhysicalParameters(mass=1.0, gravity=9.80665)
CONSTRAINTS = SwingConstraints(
    tension_max=200.0, load_factor_max=20.0, ell_min=1.0, ell_max=30.0, nu_max=5.0
)
DOMAIN = BallisticDomain(x_min=-200.0, x_max=200.0, y_max=200.0, max_duration=10.0)


def make_reachable_problem(**overrides) -> LocalTransferProblem:
    defaults = dict(
        initial_state=np.array([-0.3, 1.0, 10.0, 0.0]),
        current_anchor=(0.0, 50.0),
        target_anchor=(22.0, 27.0),
        params=PARAMS,
        constraints=CONSTRAINTS,
        u_min=-0.5,
        u_max=0.5,
        n_control_segments=3,
        t_release_min=0.1,
        t_release_max=3.0,
        capture_radius=4.0,
        ballistic_domain=DOMAIN,
    )
    defaults.update(overrides)
    return LocalTransferProblem(**defaults)


def test_problem_rejects_invalid_state_shape() -> None:
    with pytest.raises(ValueError):
        make_reachable_problem(initial_state=np.array([0.0, 0.0, 1.0]))


def test_problem_rejects_invalid_control_bounds() -> None:
    with pytest.raises(ValueError):
        make_reachable_problem(u_min=1.0, u_max=1.0)


def test_problem_rejects_invalid_release_time_bounds() -> None:
    with pytest.raises(ValueError):
        make_reachable_problem(t_release_min=2.0, t_release_max=1.0)


def test_problem_rejects_nonpositive_capture_radius() -> None:
    with pytest.raises(ValueError):
        make_reachable_problem(capture_radius=0.0)


# --- feasible capture ------------------------------------------------------------


def test_capture_of_a_reachable_target_succeeds() -> None:
    problem = make_reachable_problem()
    result = solve_local_transfer(problem)

    assert isinstance(result, LocalTransferResult)
    assert result.success is True
    assert result.failure_reason is None
    assert result.final_state is not None

    xa, ya = problem.target_anchor
    fx, fy = result.final_state[0], result.final_state[1]
    distance = math.hypot(fx - xa, fy - ya)
    assert distance <= problem.capture_radius + 1e-6

    assert result.max_constraint_violation == pytest.approx(0.0, abs=1e-6)
    assert result.objective_value == pytest.approx(
        result.release_time + result.ballistic_time, rel=1e-9
    )


def test_solution_satisfies_decision_bounds() -> None:
    problem = make_reachable_problem()
    result = solve_local_transfer(problem)
    assert problem.t_release_min - 1e-9 <= result.solution[0] <= problem.t_release_max + 1e-9
    assert np.all(result.solution[1:] >= problem.u_min - 1e-9)
    assert np.all(result.solution[1:] <= problem.u_max + 1e-9)


def test_reporting_fields_are_populated() -> None:
    problem = make_reachable_problem()
    result = solve_local_transfer(problem)
    assert result.solver == "SLSQP"
    assert result.n_objective_evaluations > 0
    assert result.n_dynamics_integrations > 0
    assert result.termination_status is not None
    assert isinstance(result.message, str) and result.message


def test_reproducible_from_the_same_initial_guess() -> None:
    problem = make_reachable_problem()
    result_a = solve_local_transfer(problem)
    result_b = solve_local_transfer(problem)
    assert result_a.objective_value == result_b.objective_value
    np.testing.assert_array_equal(result_a.solution, result_b.solution)


# --- infeasible transfer ----------------------------------------------------------


def test_unreachable_target_within_short_duration_reports_failure() -> None:
    problem = make_reachable_problem(
        target_anchor=(500.0, 500.0),
        ballistic_domain=BallisticDomain(x_min=-200.0, x_max=200.0, y_max=200.0, max_duration=1.5),
        t_release_min=0.1,
        t_release_max=1.0,
    )
    result = solve_local_transfer(problem)
    assert result.success is False
    assert result.failure_reason is not None
    assert result.max_constraint_violation > 0.0


# --- collision post-hoc check ----------------------------------------------------


def test_intervening_building_reports_collision_failure() -> None:
    blocker = Building(
        building_id="BLOCKER",
        vertices=np.array([[10.0, 0.0], [18.0, 0.0], [18.0, 35.0], [10.0, 35.0]]),
        width=8.0,
        height=35.0,
        roof_elevation=35.0,
    )
    city = City(
        buildings=(blocker,),
        destination=DestinationRegion(x_min=100.0, x_max=110.0, y_min=0.0, y_max=5.0),
    )
    problem = make_reachable_problem(city=city)
    result = solve_local_transfer(problem)
    assert result.success is False
    assert result.failure_reason == "trajectory collides with city geometry"


def test_same_problem_without_city_still_succeeds() -> None:
    # Sanity check that the collision failure above is attributable to the
    # building, not to some other change in the problem.
    problem = make_reachable_problem(city=None)
    result = solve_local_transfer(problem)
    assert result.success is True
