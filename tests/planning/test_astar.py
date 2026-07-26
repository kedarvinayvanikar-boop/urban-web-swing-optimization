"""Tests for A* global trajectory planning.

Split to match the module's own separation: `astar_search` (generic,
physics-free) is tested against a small, hand-verifiable weighted graph --
fast and independent of `solve_ivp`/`scipy.optimize` -- covering CLAUDE.md's
"A* correctness on a small manually verifiable graph," "equivalence to
Dijkstra when h=0," "admissible-heuristic behaviour," "parent
reconstruction," "no-route behaviour," and search-size termination limits.
`plan_route` (the physics wrapper) is tested against a small real city,
with scenarios chosen to stay cheap wherever the behavior under test does
not itself require running the optimizer (a trivial already-at-destination
case, and a search-budget failure that never reaches the neighbor-
generation step) plus one genuine physics scenario for destination
interception during a ballistic edge, mirroring how `test_local_transfer.py`
and `test_cache.py` derived their scenarios by simulating first.

The synthetic graph used throughout
--------------------------------------
    A --1--> B --1--> C --1--> D      (true shortest path, cost 3)
    A --10-------------------> D      (an expensive shortcut, listed first
                                        in A's neighbor list to guard
                                        against an implementation that
                                        commits to whichever path is
                                        discovered first rather than the
                                        cheapest)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from webswing.config import BallisticDomain, PhysicalParameters, SwingConstraints
from webswing.geometry.buildings import Building, City, DestinationRegion
from webswing.planning.astar import GOAL_NODE, astar_search, plan_route
from webswing.planning.state import PlanningStateResolution, discretize_state

GRAPH = {
    "A": [("D", 10.0, "A->D"), ("B", 1.0, "A->B")],
    "B": [("C", 1.0, "B->C")],
    "C": [("D", 1.0, "C->D")],
    "D": [],
}


def graph_neighbors(node: str):
    return GRAPH[node]


def is_goal_d(node: str) -> bool:
    return node == "D"


# --- astar_search: correctness on the small graph --------------------------------


def test_finds_the_true_shortest_path_not_the_shortcut() -> None:
    result = astar_search("A", is_goal_d, graph_neighbors)
    assert result.success is True
    assert result.total_cost == pytest.approx(3.0)
    assert [e.to_node for e in result.path] == ["B", "C", "D"]


def test_parent_reconstruction_chains_correctly() -> None:
    result = astar_search("A", is_goal_d, graph_neighbors)
    assert result.path[0].from_node == "A"
    for i in range(len(result.path) - 1):
        assert result.path[i].to_node == result.path[i + 1].from_node
    assert result.path[-1].to_node == "D"
    assert sum(e.cost for e in result.path) == pytest.approx(result.total_cost)


def test_edge_labels_are_preserved_in_the_path() -> None:
    result = astar_search("A", is_goal_d, graph_neighbors)
    assert [e.label for e in result.path] == ["A->B", "B->C", "C->D"]


# --- equivalence to Dijkstra (h=0) and admissible-heuristic behaviour -------------


def test_default_heuristic_is_zero_and_matches_dijkstra() -> None:
    result = astar_search("A", is_goal_d, graph_neighbors)  # heuristic omitted -> h=0
    assert result.total_cost == pytest.approx(3.0)


def test_admissible_nonzero_heuristic_finds_the_same_optimal_cost() -> None:
    # h(n) <= true remaining cost (3, 2, 1, 0) at every node: admissible.
    h = {"A": 2.0, "B": 1.0, "C": 1.0, "D": 0.0}
    result = astar_search("A", is_goal_d, graph_neighbors, heuristic=lambda n: h[n])
    assert result.success is True
    assert result.total_cost == pytest.approx(3.0)
    assert [e.to_node for e in result.path] == ["B", "C", "D"]


# --- no-route behaviour and search termination limits -----------------------------


def test_no_route_reports_failure_with_a_reason() -> None:
    result = astar_search("A", lambda n: n == "UNREACHABLE", graph_neighbors)
    assert result.success is False
    assert result.total_cost == math.inf
    assert result.path == ()
    assert result.failure_reason is not None


def test_max_expansions_limit_reports_failure() -> None:
    result = astar_search("A", is_goal_d, graph_neighbors, max_expansions=1)
    assert result.success is False
    assert "max_expansions" in result.failure_reason
    assert result.nodes_expanded == 1


def test_trivial_start_equal_to_goal_succeeds_with_empty_path() -> None:
    result = astar_search("D", is_goal_d, graph_neighbors)
    assert result.success is True
    assert result.path == ()
    assert result.total_cost == 0.0


# --- plan_route: physics wrapper --------------------------------------------------

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


def test_plan_route_trivial_start_already_in_destination_needs_no_optimization() -> None:
    result = run_plan_route(
        start_anchor_position=(20.0, 27.0), start_state=np.array([0.0, 0.0, 1.0, 0.0])
    )
    assert result.success is True
    assert result.path == ()
    assert result.total_cost == 0.0


def test_plan_route_reports_failure_when_search_budget_is_exhausted() -> None:
    result = run_plan_route(max_expansions=1)
    assert result.success is False
    assert "max_expansions" in result.failure_reason


def test_plan_route_reaches_destination_by_ballistic_interception() -> None:
    result = run_plan_route()
    assert result.success is True
    assert len(result.path) == 1
    assert result.path[0].to_node == GOAL_NODE
    assert result.total_cost == result.path[0].cost
    assert result.total_cost > 0.0


def test_plan_route_start_node_matches_independent_discretization() -> None:
    result = run_plan_route()
    expected_start = discretize_state("START", np.array([-0.3, 1.0, 10.0, 0.0]), RESOLUTION)
    assert result.path[0].from_node == expected_start
