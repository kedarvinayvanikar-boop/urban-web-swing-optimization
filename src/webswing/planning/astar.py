r"""A* global trajectory planning across candidate anchors.

Two layers, kept deliberately separate:

- `astar_search`: a generic, physics-free A* over any hashable node type,
  parameterized by caller-supplied `is_goal` and `neighbors` callables. It
  implements closed-set handling, duplicate-state detection (a cheaper path
  to an already-closed node is simply not re-explored, correct under a
  consistent heuristic), parent reconstruction, and a search-size
  termination limit -- independently testable on a small hand-built graph
  with no dependency on `optimization.local_transfer` or `solve_ivp`.
- `plan_route`: the physics-specific wrapper. It builds the neighbor
  function from a `City`'s candidate anchors, `optimization.local_transfer`,
  and `planning.cache`, and supplies the goal test (destination-region
  containment at a captured anchor, or direct interception during a
  ballistic edge).

Consistency of the default heuristic
----------------------------------------
`astar_search` does not reopen a closed node when a cheaper path to it is
later found; this is correct for a *consistent* heuristic (Dijkstra's
non-reopening property extends to A* under consistency). `h(n) = 0`
(`planning.heuristic.zero_heuristic`, the default here) is trivially
consistent: `0 <= cost(u, v) + 0` holds for every non-negative edge cost.
This is the heuristic recommended by `planning.heuristic` given no hard
speed bound has been derived for this model (see that module's docstring);
if a caller supplies `speed_bound_heuristic` instead, its consistency (not
just admissibility) on this particular graph has not been separately
verified, and a heuristic that is admissible but not consistent can in
principle require reopening for strict optimality.

Anchor identifiers
--------------------
`City.all_candidate_anchors()` returns (building_id, point) pairs, and a
building may own several anchor points, so building_id alone cannot serve
as a `planning.state.PlanningState.anchor_id`. `plan_route` synthesizes a
unique id `f"{building_id}#{index}"` per candidate anchor point.

Destination interception during a ballistic edge
-------------------------------------------------------
Per CLAUDE.md, the destination region may be reached mid-flight, before any
anchor is actually captured. For every candidate transfer, `plan_route`
re-simulates the optimizer's converged solution (`simulate_transfer`, an
inexpensive re-integration of an already-solved decision vector, not a
re-optimization) and scans its sampled ballistic path for entry into the
destination region. If found, that transfer additionally yields an edge to
a distinguished goal node at the time of first entry -- which may be
earlier, and cheaper, than the time to actually capture the transfer's
intended target anchor.

Every transfer edge's label is a `PlannedTransfer`, carrying both the
`LocalTransferProblem` and its `LocalTransferResult`, so a later consumer
(`simulation.trajectory`) can replay the same dense trajectory without
re-optimizing.
"""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Iterable, TypeVar

import numpy as np

from webswing.config import BallisticDomain, PhysicalParameters, SwingConstraints
from webswing.geometry.anchors import anchor_has_line_of_sight
from webswing.geometry.buildings import City
from webswing.geometry.collision import first_point_in_region_index, point_in_destination
from webswing.optimization.local_transfer import (
    LocalTransferProblem,
    LocalTransferResult,
    TransferSimulation,
    simulate_transfer,
    solve_local_transfer,
)
from webswing.planning.cache import TransferCache, edge_cost, resulting_planning_state
from webswing.planning.heuristic import node_position, zero_heuristic
from webswing.planning.state import PlanningState, PlanningStateResolution, discretize_state, representative_state

Node = TypeVar("Node", bound=Hashable)

GOAL_NODE = "__DESTINATION_REACHED__"
"""Sentinel node representing direct destination-region interception, distinct
from any real `PlanningState` (a different, non-equal type)."""


@dataclass(frozen=True)
class SearchEdge(Generic[Node]):
    """One traversed edge in a reconstructed A* path.

    Parameters
    ----------
    from_node, to_node : Node
        Endpoints of the edge.
    cost : float
        Edge cost (elapsed travel time for `plan_route`).
    label : object
        Caller-defined payload attached to the edge (for `plan_route`, the
        `LocalTransferResult` that produced it).
    """

    from_node: Node
    to_node: Node
    cost: float
    label: object


@dataclass(frozen=True)
class SearchResult(Generic[Node]):
    """Outcome of an `astar_search` (or `plan_route`) call.

    Parameters
    ----------
    success : bool
        Whether the goal was reached.
    path : tuple[SearchEdge, ...]
        Edges from start to goal, in order. Empty if `success` is False.
    total_cost : float
        Sum of the path's edge costs; `math.inf` if `success` is False.
    nodes_expanded, nodes_generated : int
        Search-effort counters.
    failure_reason : str or None
        Human-readable reason `success` is False; None if `success` is True.
    """

    success: bool
    path: tuple[SearchEdge[Node], ...]
    total_cost: float
    nodes_expanded: int
    nodes_generated: int
    failure_reason: str | None


def _reconstruct_path(
    goal: Node, parent: dict[Node, tuple[Node, SearchEdge[Node]]]
) -> tuple[SearchEdge[Node], ...]:
    edges: list[SearchEdge[Node]] = []
    current = goal
    while current in parent:
        previous, edge = parent[current]
        edges.append(edge)
        current = previous
    edges.reverse()
    return tuple(edges)


def astar_search(
    start: Node,
    is_goal: Callable[[Node], bool],
    neighbors: Callable[[Node], Iterable[tuple[Node, float, object]]],
    heuristic: Callable[[Node], float] | None = None,
    max_expansions: int = 100_000,
) -> SearchResult[Node]:
    """Run A* (or, with the default heuristic, Dijkstra) over an implicit graph.

    Parameters
    ----------
    start : Node
        Starting node.
    is_goal : Callable[[Node], bool]
        Goal test.
    neighbors : Callable[[Node], Iterable[tuple[Node, float, object]]]
        For a node, yields `(neighbor, edge_cost, label)` triples. A yielded
        edge with a negative or non-finite cost is treated as infeasible
        and skipped.
    heuristic : Callable[[Node], float] or None, optional
        Admissible lower bound on remaining cost to the goal. Defaults to
        the constant zero heuristic (see module docstring on consistency).
    max_expansions : int, optional
        Maximum number of nodes to pop and expand before reporting failure.

    Returns
    -------
    SearchResult[Node]
        See `SearchResult` for field descriptions.
    """
    if heuristic is None:
        heuristic = lambda _node: 0.0  # noqa: E731

    counter = itertools.count()
    g_score: dict[Node, float] = {start: 0.0}
    parent: dict[Node, tuple[Node, SearchEdge[Node]]] = {}
    closed: set[Node] = set()
    open_heap: list[tuple[float, int, Node]] = [(heuristic(start), next(counter), start)]
    nodes_expanded = 0
    nodes_generated = 1

    while open_heap:
        _, _, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        closed.add(node)
        nodes_expanded += 1

        if is_goal(node):
            return SearchResult(
                success=True,
                path=_reconstruct_path(node, parent),
                total_cost=g_score[node],
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                failure_reason=None,
            )

        if nodes_expanded >= max_expansions:
            return SearchResult(
                success=False,
                path=(),
                total_cost=math.inf,
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                failure_reason=f"search exceeded max_expansions={max_expansions} without reaching the goal",
            )

        for neighbor, cost, label in neighbors(node):
            if not (math.isfinite(cost) and cost >= 0.0):
                continue
            if neighbor in closed:
                continue
            tentative_g = g_score[node] + cost
            if tentative_g < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative_g
                parent[neighbor] = (node, SearchEdge(node, neighbor, cost, label))
                heapq.heappush(open_heap, (tentative_g + heuristic(neighbor), next(counter), neighbor))
                nodes_generated += 1

    return SearchResult(
        success=False,
        path=(),
        total_cost=math.inf,
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        failure_reason="open set exhausted without reaching the goal",
    )


@dataclass(frozen=True)
class _AnchorInfo:
    position: tuple[float, float]
    owning_building_id: str | None


def _enumerate_city_anchors(city: City) -> dict[str, _AnchorInfo]:
    anchors: dict[str, _AnchorInfo] = {}
    for building in city.buildings:
        for index, point in enumerate(building.candidate_anchors):
            anchors[f"{building.building_id}#{index}"] = _AnchorInfo(
                position=point, owning_building_id=building.building_id
            )
    return anchors


@dataclass(frozen=True)
class PlannedTransfer:
    """A transfer edge's problem and solved result, carried as an edge label.

    Carrying `problem` alongside `result` lets a caller replay the edge's
    exact dense trajectory later (`simulation.trajectory`) via
    `optimization.local_transfer.simulate_transfer(result.solution, problem)`
    -- without re-optimizing, and without needing to reconstruct which
    anchor, incoming state, and physical parameters produced this edge.
    """

    problem: LocalTransferProblem
    result: LocalTransferResult


def _destination_interception_time(sim: TransferSimulation, city: City) -> float | None:
    if sim.ballistic_states is None or sim.ballistic_times is None:
        return None
    points = [(float(row[0]), float(row[1])) for row in sim.ballistic_states]
    index = first_point_in_region_index(points, city.destination)
    if index is None:
        return None
    return float(sim.t_release + sim.ballistic_times[index])


def plan_route(
    start_anchor_id: str,
    start_anchor_position: tuple[float, float],
    start_state: np.ndarray,
    city: City,
    params: PhysicalParameters,
    constraints: SwingConstraints,
    resolution: PlanningStateResolution,
    u_min: float,
    u_max: float,
    n_control_segments: int,
    t_release_min: float,
    t_release_max: float,
    capture_radius: float,
    ballistic_domain: BallisticDomain,
    heuristic: Callable[[PlanningState], float] | None = None,
    max_expansions: int = 1_000,
    max_attachment_range: float | None = None,
) -> SearchResult:
    """Plan a minimum-time route from a start state to `city.destination`.

    Parameters
    ----------
    start_anchor_id : str
        Identifier for the anchor `start_state` is attached to. Need not be
        one of `city`'s own candidate anchors.
    start_anchor_position : tuple[float, float]
        Position of that anchor, in meters.
    start_state : np.ndarray, shape (4,)
        Continuous attached state [theta, omega, ell, nu] at the start.
    city : City
        Environment: buildings (and their candidate anchors), and the
        destination region.
    params : PhysicalParameters
        Mass and gravitational acceleration.
    constraints : SwingConstraints
        Swing feasibility thresholds, forwarded to every local transfer.
    resolution : PlanningStateResolution
        Bin widths for discretizing planning states.
    u_min, u_max : float
        Radial control bounds, in m/s^2, forwarded to every local transfer.
    n_control_segments : int
        Number of equal-interval control segments per local transfer.
    t_release_min, t_release_max : float
        Release-time bounds, in seconds, forwarded to every local transfer.
    capture_radius : float
        Anchor capture-region radius, in meters, forwarded to every local
        transfer.
    ballistic_domain : BallisticDomain
        Ballistic domain bounds and max duration, forwarded to every local
        transfer.
    heuristic : Callable[[PlanningState], float] or None, optional
        Base heuristic with the `planning.heuristic` signature
        `(state, resolution, anchor_position, destination) -> float`.
        Defaults to `zero_heuristic` (see module docstring on consistency).
    max_expansions : int, optional
        Search-size termination limit.
    max_attachment_range : float or None, optional
        Forwarded to `planning.cache.resulting_planning_state`.

    Returns
    -------
    SearchResult[PlanningState]
        `SearchEdge.label` on each edge is a `PlannedTransfer` (the
        `LocalTransferProblem` and the `LocalTransferResult` that produced
        it). `SearchEdge.to_node` is `GOAL_NODE` for an edge that reaches
        the destination by ballistic interception rather than by capturing
        its intended target anchor.
    """
    base_heuristic = heuristic if heuristic is not None else zero_heuristic

    anchor_positions: dict[str, _AnchorInfo] = {
        start_anchor_id: _AnchorInfo(position=start_anchor_position, owning_building_id=None)
    }
    anchor_positions.update(_enumerate_city_anchors(city))

    cache = TransferCache()
    start_node = discretize_state(start_anchor_id, start_state, resolution)

    def node_heuristic(node: PlanningState | str) -> float:
        if node == GOAL_NODE:
            return 0.0
        anchor_position = anchor_positions[node.anchor_id].position
        return base_heuristic(node, resolution, anchor_position, city.destination)

    def is_goal(node: PlanningState | str) -> bool:
        if node == GOAL_NODE:
            return True
        anchor_position = anchor_positions[node.anchor_id].position
        position = node_position(node, resolution, anchor_position)
        return point_in_destination(position, city.destination)

    def neighbors(node: PlanningState | str) -> list[tuple[PlanningState | str, float, object]]:
        if node == GOAL_NODE:
            return []

        continuous_state = representative_state(node, resolution)
        current_anchor_position = anchor_positions[node.anchor_id].position
        current_body_position = node_position(node, resolution, current_anchor_position)

        edges: list[tuple[PlanningState | str, float, object]] = []
        for anchor_id, info in anchor_positions.items():
            if anchor_id == node.anchor_id:
                continue
            if not anchor_has_line_of_sight(
                current_body_position, info.owning_building_id, info.position, city
            ):
                continue

            problem = LocalTransferProblem(
                initial_state=continuous_state,
                current_anchor=current_anchor_position,
                target_anchor=info.position,
                params=params,
                constraints=constraints,
                u_min=u_min,
                u_max=u_max,
                n_control_segments=n_control_segments,
                t_release_min=t_release_min,
                t_release_max=t_release_max,
                capture_radius=capture_radius,
                ballistic_domain=ballistic_domain,
                city=city,
            )
            result: LocalTransferResult = cache.get_or_compute(
                node, anchor_id, lambda problem=problem: solve_local_transfer(problem)
            )
            label = PlannedTransfer(problem, result)

            sim = simulate_transfer(result.solution, problem)
            interception_time = _destination_interception_time(sim, city)
            if interception_time is not None:
                edges.append((GOAL_NODE, interception_time, label))

            neighbor_node = resulting_planning_state(
                result, anchor_id, info.position, params, resolution, max_attachment_range
            )
            if neighbor_node is not None:
                edges.append((neighbor_node, edge_cost(result), label))

        return edges

    return astar_search(
        start_node, is_goal, neighbors, heuristic=node_heuristic, max_expansions=max_expansions
    )
