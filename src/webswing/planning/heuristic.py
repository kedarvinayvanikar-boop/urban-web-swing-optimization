r"""A* heuristics for the global trajectory-planning search over `PlanningState` nodes.

CLAUDE.md permits `h(n) = d_straight(n, goal) / v_max` as an admissible
lower bound on remaining travel time, but only when `v_max` is a valid hard
upper bound on achievable speed under the model, and requires falling back
to `h(n) = 0` (reducing A* to Dijkstra, still correct) whenever no such
bound has been established.

No v_max has been derived or certified in this repository
--------------------------------------------------------------
Establishing a true hard speed bound for this model is nontrivial: ballistic
speed can grow with the domain's height range via gravity, and swing speed
can additionally grow from work done by the radial control over however
much of the remaining path is still ahead, which is not obviously bounded
without further assumptions (e.g. a maximum total control-energy budget).
No such derivation has been carried out here (it belongs in
`docs/methodology.md`'s "A* heuristic admissibility" section as a proven
result, not asserted in code), so `zero_heuristic` is the default and the
recommended choice for `planning.astar` until a `v_max` is independently
derived and justified. `speed_bound_heuristic` is provided for that future
use, but it is the caller's responsibility to supply a `v_max` they can
justify -- an unjustified or too-large `v_max` silently breaks A*'s
optimality guarantee without causing any runtime error.

Goal representation
----------------------
The planning goal is the destination region (CLAUDE.md: "a specified start
state to a destination region"), not a specific anchor. A node's
representative physical position is derived from its anchor's fixed
position and its bin-center attachment state (`planning.state`), reusing
the same coordinate mapping as `dynamics.release.attached_position` rather
than re-deriving it.
"""

from __future__ import annotations

import math

from webswing.dynamics.release import attached_position
from webswing.geometry.buildings import DestinationRegion
from webswing.planning.state import PlanningState, PlanningStateResolution, representative_state


def node_position(
    state: PlanningState, resolution: PlanningStateResolution, anchor_position: tuple[float, float]
) -> tuple[float, float]:
    """Return the representative (x, y) position of a planning node.

    Uses the bin-center attachment state (`representative_state`) together
    with the fixed position of the anchor `state` is attached to.

    Parameters
    ----------
    state : PlanningState
        Planning-graph node.
    resolution : PlanningStateResolution
        Bin widths `state` was discretized under.
    anchor_position : tuple[float, float]
        Position of the anchor identified by `state.anchor_id`, in meters.

    Returns
    -------
    tuple[float, float]
        Representative (x, y) position, in meters.
    """
    theta, _omega, ell, _nu = representative_state(state, resolution)
    return attached_position(theta, ell, anchor_position[0], anchor_position[1])


def distance_to_region(point: tuple[float, float], region: DestinationRegion) -> float:
    """Return the Euclidean distance from a point to the nearest point of a rectangular region.

    Parameters
    ----------
    point : tuple[float, float]
        Query point, in meters.
    region : DestinationRegion
        Axis-aligned destination region.

    Returns
    -------
    float
        Distance, in meters. Zero if `point` is inside or on the boundary
        of `region`.
    """
    x, y = point
    dx = max(region.x_min - x, 0.0, x - region.x_max)
    dy = max(region.y_min - y, 0.0, y - region.y_max)
    return math.hypot(dx, dy)


def zero_heuristic(
    state: PlanningState,
    resolution: PlanningStateResolution,
    anchor_position: tuple[float, float],
    destination: DestinationRegion,
) -> float:
    """Return 0.0 unconditionally: the trivially admissible heuristic.

    Reduces A* to Dijkstra's algorithm while preserving correctness
    (CLAUDE.md); the recommended default until a hard speed bound is
    derived and justified (see module docstring). Takes the same signature
    as `speed_bound_heuristic` so `planning.astar` can select either
    interchangeably.

    Returns
    -------
    float
        Always 0.0.
    """
    return 0.0


def speed_bound_heuristic(
    state: PlanningState,
    resolution: PlanningStateResolution,
    anchor_position: tuple[float, float],
    destination: DestinationRegion,
    v_max: float,
) -> float:
    """Return `d_straight(node, destination) / v_max`.

    Admissible only if `v_max` is a genuine hard upper bound on achievable
    speed under this model -- not established or certified by this
    repository (see module docstring). Supplying too large a `v_max`
    silently breaks A*'s optimality guarantee.

    Parameters
    ----------
    state : PlanningState
        Planning-graph node.
    resolution : PlanningStateResolution
        Bin widths `state` was discretized under.
    anchor_position : tuple[float, float]
        Position of the anchor identified by `state.anchor_id`, in meters.
    destination : DestinationRegion
        Goal region.
    v_max : float
        Caller-supplied, caller-justified hard upper bound on achievable
        speed, in m/s. Must be finite and strictly positive.

    Returns
    -------
    float
        Straight-line distance to `destination` divided by `v_max`.

    Raises
    ------
    ValueError
        If `v_max` is not finite and strictly positive.
    """
    if not (math.isfinite(v_max) and v_max > 0.0):
        raise ValueError(f"v_max must be finite and strictly positive, got {v_max!r}")
    point = node_position(state, resolution, anchor_position)
    return distance_to_region(point, destination) / v_max
