r"""Assembly of a planned route into one continuous, global-time trajectory.

`planning.astar.plan_route` returns a sequence of edges, each carrying a
`PlannedTransfer` (a `LocalTransferProblem` plus its solved
`LocalTransferResult`) and its own release/ballistic timeline starting from
t = 0. `assemble_trajectory` replays every edge's already-solved decision
vector (`optimization.local_transfer.simulate_transfer`, an inexpensive
re-integration, not a re-optimization) and concatenates the per-edge swing
and ballistic samples into one global timeline, converting Cartesian
positions via `dynamics.release.attached_position`/`attached_velocity`
during swing portions. This is the "must not rerun the optimizer... while
rendering" requirement from CLAUDE.md's Visualization Requirements: once
assembled, a `Trajectory` is pure stored data, replayable by
`simulation.evaluator` and `visualization` without touching `solve_ivp` or
`scipy.optimize` again.

What counts as the trajectory's starting point
--------------------------------------------------
The first edge's own t=0 sample (the bin-center *representative* state
`plan_route` actually solved the first transfer from) is used as the
trajectory's starting point, not the caller's original continuous
`start_state` -- those two generally differ slightly, since
`plan_route` discretizes `start_state` into a `PlanningState` before
solving anything, and every local transfer is optimized from that bin's
*representative* state, not the exact original value. Showing what was
actually optimized and would actually be executed is more useful and more
honest than an artificial preamble point that doesn't match any edge's own
initial condition. The one exception is a `path` of zero edges (the start
was already inside the destination region, so nothing was ever solved) --
there, `assemble_trajectory` falls back to the raw `start_state` directly,
since no edge exists to derive a point from.

Destination-interception truncation
----------------------------------------
For an edge ending at `planning.astar.GOAL_NODE` (destination reached
mid-flight), only the ballistic samples up to and including first entry
into the destination region are kept; later samples belong to a ballistic
arc that was never actually flown once the destination was reached. The
edge's own `cost` (not the last kept sample's time, which sits on a fixed
sampling grid and does not land exactly on the entry instant) is used to
advance the global clock, so the assembled trajectory's total duration
matches `SearchResult.total_cost` exactly rather than accumulating grid
rounding error across edges.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from webswing.dynamics.release import attached_position, attached_velocity
from webswing.planning.astar import GOAL_NODE, PlannedTransfer, SearchEdge
from webswing.optimization.local_transfer import simulate_transfer

Point = tuple[float, float]


@dataclass(frozen=True)
class Trajectory:
    """A stitched, global-time trajectory ready for replay/visualization.

    Parameters
    ----------
    times : np.ndarray, shape (N,)
        Global elapsed time at each sample, in seconds, non-decreasing.
    positions : np.ndarray, shape (N, 2)
        Cartesian (x, y) position at each sample, in meters.
    velocities : np.ndarray, shape (N, 2)
        Cartesian (vx, vy) velocity at each sample, in m/s.
    modes : tuple[str, ...], length N
        "swing" or "ballistic" at each sample.
    active_anchor_ids : tuple[str or None, ...], length N
        Anchor identifier the body is attached to at each sample; None
        during ballistic samples.
    web_lengths : np.ndarray, shape (N,)
        Web length at each sample, in meters; NaN during ballistic samples.
    angular_rates : np.ndarray, shape (N,)
        Angular rate theta_dot at each sample, in rad/s; NaN during
        ballistic samples.
    radial_rates : np.ndarray, shape (N,)
        Radial rate l_dot at each sample, in m/s; NaN during ballistic
        samples.

    Raises
    ------
    ValueError
        If the array/tuple lengths are inconsistent.
    """

    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    modes: tuple[str, ...]
    active_anchor_ids: tuple[str | None, ...]
    web_lengths: np.ndarray
    angular_rates: np.ndarray
    radial_rates: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.times)
        for name, value in (
            ("positions", self.positions),
            ("velocities", self.velocities),
            ("web_lengths", self.web_lengths),
            ("angular_rates", self.angular_rates),
            ("radial_rates", self.radial_rates),
        ):
            if len(value) != n:
                raise ValueError(f"{name} has length {len(value)}, expected {n} (matching times)")
        if len(self.modes) != n or len(self.active_anchor_ids) != n:
            raise ValueError("modes and active_anchor_ids must have the same length as times")


def _single_point_trajectory(
    anchor_id: str, anchor_position: Point, state: np.ndarray
) -> Trajectory:
    theta, omega, ell, nu = state
    position = attached_position(theta, ell, anchor_position[0], anchor_position[1])
    velocity = attached_velocity(theta, omega, ell, nu)
    return Trajectory(
        times=np.array([0.0]),
        positions=np.array([position]),
        velocities=np.array([velocity]),
        modes=("swing",),
        active_anchor_ids=(anchor_id,),
        web_lengths=np.array([float(ell)]),
        angular_rates=np.array([float(omega)]),
        radial_rates=np.array([float(nu)]),
    )


def assemble_trajectory(
    path: tuple[SearchEdge, ...],
    start_anchor_id: str,
    start_anchor_position: Point,
    start_state: np.ndarray,
) -> Trajectory:
    """Assemble a planned route's edges into one continuous, global-time trajectory.

    Parameters
    ----------
    path : tuple[SearchEdge, ...]
        `SearchResult.path` from `planning.astar.plan_route`. Every edge's
        `label` must be a `PlannedTransfer`.
    start_anchor_id : str
        Identifier of the anchor `start_state` is attached to (used only
        when `path` is empty; see module docstring).
    start_anchor_position : tuple[float, float]
        Position of that anchor, in meters (used only when `path` is
        empty).
    start_state : np.ndarray, shape (4,)
        Continuous attached state [theta, omega, ell, nu] at the start
        (used only when `path` is empty).

    Returns
    -------
    Trajectory
        The assembled trajectory.

    Raises
    ------
    TypeError
        If any edge's `label` is not a `PlannedTransfer`.
    """
    if not path:
        return _single_point_trajectory(start_anchor_id, start_anchor_position, start_state)

    times: list[float] = []
    positions: list[Point] = []
    velocities: list[Point] = []
    modes: list[str] = []
    active_anchor_ids: list[str | None] = []
    web_lengths: list[float] = []
    angular_rates: list[float] = []
    radial_rates: list[float] = []

    edge_start_time = 0.0
    for edge in path:
        if not isinstance(edge.label, PlannedTransfer):
            raise TypeError(
                "assemble_trajectory requires a path whose edge labels are "
                f"PlannedTransfer (as produced by planning.astar.plan_route); "
                f"got {type(edge.label)!r}"
            )
        problem = edge.label.problem
        result = edge.label.result
        sim = simulate_transfer(result.solution, problem)

        for t_local, state in zip(sim.swing_times, sim.swing_states):
            theta, omega, ell, nu = state
            position = attached_position(theta, ell, problem.current_anchor[0], problem.current_anchor[1])
            velocity = attached_velocity(theta, omega, ell, nu)
            times.append(edge_start_time + float(t_local))
            positions.append(position)
            velocities.append(velocity)
            modes.append("swing")
            active_anchor_ids.append(edge.from_node.anchor_id)
            web_lengths.append(float(ell))
            angular_rates.append(float(omega))
            radial_rates.append(float(nu))

        ballistic_times_local = sim.ballistic_times
        ballistic_states = sim.ballistic_states
        if ballistic_times_local is not None and edge.to_node == GOAL_NODE:
            cutoff_local_time = edge.cost - result.release_time
            keep = ballistic_times_local <= cutoff_local_time + 1.0e-9
            ballistic_times_local = ballistic_times_local[keep]
            ballistic_states = ballistic_states[keep]

        if ballistic_times_local is not None:
            for t_local, state in zip(ballistic_times_local, ballistic_states):
                x, y, vx, vy = state
                times.append(edge_start_time + result.release_time + float(t_local))
                positions.append((float(x), float(y)))
                velocities.append((float(vx), float(vy)))
                modes.append("ballistic")
                active_anchor_ids.append(None)
                web_lengths.append(math.nan)
                angular_rates.append(math.nan)
                radial_rates.append(math.nan)

        edge_start_time += edge.cost

    return Trajectory(
        times=np.array(times),
        positions=np.array(positions),
        velocities=np.array(velocities),
        modes=tuple(modes),
        active_anchor_ids=tuple(active_anchor_ids),
        web_lengths=np.array(web_lengths),
        angular_rates=np.array(angular_rates),
        radial_rates=np.array(radial_rates),
    )
