r"""Top-level orchestration: plan a route, then assemble and evaluate its trajectory once.

This module introduces no new algorithms. It exists so a caller (and,
eventually, `visualization`) can produce the complete stored output of a
run -- the plan, its stitched `Trajectory`, and its force-derived
`TrajectoryEvaluation` -- with a single call, computed exactly once.
CLAUDE.md's Visualization Requirements are explicit that animation code
"must not rerun the optimizer or numerical integrator while rendering";
`run_simulation` is the one place that does the actual planning and
replaying, so everything downstream (HUD, static plots, animation) only
ever reads the resulting `SimulationRun`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from webswing.config import BallisticDomain, PhysicalParameters, SwingConstraints
from webswing.geometry.buildings import City
from webswing.planning.astar import SearchResult, plan_route
from webswing.planning.state import PlanningState, PlanningStateResolution
from webswing.simulation.evaluator import TrajectoryEvaluation, evaluate_trajectory
from webswing.simulation.trajectory import Trajectory, assemble_trajectory


@dataclass(frozen=True)
class SimulationRun:
    """The complete stored output of one planning-and-replay run.

    Parameters
    ----------
    search_result : SearchResult
        The raw A* search outcome (path, cost, effort counters, failure
        reason if applicable).
    trajectory : Trajectory or None
        The stitched, global-time trajectory, or None if `success` is
        False (no route was found, so there is nothing to assemble).
    evaluation : TrajectoryEvaluation or None
        Force-derived quantities index-aligned with `trajectory`, or None
        if `success` is False.
    success : bool
        Whether a route was found. Mirrors `search_result.success`.
    failure_reason : str or None
        Mirrors `search_result.failure_reason`.
    """

    search_result: SearchResult
    trajectory: Trajectory | None
    evaluation: TrajectoryEvaluation | None
    success: bool
    failure_reason: str | None


def run_simulation(
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
) -> SimulationRun:
    """Plan a route and assemble/evaluate its complete stored trajectory in one call.

    Every parameter is forwarded directly to `planning.astar.plan_route`;
    see that function for parameter descriptions. On success, the
    resulting path is also passed through `simulation.trajectory.
    assemble_trajectory` and `simulation.evaluator.evaluate_trajectory`.

    Returns
    -------
    SimulationRun
        See `SimulationRun` for field descriptions.
    """
    search_result = plan_route(
        start_anchor_id=start_anchor_id,
        start_anchor_position=start_anchor_position,
        start_state=start_state,
        city=city,
        params=params,
        constraints=constraints,
        resolution=resolution,
        u_min=u_min,
        u_max=u_max,
        n_control_segments=n_control_segments,
        t_release_min=t_release_min,
        t_release_max=t_release_max,
        capture_radius=capture_radius,
        ballistic_domain=ballistic_domain,
        heuristic=heuristic,
        max_expansions=max_expansions,
        max_attachment_range=max_attachment_range,
    )

    if not search_result.success:
        return SimulationRun(
            search_result=search_result,
            trajectory=None,
            evaluation=None,
            success=False,
            failure_reason=search_result.failure_reason,
        )

    trajectory = assemble_trajectory(
        search_result.path, start_anchor_id, start_anchor_position, start_state
    )
    evaluation = evaluate_trajectory(search_result.path, start_state, params, constraints)

    return SimulationRun(
        search_result=search_result,
        trajectory=trajectory,
        evaluation=evaluation,
        success=True,
        failure_reason=None,
    )
