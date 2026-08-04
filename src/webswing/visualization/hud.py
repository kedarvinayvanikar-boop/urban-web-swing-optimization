r"""Dynamic HUD text: per-sample field values and formatting.

CLAUDE.md's dynamic HUD must display: simulation time; current motion
mode; position; speed; active anchor; web length; retraction/extension
speed; angular velocity; current tension; maximum permitted tension; load
factor; current planner edge; cumulative travel time. Every one of these
except "current planner edge" is already a field (or a one-line
computation from a field) of `simulation.trajectory.Trajectory` and
`simulation.evaluator.TrajectoryEvaluation` -- this module does not
recompute any physics, only reads and formats those.

"Simulation time" and "cumulative travel time" coincide in this model
(there is no paused/idle time distinct from elapsed trajectory time), so
`HudFrame` reports the same value for both, per CLAUDE.md's explicit
request for both fields rather than silently dropping one as redundant.

Current planner edge
------------------------
`Trajectory` and `TrajectoryEvaluation` are index-aligned with each other
but do not themselves record which `planning.astar.SearchEdge` produced
each sample. `edge_indices_for_path` recovers that mapping once (via the
same per-edge `simulate_transfer` replay `trajectory.py`/`evaluator.py`
already perform -- not a re-optimization, and done once as preprocessing,
not per frame), so a caller iterating frames can look up
`edge_indices[sample_index]` without re-deriving it per frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from matplotlib.axes import Axes
from matplotlib.text import Text

from webswing.config import SwingConstraints
from webswing.optimization.local_transfer import simulate_transfer
from webswing.planning.astar import GOAL_NODE, PlannedTransfer, SearchEdge
from webswing.simulation.evaluator import TrajectoryEvaluation
from webswing.simulation.trajectory import Trajectory


def edge_indices_for_path(path: tuple[SearchEdge, ...]) -> tuple[int | None, ...]:
    """Return, for each sample `assemble_trajectory(path, ...)` would produce, its edge index.

    Parameters
    ----------
    path : tuple[SearchEdge, ...]
        `SearchResult.path` from `planning.astar.plan_route`. Every edge's
        `label` must be a `PlannedTransfer`.

    Returns
    -------
    tuple[int or None, ...]
        `path` index of the edge that produced each sample, in the same
        order and count `simulation.trajectory.assemble_trajectory` and
        `simulation.evaluator.evaluate_trajectory` produce for the same
        `path`. `(None,)` if `path` is empty (the trivial single-point
        case, where no edge exists at all).

    Raises
    ------
    TypeError
        If any edge's `label` is not a `PlannedTransfer`.
    """
    if not path:
        return (None,)

    indices: list[int | None] = []
    for edge_index, edge in enumerate(path):
        if not isinstance(edge.label, PlannedTransfer):
            raise TypeError(
                "edge_indices_for_path requires a path whose edge labels are "
                f"PlannedTransfer (as produced by planning.astar.plan_route); "
                f"got {type(edge.label)!r}"
            )
        problem = edge.label.problem
        result = edge.label.result
        sim = simulate_transfer(result.solution, problem)

        indices.extend([edge_index] * len(sim.swing_times))

        ballistic_times_local = sim.ballistic_times
        if ballistic_times_local is not None and edge.to_node == GOAL_NODE:
            cutoff_local_time = edge.cost - result.release_time
            ballistic_times_local = ballistic_times_local[
                ballistic_times_local <= cutoff_local_time + 1.0e-9
            ]
        n_ballistic = 0 if ballistic_times_local is None else len(ballistic_times_local)
        indices.extend([edge_index] * n_ballistic)

    return tuple(indices)


@dataclass(frozen=True)
class HudFrame:
    """All HUD field values for a single trajectory sample.

    Parameters
    ----------
    simulation_time : float
        Elapsed time, in seconds.
    mode : str
        "swing" or "ballistic".
    position : tuple[float, float]
        Cartesian position, in meters.
    speed : float
        `|velocity|`, in m/s.
    active_anchor_id : str or None
        Attached anchor identifier, or None during ballistic samples.
    web_length : float
        In meters; NaN during ballistic samples.
    radial_rate : float
        l_dot, in m/s; NaN during ballistic samples.
    angular_rate : float
        theta_dot, in rad/s; NaN during ballistic samples.
    tension : float
        In newtons; NaN during ballistic samples.
    tension_max : float
        Configured maximum permitted tension, in newtons.
    load_factor : float
        T / (m * g), dimensionless; NaN during ballistic samples.
    edge_index : int or None
        `path` index of the planner edge that produced this sample, or
        None if no edge exists (the trivial already-at-destination case).
    cumulative_travel_time : float
        Equal to `simulation_time` in this model; see module docstring.
    """

    simulation_time: float
    mode: str
    position: tuple[float, float]
    speed: float
    active_anchor_id: str | None
    web_length: float
    radial_rate: float
    angular_rate: float
    tension: float
    tension_max: float
    load_factor: float
    edge_index: int | None
    cumulative_travel_time: float


def build_hud_frame(
    trajectory: Trajectory,
    evaluation: TrajectoryEvaluation,
    edge_indices: tuple[int | None, ...],
    constraints: SwingConstraints,
    sample_index: int,
) -> HudFrame:
    """Build the HUD field values for one trajectory sample.

    Parameters
    ----------
    trajectory : Trajectory
        Assembled trajectory.
    evaluation : TrajectoryEvaluation
        Force-derived quantities for the same path `trajectory` was built
        from, index-aligned with it.
    edge_indices : tuple[int or None, ...]
        From `edge_indices_for_path`, for the same path.
    constraints : SwingConstraints
        Supplies `tension_max`.
    sample_index : int
        Index into `trajectory`/`evaluation`/`edge_indices`.

    Returns
    -------
    HudFrame
        See `HudFrame` for field descriptions.

    Raises
    ------
    IndexError
        If `sample_index` is out of range.
    """
    n = len(trajectory.times)
    if not (0 <= sample_index < n):
        raise IndexError(f"sample_index {sample_index!r} out of range for trajectory of length {n}")

    vx, vy = trajectory.velocities[sample_index]
    return HudFrame(
        simulation_time=float(trajectory.times[sample_index]),
        mode=trajectory.modes[sample_index],
        position=(float(trajectory.positions[sample_index, 0]), float(trajectory.positions[sample_index, 1])),
        speed=float(math.hypot(vx, vy)),
        active_anchor_id=trajectory.active_anchor_ids[sample_index],
        web_length=float(trajectory.web_lengths[sample_index]),
        radial_rate=float(trajectory.radial_rates[sample_index]),
        angular_rate=float(trajectory.angular_rates[sample_index]),
        tension=float(evaluation.tensions[sample_index]),
        tension_max=constraints.tension_max,
        load_factor=float(evaluation.load_factors[sample_index]),
        edge_index=edge_indices[sample_index],
        cumulative_travel_time=float(trajectory.times[sample_index]),
    )


def _format_or_na(value: float, fmt: str) -> str:
    return "n/a" if math.isnan(value) else format(value, fmt)


def format_hud_text(frame: HudFrame) -> str:
    """Render a `HudFrame` as a multi-line HUD text block.

    Parameters
    ----------
    frame : HudFrame
        Field values to format.

    Returns
    -------
    str
        Multi-line text, NaN swing-only fields rendered as "n/a".
    """
    anchor_str = frame.active_anchor_id if frame.active_anchor_id is not None else "none (in flight)"
    edge_str = f"edge {frame.edge_index}" if frame.edge_index is not None else "none"

    lines = [
        f"t = {frame.simulation_time:.2f} s    mode: {frame.mode}",
        f"position: ({frame.position[0]:.2f}, {frame.position[1]:.2f}) m    speed: {frame.speed:.2f} m/s",
        f"anchor: {anchor_str}    web length: {_format_or_na(frame.web_length, '.2f')} m"
        f"    l_dot: {_format_or_na(frame.radial_rate, '+.2f')} m/s",
        f"omega: {_format_or_na(frame.angular_rate, '+.2f')} rad/s",
        f"tension: {_format_or_na(frame.tension, '.1f')} / {frame.tension_max:.1f} N"
        f"    load factor: {_format_or_na(frame.load_factor, '.2f')}",
        f"planner edge: {edge_str}    cumulative travel time: {frame.cumulative_travel_time:.2f} s",
    ]
    return "\n".join(lines)


def draw_hud(ax: Axes, frame: HudFrame, loc: tuple[float, float] = (0.02, 0.02)) -> Text:
    """Draw a `HudFrame` as a text block anchored to `ax`'s bottom-left corner.

    Defaults to the bottom-left rather than the top-left: start anchors in
    this domain are almost always elevated (rooftops), so a top-anchored
    HUD tends to collide with the start marker and the beginning of the
    swing trace, while the ground-level region near a destination region is
    comparatively empty.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    frame : HudFrame
        Field values to draw.
    loc : tuple[float, float], optional
        Anchor position in axes-fraction coordinates (0-1). Defaults to
        the bottom-left corner.

    Returns
    -------
    matplotlib.text.Text
        The created text artist.
    """
    return ax.text(
        loc[0],
        loc[1],
        format_hud_text(frame),
        transform=ax.transAxes,
        verticalalignment="bottom",
        horizontalalignment="left",
        fontsize=9,
        family="monospace",
        color="#ECEFF1",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#05070D",
            edgecolor="#0D2F87",
            linewidth=1.5,
            alpha=0.92,
        ),
    )
