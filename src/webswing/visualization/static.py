r"""Static overview plotting: city geometry, planned route, and trajectory trace.

Renders a single non-animated `matplotlib` figure summarizing a planning
run: building polygons, the ground plane, the destination region, every
candidate anchor, the anchors actually used by the planned route, the
start location, and the full trajectory trace with attached ("swing") and
ballistic portions in visually distinguishable line styles, per CLAUDE.md's
Visualization Requirements.

Scope relative to `animation.py` and `hud.py` (not yet built)
-------------------------------------------------------------------
This module renders only elements that make sense in a single static image
-- there is no "current time," so it does not draw the active web line or
the current-position/velocity markers CLAUDE.md lists among the animation
requirements; those are inherently per-frame concepts and belong to
`animation.py`. Likewise, no HUD text (simulation time, current tension,
etc.) is drawn here; that is `hud.py`'s responsibility.

Every function here only reads already-computed `City`,
`simulation.trajectory.Trajectory`, and `planning.astar.SearchEdge` data --
no `solve_ivp` or `scipy.optimize` call occurs anywhere in this module.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle

from webswing.geometry.buildings import City
from webswing.planning.astar import GOAL_NODE, PlannedTransfer, SearchEdge
from webswing.simulation.trajectory import Trajectory

Point = tuple[float, float]

_SWING_STYLE = dict(color="tab:blue", linestyle="-", linewidth=1.5, label="swing")
_BALLISTIC_STYLE = dict(color="tab:orange", linestyle="--", linewidth=1.5, label="ballistic")
_BUILDING_STYLE = dict(facecolor="0.75", edgecolor="0.3", linewidth=1.0, zorder=1)
_DESTINATION_STYLE = dict(facecolor="tab:green", alpha=0.25, edgecolor="tab:green", zorder=1)
_GROUND_STYLE = dict(color="0.3", linewidth=1.5, zorder=1)
_CANDIDATE_ANCHOR_STYLE = dict(marker="o", s=15, color="0.5", zorder=2, label="candidate anchor")
_SELECTED_ANCHOR_STYLE = dict(marker="o", s=45, color="tab:red", zorder=3, label="selected anchor")
_START_STYLE = dict(marker="*", s=200, color="tab:purple", zorder=4, label="start")


def plot_city(ax: Axes, city: City, show_candidate_anchors: bool = True) -> None:
    """Draw the ground plane, building polygons, and destination region onto `ax`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    city : City
        Environment to draw.
    show_candidate_anchors : bool, optional
        Whether to scatter every candidate anchor in `city`. Defaults to
        True.
    """
    ax.axhline(y=0.0, **_GROUND_STYLE)

    for building in city.buildings:
        ax.add_patch(Polygon(building.vertices, closed=True, **_BUILDING_STYLE))

    destination = city.destination
    ax.add_patch(
        Rectangle(
            (destination.x_min, destination.y_min),
            destination.x_max - destination.x_min,
            destination.y_max - destination.y_min,
            **_DESTINATION_STYLE,
        )
    )

    if show_candidate_anchors:
        anchors = [point for _building_id, point in city.all_candidate_anchors()]
        if anchors:
            xs, ys = zip(*anchors)
            ax.scatter(xs, ys, **_CANDIDATE_ANCHOR_STYLE)


def plot_start(ax: Axes, start_position: Point) -> None:
    """Draw a marker at the start location.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    start_position : tuple[float, float]
        Start position, in meters.
    """
    ax.scatter([start_position[0]], [start_position[1]], **_START_STYLE)


def _contiguous_mode_runs(modes: tuple[str, ...]) -> list[tuple[str, int, int]]:
    """Return (mode, start_index, end_index_inclusive) for each contiguous same-mode run."""
    if not modes:
        return []
    runs: list[tuple[str, int, int]] = []
    run_start = 0
    for i in range(1, len(modes)):
        if modes[i] != modes[run_start]:
            runs.append((modes[run_start], run_start, i - 1))
            run_start = i
    runs.append((modes[run_start], run_start, len(modes) - 1))
    return runs


def plot_trajectory(ax: Axes, trajectory: Trajectory) -> None:
    """Draw the full trajectory trace, split into swing and ballistic runs.

    Each contiguous run of samples sharing a mode is drawn as its own line
    segment (a "swing" run in a solid style, a "ballistic" run dashed), so a
    multi-hop route alternating modes several times renders with the
    correct style at every point along it.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    trajectory : Trajectory
        Trajectory to draw.
    """
    seen_labels: set[str] = set()
    for mode, start, end in _contiguous_mode_runs(trajectory.modes):
        style = dict(_SWING_STYLE if mode == "swing" else _BALLISTIC_STYLE)
        if style["label"] in seen_labels:
            del style["label"]
        else:
            seen_labels.add(style["label"])
        segment = trajectory.positions[start : end + 1]
        ax.plot(segment[:, 0], segment[:, 1], **style)


def plot_selected_anchors(ax: Axes, path: tuple[SearchEdge, ...]) -> None:
    """Draw the anchors actually used by a planned route's edges.

    Anchor positions are read directly from each edge's `PlannedTransfer`
    (`problem.current_anchor` and, unless the edge ends at
    `planning.astar.GOAL_NODE`, `problem.target_anchor`), not from a
    separate anchor lookup.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    path : tuple[SearchEdge, ...]
        `SearchResult.path` from `planning.astar.plan_route`. Every edge's
        `label` must be a `PlannedTransfer`.
    """
    positions: set[Point] = set()
    for edge in path:
        if not isinstance(edge.label, PlannedTransfer):
            raise TypeError(
                "plot_selected_anchors requires a path whose edge labels are "
                f"PlannedTransfer (as produced by planning.astar.plan_route); "
                f"got {type(edge.label)!r}"
            )
        positions.add(edge.label.problem.current_anchor)
        if edge.to_node != GOAL_NODE:
            positions.add(edge.label.problem.target_anchor)

    if positions:
        xs, ys = zip(*positions)
        ax.scatter(xs, ys, **_SELECTED_ANCHOR_STYLE)


def render_static_overview(
    city: City,
    path: tuple[SearchEdge, ...],
    trajectory: Trajectory,
    start_position: Point,
    figsize: tuple[float, float] = (10.0, 8.0),
) -> Figure:
    """Render a complete static overview figure.

    Combines `plot_city`, `plot_trajectory`, `plot_selected_anchors`, and
    `plot_start` onto one new figure with an equal-aspect axes and a
    legend.

    Parameters
    ----------
    city : City
        Environment to draw.
    path : tuple[SearchEdge, ...]
        `SearchResult.path` from `planning.astar.plan_route`.
    trajectory : Trajectory
        Assembled trajectory to draw (`simulation.trajectory.assemble_trajectory`).
    start_position : tuple[float, float]
        Start position, in meters.
    figsize : tuple[float, float], optional
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
        The rendered figure. The caller is responsible for saving/showing
        it; this function does not call `plt.show()`.
    """
    fig, ax = plt.subplots(figsize=figsize)

    plot_city(ax, city)
    plot_trajectory(ax, trajectory)
    plot_selected_anchors(ax, path)
    plot_start(ax, start_position)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize="small")
    return fig
