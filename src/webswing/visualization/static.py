r"""Static overview plotting: city geometry, planned route, and trajectory trace.

Renders a single non-animated `matplotlib` figure summarizing a planning
run: building polygons, the ground plane, the destination region, every
candidate anchor, the anchors actually used by the planned route, the
start location, and the full trajectory trace with attached ("swing") and
ballistic portions in visually distinguishable line styles, per CLAUDE.md's
Visualization Requirements.

Dark theme and glow rendering
----------------------------------
The figure uses a dark "night city" palette rather than a plain white
background: `apply_dark_theme` paints an actual vertical night-sky gradient
plus a scattered star field behind everything (a single flat fill color
reads as a placeholder; a gradient with depth does not), and sets both the
figure's and the axes' facecolor as its base (recoloring only the axes
leaves a plain white margin around the plot, which is the wrong fix), plus
tick/label/title/spine/legend colors readable against it. It also draws a
small spider silhouette dangling from a silk thread in a corner -- a
literal nod to the web-swinging mechanic this package models, not a
generic mascot. Glow is rendered with `matplotlib.patheffects.withStroke`
-- a soft halo stroke drawn under an artist's normal rendering -- applied
directly to each line/marker, not by stacking duplicate artists at
different widths/alphas. This keeps artist counts (and therefore the
structural tests in `test_static.py`) equal to the number of actual data
elements, and produces a cleaner glow than a hand-rolled multi-line stack.

Color palette
----------------
Swing (attached, on-web) motion is red, ballistic (released, free-flight)
motion is blue, and the web/silk itself (ground line, web-icon accents) is
silver-white -- a loose nod to the Spider-Man palette this project's
"Spider-Man-inspired web-swinging" framing (CLAUDE.md) draws on, without
reproducing any copyrighted character artwork.

Scope relative to `animation.py` and `hud.py`
--------------------------------------------------
This module renders only elements that make sense in a single static image
-- there is no "current time," so it does not draw the active web line or
the current-position/velocity markers CLAUDE.md lists among the animation
requirements; those are inherently per-frame concepts and belong to
`animation.py`. Likewise, no HUD text (simulation time, current tension,
etc.) is drawn here; that is `hud.py`'s responsibility. `render_static_overview`,
the top-level convenience function, does not include
`plot_constraint_failure_location` (which needs a `TrajectoryEvaluation`,
not just a `Trajectory`) -- `animation.py` wires that primitive in
directly, since it already requires an evaluation for the HUD.

Every function here only reads already-computed `City`,
`simulation.trajectory.Trajectory`, and `planning.astar.SearchEdge` data --
no `solve_ivp` or `scipy.optimize` call occurs anywhere in this module.
"""

from __future__ import annotations

import numpy as np
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, Polygon, Rectangle

from webswing.geometry.buildings import City
from webswing.planning.astar import GOAL_NODE, PlannedTransfer, SearchEdge
from webswing.simulation.evaluator import TrajectoryEvaluation
from webswing.simulation.trajectory import Trajectory

Point = tuple[float, float]

# --- Palette ------------------------------------------------------------------------

_FIGURE_BG = "#05070D"
_AXES_BG = "#05070D"
_SKY_TOP_COLOR = "#050810"
_SKY_HORIZON_COLOR = "#1A2540"
_GRID_COLOR = "#26344F"
_TEXT_COLOR = "#ECEFF1"
_MUTED_TEXT_COLOR = "#8592A6"
_SPINE_COLOR = "#2A3F5A"

_SWING_COLOR = "#FF1744"
_BALLISTIC_COLOR = "#2979FF"
_GROUND_COLOR = "#161F35"
_GROUND_LINE_COLOR = "#B0BEC5"
_BUILDING_FACE = "#161E2E"
_BUILDING_EDGE = "#3F5C8A"
_DESTINATION_COLOR = "#00E676"
_CANDIDATE_ANCHOR_COLOR = "#B0BEC5"
_SELECTED_ANCHOR_COLOR = "#FF1744"
_START_COLOR = "#ECEFF1"
_FAILURE_COLOR = "#FFD600"
_SPIDER_COLOR = "#0B0D10"
_WEB_THREAD_COLOR = "#CFD8DC"


def _glow(linewidth: float, color: str, alpha: float = 0.55) -> list:
    """Return a `path_effects` list giving an artist a soft glow halo, plus normal rendering on top."""
    return [path_effects.withStroke(linewidth=linewidth, foreground=color, alpha=alpha), path_effects.Normal()]


_SWING_STYLE = dict(
    color=_SWING_COLOR,
    linestyle="-",
    linewidth=2.2,
    zorder=3,
    label="swing",
    path_effects=_glow(6.0, _SWING_COLOR, alpha=0.5),
)
_BALLISTIC_STYLE = dict(
    color=_BALLISTIC_COLOR,
    linestyle="--",
    linewidth=2.2,
    zorder=3,
    label="ballistic",
    path_effects=_glow(6.0, _BALLISTIC_COLOR, alpha=0.5),
)
_BUILDING_STYLE = dict(facecolor=_BUILDING_FACE, edgecolor=_BUILDING_EDGE, linewidth=1.4, zorder=1)
_DESTINATION_STYLE = dict(
    facecolor=_DESTINATION_COLOR, alpha=0.18, edgecolor=_DESTINATION_COLOR, linewidth=2.2, linestyle="--", zorder=1
)
_DESTINATION_LABEL_STYLE = dict(
    ha="center",
    va="bottom",
    fontsize=9,
    fontweight="bold",
    color=_DESTINATION_COLOR,
    zorder=2,
    bbox=dict(boxstyle="round,pad=0.25", facecolor=_AXES_BG, alpha=0.85, edgecolor=_DESTINATION_COLOR, linewidth=1.0),
)
_FAILURE_STYLE = dict(
    marker="x",
    s=170,
    color=_FAILURE_COLOR,
    linewidths=3.0,
    zorder=6,
    label="constraint failure",
    path_effects=_glow(6.0, _FAILURE_COLOR, alpha=0.6),
)
_GROUND_LINE_STYLE = dict(color=_GROUND_LINE_COLOR, linewidth=1.5, alpha=0.7, zorder=1)
_CANDIDATE_ANCHOR_STYLE = dict(
    marker="o",
    s=35,
    facecolor=_CANDIDATE_ANCHOR_COLOR,
    edgecolor="#37474F",
    linewidths=1.0,
    zorder=2,
    label="candidate anchor",
)
_SELECTED_ANCHOR_STYLE = dict(
    marker="o",
    s=75,
    facecolor=_SELECTED_ANCHOR_COLOR,
    edgecolor="white",
    linewidths=1.2,
    zorder=4,
    label="selected anchor",
    path_effects=_glow(6.0, _SELECTED_ANCHOR_COLOR, alpha=0.6),
)
_START_STYLE = dict(
    marker="*",
    s=300,
    facecolor=_START_COLOR,
    edgecolor=_SWING_COLOR,
    linewidths=1.4,
    zorder=5,
    label="start",
    path_effects=_glow(8.0, _START_COLOR, alpha=0.6),
)


def _paint_night_sky(ax: Axes, n_stars: int = 70, seed: int = 7) -> None:
    """Paint a vertical night-sky gradient and a scattered star field behind everything.

    Positioned in axes-fraction coordinates (`transform=ax.transAxes`), so
    it always fills the visible plot area regardless of the data's actual
    extent. Star positions use a fixed seed so the figure is reproducible.
    """
    gradient = np.linspace(0.0, 1.0, 256).reshape(-1, 1)
    cmap = LinearSegmentedColormap.from_list("night_sky", [_SKY_HORIZON_COLOR, _SKY_TOP_COLOR])
    ax.imshow(
        gradient,
        aspect="auto",
        cmap=cmap,
        extent=(0.0, 1.0, 0.0, 1.0),
        transform=ax.transAxes,
        origin="lower",
        zorder=-10,
    )

    rng = np.random.default_rng(seed)
    star_x = rng.uniform(0.02, 0.98, n_stars)
    star_y = rng.uniform(0.40, 0.97, n_stars)
    star_alpha = rng.uniform(0.2, 0.9, n_stars)
    star_size = rng.uniform(1.0, 5.0, n_stars)
    colors = [(1.0, 1.0, 1.0, alpha) for alpha in star_alpha]
    ax.scatter(star_x, star_y, s=star_size, c=colors, transform=ax.transAxes, zorder=-9, linewidths=0)


def _draw_spider_watermark(ax: Axes) -> None:
    """Draw a small spider silhouette dangling from a silk thread in the top-left corner.

    A literal nod to the web-swinging mechanic this package models, not a
    generic mascot. Positioned in axes-fraction coordinates so it stays a
    small, fixed corner watermark regardless of the data's actual extent.
    """
    thread_top = (0.045, 1.0)
    thread_bottom = (0.045, 0.86)
    ax.plot(
        [thread_top[0], thread_bottom[0]],
        [thread_top[1], thread_bottom[1]],
        color=_WEB_THREAD_COLOR,
        linewidth=1.0,
        alpha=0.85,
        solid_capstyle="round",
        zorder=50,
        transform=ax.transAxes,
        path_effects=_glow(3.0, _WEB_THREAD_COLOR, alpha=0.4),
    )

    body_x, body_y = thread_bottom
    abdomen_center = (body_x, body_y - 0.018)
    ax.add_patch(
        Ellipse(
            abdomen_center,
            width=0.026,
            height=0.020,
            facecolor=_SPIDER_COLOR,
            edgecolor=_WEB_THREAD_COLOR,
            linewidth=0.6,
            zorder=51,
            transform=ax.transAxes,
        )
    )
    ax.add_patch(
        Ellipse(
            (body_x, body_y - 0.003),
            width=0.015,
            height=0.013,
            facecolor=_SPIDER_COLOR,
            edgecolor=_WEB_THREAD_COLOR,
            linewidth=0.6,
            zorder=51,
            transform=ax.transAxes,
        )
    )

    leg_length = 0.022
    for angle_deg in (200.0, 230.0, 260.0, -20.0, -50.0, -80.0):
        angle = np.radians(angle_deg)
        dx = leg_length * np.cos(angle)
        dy = leg_length * np.sin(angle) * 0.6
        ax.plot(
            [abdomen_center[0], abdomen_center[0] + dx],
            [abdomen_center[1], abdomen_center[1] + dy],
            color=_SPIDER_COLOR,
            linewidth=1.1,
            solid_capstyle="round",
            zorder=50,
            transform=ax.transAxes,
        )


def apply_dark_theme(fig: Figure, ax: Axes) -> None:
    """Apply the shared dark "night city" theme to a figure and its axes.

    Sets both the figure's and the axes' background (recoloring only the
    axes leaves a plain white margin around the plot), paints a night-sky
    gradient and star field, draws the spider watermark, and sets grid,
    spine, tick, and label colors readable against it. Shared by
    `render_static_overview` and `animation.render_animation` so the two
    match.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to theme.
    ax : matplotlib.axes.Axes
        Axes to theme.
    """
    fig.patch.set_facecolor(_FIGURE_BG)
    ax.set_facecolor(_AXES_BG)
    _paint_night_sky(ax)
    ax.grid(True, linestyle=":", alpha=0.25, color=_GRID_COLOR, zorder=0)
    for spine in ax.spines.values():
        spine.set_color(_SPINE_COLOR)
    ax.tick_params(colors=_MUTED_TEXT_COLOR)
    ax.xaxis.label.set_color(_TEXT_COLOR)
    ax.yaxis.label.set_color(_TEXT_COLOR)
    ax.title.set_color(_TEXT_COLOR)
    _draw_spider_watermark(ax)


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
    # Ground band depth scales with the tallest building, so it reads as a
    # proportionate strip of "ground mass" rather than a fixed-size band
    # that looks too thin or too thick depending on the scene's scale.
    city_scale = max((b.roof_elevation for b in city.buildings), default=5.0)
    ax.axhspan(-0.08 * city_scale, 0.0, facecolor=_GROUND_COLOR, zorder=1)
    ax.axhline(y=0.0, **_GROUND_LINE_STYLE)

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
    label_margin = 0.15 * (destination.y_max - destination.y_min)
    ax.text(
        0.5 * (destination.x_min + destination.x_max),
        destination.y_max + label_margin,
        "DESTINATION",
        **_DESTINATION_LABEL_STYLE,
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
    segment (a "swing" run in a solid style, a "ballistic" run dashed, each
    with a glow rendered via `path_effects`), so a multi-hop route
    alternating modes several times renders with the correct style at
    every point along it.

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


def plot_constraint_failure_location(
    ax: Axes, trajectory: Trajectory, evaluation: TrajectoryEvaluation
) -> bool:
    """Mark the first sample where tension or load-factor margin is negative, if any.

    `evaluation.tension_margins`/`load_factor_margins` are NaN during
    ballistic samples; a NaN comparison is always False, so ballistic
    samples never spuriously trigger this without any special-casing.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    trajectory : Trajectory
        Trajectory the samples' positions come from.
    evaluation : TrajectoryEvaluation
        Margins to check, index-aligned with `trajectory`.

    Returns
    -------
    bool
        True if a violation was found and marked; False if `trajectory`
        (as evaluated) is fully feasible. A route accepted by
        `optimization.local_transfer` should never actually violate its
        own constraints, so this is expected to return False in practice;
        it exists as the defensive diagnostic marker CLAUDE.md requires.
    """
    violated = np.where((evaluation.tension_margins < 0.0) | (evaluation.load_factor_margins < 0.0))[0]
    if len(violated) == 0:
        return False
    index = int(violated[0])
    ax.scatter([trajectory.positions[index, 0]], [trajectory.positions[index, 1]], **_FAILURE_STYLE)
    return True


def render_static_overview(
    city: City,
    path: tuple[SearchEdge, ...],
    trajectory: Trajectory,
    start_position: Point,
    figsize: tuple[float, float] = (10.0, 8.0),
) -> Figure:
    """Render a complete static overview figure.

    Combines `plot_city`, `plot_trajectory`, `plot_selected_anchors`, and
    `plot_start` onto one new figure with an equal-aspect axes, a dark
    theme (`apply_dark_theme`), and a legend.

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
    fig, ax = plt.subplots(figsize=figsize, dpi=140)
    apply_dark_theme(fig, ax)

    plot_city(ax, city)
    plot_trajectory(ax, trajectory)
    plot_selected_anchors(ax, path)
    plot_start(ax, start_position)

    ax.set_title(f"Planned Route Overview — total travel time {trajectory.times[-1]:.2f} s", fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    legend = ax.legend(loc="best", fontsize="small", framealpha=0.9, facecolor=_AXES_BG, edgecolor=_SPINE_COLOR)
    for text in legend.get_texts():
        text.set_color(_TEXT_COLOR)
    return fig
