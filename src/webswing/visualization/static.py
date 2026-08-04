r"""Static overview plotting: city geometry, planned route, and trajectory trace.

Renders a single non-animated `matplotlib` figure summarizing a planning
run: building volumes, the ground plane, the destination region, every
candidate anchor, the anchors actually used by the planned route, the
start location, and the full trajectory trace with attached ("swing") and
ballistic portions in visually distinguishable line styles, per CLAUDE.md's
Visualization Requirements.

Rendering-only 3D convention
----------------------------------
CLAUDE.md's physics model is strictly two-dimensional: every dynamical
quantity (position, velocity, tension, ...) lives in the (x, y) plane
defined there (x horizontal, y vertical, ground at y = 0). Nothing in this
module changes that -- `City`, `Building`, and `Trajectory` carry no third
spatial coordinate, and no dynamics/collision/optimization code is touched
here.

To render a solid-looking city rather than a flat silhouette, this module
draws on a `mpl_toolkits.mplot3d` axes using a purely cosmetic depth axis
that does not exist anywhere in the physics model:

    render_x = x        render_y = depth (synthetic)        render_z = y

`depth = 0` is the *trajectory plane* -- the flat plane the actual (x, y)
physics lives on, holding the trajectory trace, anchors, start marker, and
destination region, exactly as in the previous flat 2D rendering. Each
building's 2D silhouette (its `vertices`, defined in the real x/y plane) is
extruded symmetrically into a solid prism spanning
`depth in [-half_depth, +half_depth]` around that same plane
(`_building_render_depth`/`_extrude_polygon_faces`), so the scene reads as
a city block the trajectory threads through. `half_depth` is fixed at a
constant fraction of each building's own (physical) width purely for
visual proportion -- it is not a modelled or measured building property,
carries no physical units of consequence, and cannot affect collision,
tension, or optimization results, which never read this module.

Layered-axes compositing
----------------------------
`mpl_toolkits.mplot3d.Axes3D` can only draw artists that implement 3D
projection (`do_3d_projection`); plain 2D patches/collections added via
`ax.add_patch`/`ax.scatter(..., transform=ax.transAxes)` raise at draw time
if placed directly on it. The night-sky gradient, star field, and spider
watermark are therefore painted on a separate, fully transparent 2D axes
(`_build_background_axes`) stacked at the same figure position *underneath*
the 3D data axes (whose facecolor and axis panes are set fully transparent
so the background shows through) -- not by changing what those two paint
functions do, which remain plain-`Axes` painters unaware that a 3D scene
sits above them.

Dark theme and glow rendering
----------------------------------
The figure uses a dark "night city" palette rather than a plain white
background: `apply_dark_theme` paints an actual vertical night-sky gradient
plus a scattered star field behind everything (a single flat fill color
reads as a placeholder; a gradient with depth does not), and sets both the
figure's and the background axes' facecolor as its base, plus tick/label/
title/pane/legend colors readable against it. It also draws a small spider
silhouette dangling from a silk thread in a corner -- a literal nod to the
web-swinging mechanic this package models, not a generic mascot. Glow is
rendered with `matplotlib.patheffects.withStroke` -- a soft halo stroke
drawn under an artist's normal rendering -- applied directly to each
line/marker, not by stacking duplicate artists at different widths/alphas.
`mplot3d`'s `Line3D` and `Path3DCollection` both support `path_effects`
identically to their 2D counterparts, so this technique carries over
unchanged.

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
etc.) is drawn here; that is `hud.py`'s responsibility, and (because HUD
text is a flat UI overlay, not a 3D-projected scene element) `animation.py`
draws it on its own additional foreground 2D axes rather than on the
`Axes3D` this module builds. `render_static_overview`, the top-level
convenience function, does not include `plot_constraint_failure_location`
(which needs a `TrajectoryEvaluation`, not just a `Trajectory`) --
`animation.py` wires that primitive in directly, since it already requires
an evaluation for the HUD.

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
from matplotlib.patches import Circle, Ellipse
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D

from webswing.geometry.buildings import Building, City
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

_SWING_COLOR = "#8B0018"
_BALLISTIC_COLOR = "#0D2F87"
_GROUND_COLOR = "#161F35"
_GROUND_LINE_COLOR = "#B0BEC5"
_BUILDING_CAP_FACE = "#26364F"
_BUILDING_ROOF_FACE = "#3A527A"
_BUILDING_WALL_FACE = "#141B29"
_BUILDING_EDGE = "#3F5C8A"
_DESTINATION_COLOR = "#00E676"
_CANDIDATE_ANCHOR_COLOR = "#B0BEC5"
_SELECTED_ANCHOR_COLOR = "#8B0018"
_START_COLOR = "#ECEFF1"
_FAILURE_COLOR = "#FFD600"
_SPIDER_COLOR = "#0B0D10"
_SPIDER_EMBLEM_COLOR = "#8B0018"
_WEB_THREAD_COLOR = "#CFD8DC"

# Each building's render-only depth (see module docstring) is this fraction
# of its own physical width, split evenly on either side of the depth=0
# trajectory plane. A cosmetic proportionality constant, not a modelled or
# measured quantity.
_DEPTH_FRACTION_OF_WIDTH = 0.6


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
_DESTINATION_LABEL_STYLE = dict(
    ha="center",
    va="bottom",
    fontsize=9,
    fontweight="bold",
    color=_DESTINATION_COLOR,
    zorder=2,
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
    `ax` must be a plain 2D axes (see module docstring on layered-axes
    compositing) -- never the 3D data axes.
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
    """Draw a spider silhouette dangling from a silk thread, ringed by an emblem badge, in the top-left corner.

    A literal nod to the web-swinging mechanic this package models, not a
    generic mascot, and not a reproduction of any copyrighted character
    artwork (an abstract arachnid silhouette plus an unfilled ring accent,
    not the mask or logo of any specific character). Positioned in
    axes-fraction coordinates so it stays a fixed corner emblem regardless
    of the data's actual extent. `ax` must be a plain 2D axes (see module
    docstring on layered-axes compositing) -- never the 3D data axes.
    """
    thread_top = (0.07, 1.0)
    thread_bottom = (0.07, 0.762)
    ax.plot(
        [thread_top[0], thread_bottom[0]],
        [thread_top[1], thread_bottom[1]],
        color=_WEB_THREAD_COLOR,
        linewidth=1.4,
        alpha=0.85,
        solid_capstyle="round",
        zorder=50,
        transform=ax.transAxes,
        path_effects=_glow(4.0, _WEB_THREAD_COLOR, alpha=0.4),
    )

    body_x, body_y = thread_bottom
    abdomen_center = (body_x, body_y - 0.0306)
    ax.add_patch(
        Circle(
            (body_x, 0.744),
            radius=0.055,
            facecolor="none",
            edgecolor=_SPIDER_EMBLEM_COLOR,
            linewidth=1.6,
            zorder=49,
            transform=ax.transAxes,
            path_effects=_glow(5.0, _SPIDER_EMBLEM_COLOR, alpha=0.55),
        )
    )
    ax.add_patch(
        Ellipse(
            abdomen_center,
            width=0.0442,
            height=0.034,
            facecolor=_SPIDER_COLOR,
            edgecolor=_WEB_THREAD_COLOR,
            linewidth=0.8,
            zorder=51,
            transform=ax.transAxes,
        )
    )
    ax.add_patch(
        Ellipse(
            (body_x, body_y - 0.0051),
            width=0.0255,
            height=0.0221,
            facecolor=_SPIDER_COLOR,
            edgecolor=_WEB_THREAD_COLOR,
            linewidth=0.8,
            zorder=51,
            transform=ax.transAxes,
        )
    )

    leg_length = 0.0374
    for angle_deg in (200.0, 230.0, 260.0, -20.0, -50.0, -80.0):
        angle = np.radians(angle_deg)
        dx = leg_length * np.cos(angle)
        dy = leg_length * np.sin(angle) * 0.6
        ax.plot(
            [abdomen_center[0], abdomen_center[0] + dx],
            [abdomen_center[1], abdomen_center[1] + dy],
            color=_SPIDER_COLOR,
            linewidth=1.4,
            solid_capstyle="round",
            zorder=50,
            transform=ax.transAxes,
        )


def _build_background_axes(fig: Figure, ax3d: Axes3D) -> Axes:
    """Create a fully transparent 2D axes stacked behind `ax3d` at the same figure position.

    `ax3d`'s own facecolor and axis panes are made transparent by
    `apply_dark_theme` so this background (night sky, stars, spider
    watermark) shows through around and behind the 3D scene, without ever
    being a 3D-projected part of it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure both axes belong to.
    ax3d : mpl_toolkits.mplot3d.axes3d.Axes3D
        The 3D data axes this background sits behind.

    Returns
    -------
    matplotlib.axes.Axes
        The new background axes.
    """
    bg_ax = fig.add_axes(ax3d.get_position(), zorder=ax3d.get_zorder() - 1)
    bg_ax.set_facecolor("none")
    bg_ax.patch.set_alpha(0.0)
    bg_ax.set_xticks([])
    bg_ax.set_yticks([])
    for spine in bg_ax.spines.values():
        spine.set_visible(False)
    return bg_ax


def apply_dark_theme(fig: Figure, ax: Axes3D) -> None:
    """Apply the shared dark "night city" theme to a figure and its 3D data axes.

    Sets the figure's background, builds a transparent background axes
    behind `ax` (`_build_background_axes`) to hold the night-sky gradient,
    star field, and spider watermark (none of which `ax`, a 3D axes, can
    render directly -- see module docstring), makes `ax`'s own facecolor
    and axis panes transparent so that background shows through, and sets
    grid/pane/tick/label/title colors readable against it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to theme.
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        3D data axes to theme.
    """
    fig.patch.set_facecolor(_FIGURE_BG)

    bg_ax = _build_background_axes(fig, ax)
    _paint_night_sky(bg_ax)
    _draw_spider_watermark(bg_ax)

    ax.set_facecolor((0.0, 0.0, 0.0, 0.0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))
        axis.pane.set_edgecolor(_SPINE_COLOR)
    ax.grid(True, linestyle=":", alpha=0.25, color=_GRID_COLOR)
    ax.tick_params(colors=_MUTED_TEXT_COLOR)
    ax.xaxis.label.set_color(_TEXT_COLOR)
    ax.zaxis.label.set_color(_TEXT_COLOR)
    ax.title.set_color(_TEXT_COLOR)

    # The depth axis is a rendering-only construct (module docstring) with
    # no physical meaning, so it is drawn without ticks/labels rather than
    # implying a measurable quantity.
    ax.set_yticks([])
    ax.yaxis.label.set_text("")


def _building_render_depth(building: Building) -> float:
    """Return the total render-only depth used to extrude `building` into a 3D prism.

    See module docstring: this is `_DEPTH_FRACTION_OF_WIDTH` times the
    building's own (physical) `width`, split evenly on either side of the
    depth=0 trajectory plane. Not a modelled or measured building
    property.

    Parameters
    ----------
    building : Building
        Building whose (physical) `width` sets the render depth.

    Returns
    -------
    float
        Total depth, in meters, of the extruded prism (i.e. `2 * half_depth`).
    """
    return _DEPTH_FRACTION_OF_WIDTH * building.width


def _extrude_polygon_faces(vertices_xy: np.ndarray, half_depth: float) -> list[np.ndarray]:
    """Return the face list of a right prism formed by extruding a 2D polygon along the depth axis.

    Given a closed 2D polygon (`vertices_xy`, shape (N, 2), in (x, y)
    order), returns the front cap (at depth = +`half_depth`), the back cap
    (at depth = -`half_depth`), and one quadrilateral side wall per polygon
    edge connecting the corresponding front and back edge -- the standard
    construction of a prism from a 2D cross-section swept along a third
    axis. Each returned face is an (M, 3) array of vertices ordered
    (render_x, render_depth, render_z) = (x, depth, y), matching this
    module's rendering convention (module docstring).

    Parameters
    ----------
    vertices_xy : np.ndarray, shape (N, 2), N >= 3
        Polygon boundary vertices, in (x, y) order, describing one closed
        loop (the last vertex is implicitly connected back to the first).
    half_depth : float
        Half the total extrusion depth. Must be finite and non-negative.

    Returns
    -------
    list[np.ndarray]
        `N + 2` faces: `[front_cap, back_cap, side_wall_0, ..., side_wall_{N-1}]`.
        `side_wall_i` connects polygon edge `(i, i+1 mod N)`.
    """
    verts = np.asarray(vertices_xy, dtype=float)
    n = verts.shape[0]
    front = np.column_stack([verts[:, 0], np.full(n, half_depth), verts[:, 1]])
    back = np.column_stack([verts[:, 0], np.full(n, -half_depth), verts[:, 1]])

    faces = [front, back]
    for i in range(n):
        j = (i + 1) % n
        quad = np.array([front[i], front[j], back[j], back[i]])
        faces.append(quad)
    return faces


def _face_kinds_for_polygon(vertices_xy: np.ndarray, roof_elevation: float, tolerance: float = 1.0e-6) -> list[str]:
    """Classify each face `_extrude_polygon_faces` would produce as `"cap"`, `"roof"`, or `"wall"`.

    The two caps (front/back copies of the full silhouette) are always
    `"cap"`. A side-wall quad is `"roof"` only if both endpoints of its
    source polygon edge sit at `roof_elevation` (within `tolerance`) --
    the single edge tracing the building's flat roofline -- and `"wall"`
    otherwise (this also covers the ground-level edge, which is never
    visible behind the opaque ground slab). Used only to pick a facecolor
    per face for a "glass tower vs. sunlit roof vs. shadowed wall" read;
    it has no bearing on the extruded geometry itself.

    Parameters
    ----------
    vertices_xy : np.ndarray, shape (N, 2), N >= 3
        Same polygon passed to `_extrude_polygon_faces`.
    roof_elevation : float
        The building's `roof_elevation` (its polygon's maximum y).
    tolerance : float, optional
        Absolute tolerance for the roofline comparison.

    Returns
    -------
    list[str]
        `N + 2` entries, in the same order `_extrude_polygon_faces` returns
        its faces: `["cap", "cap", <one per polygon edge>]`.
    """
    verts = np.asarray(vertices_xy, dtype=float)
    n = verts.shape[0]
    kinds = ["cap", "cap"]
    for i in range(n):
        j = (i + 1) % n
        on_roof = abs(verts[i, 1] - roof_elevation) <= tolerance and abs(verts[j, 1] - roof_elevation) <= tolerance
        kinds.append("roof" if on_roof else "wall")
    return kinds


def _flat_quad(x_min: float, x_max: float, z_min: float, z_max: float, depth: float = 0.0) -> np.ndarray:
    """Return a single flat rectangular face lying in the depth=`depth` plane.

    Used for the destination region and the ground slab, both of which are
    genuine (x, y)-plane physics regions (`DestinationRegion`, ground at
    y = 0) and therefore belong exactly on the depth=0 trajectory plane by
    default.
    """
    return np.array(
        [
            [x_min, depth, z_min],
            [x_max, depth, z_min],
            [x_max, depth, z_max],
            [x_min, depth, z_max],
        ]
    )


def plot_city(ax: Axes3D, city: City, show_candidate_anchors: bool = True) -> None:
    """Draw the ground plane, extruded building volumes, and destination region onto `ax`.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        Target 3D data axes.
    city : City
        Environment to draw.
    show_candidate_anchors : bool, optional
        Whether to scatter every candidate anchor in `city`. Defaults to
        True.
    """
    city_scale = max((b.roof_elevation for b in city.buildings), default=5.0)

    all_x = [x for b in city.buildings for x, _y in b.vertices]
    all_x += [city.destination.x_min, city.destination.x_max]
    x_min, x_max = (min(all_x), max(all_x)) if all_x else (-10.0, 10.0)
    x_margin = 0.15 * max(x_max - x_min, 1.0)

    ground = _flat_quad(x_min - x_margin, x_max + x_margin, -0.08 * city_scale, 0.0)
    ax.add_collection3d(Poly3DCollection([ground], facecolor=_GROUND_COLOR, edgecolor="none", zorder=1))
    ax.plot(
        [x_min - x_margin, x_max + x_margin], [0.0, 0.0], [0.0, 0.0], **_GROUND_LINE_STYLE
    )

    facecolor_by_kind = {"cap": _BUILDING_CAP_FACE, "roof": _BUILDING_ROOF_FACE, "wall": _BUILDING_WALL_FACE}
    for building in city.buildings:
        half_depth = _building_render_depth(building) / 2.0
        faces = _extrude_polygon_faces(building.vertices, half_depth)
        kinds = _face_kinds_for_polygon(building.vertices, building.roof_elevation)
        facecolors = [facecolor_by_kind[kind] for kind in kinds]
        ax.add_collection3d(
            Poly3DCollection(
                faces,
                facecolors=facecolors,
                edgecolor=_BUILDING_EDGE,
                linewidths=1.0,
                alpha=0.97,
                zorder=2,
            )
        )

    destination = city.destination
    quad = _flat_quad(destination.x_min, destination.x_max, destination.y_min, destination.y_max)
    ax.add_collection3d(
        Poly3DCollection(
            [quad],
            facecolor=_DESTINATION_COLOR,
            alpha=0.18,
            edgecolor=_DESTINATION_COLOR,
            linewidths=2.2,
            zorder=1,
        )
    )
    label_margin = 0.15 * (destination.y_max - destination.y_min)
    ax.text(
        0.5 * (destination.x_min + destination.x_max),
        0.0,
        destination.y_max + label_margin,
        "DESTINATION",
        **_DESTINATION_LABEL_STYLE,
    )

    if show_candidate_anchors:
        anchors = [point for _building_id, point in city.all_candidate_anchors()]
        if anchors:
            xs, ys = zip(*anchors)
            ax.scatter(xs, [0.0] * len(xs), ys, **_CANDIDATE_ANCHOR_STYLE)


def plot_start(ax: Axes3D, start_position: Point) -> None:
    """Draw a marker at the start location, on the depth=0 trajectory plane.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        Target 3D data axes.
    start_position : tuple[float, float]
        Start position, in meters.
    """
    ax.scatter([start_position[0]], [0.0], [start_position[1]], **_START_STYLE)


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


def plot_trajectory(ax: Axes3D, trajectory: Trajectory) -> None:
    """Draw the full trajectory trace, split into swing and ballistic runs, on the depth=0 plane.

    Each contiguous run of samples sharing a mode is drawn as its own line
    segment (a "swing" run in a solid style, a "ballistic" run dashed, each
    with a glow rendered via `path_effects`), so a multi-hop route
    alternating modes several times renders with the correct style at
    every point along it.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        Target 3D data axes.
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
        depths = np.zeros(len(segment))
        ax.plot(segment[:, 0], depths, segment[:, 1], **style)


def plot_selected_anchors(ax: Axes3D, path: tuple[SearchEdge, ...]) -> None:
    """Draw the anchors actually used by a planned route's edges, on the depth=0 plane.

    Anchor positions are read directly from each edge's `PlannedTransfer`
    (`problem.current_anchor` and, unless the edge ends at
    `planning.astar.GOAL_NODE`, `problem.target_anchor`), not from a
    separate anchor lookup.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        Target 3D data axes.
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
        ax.scatter(xs, [0.0] * len(xs), ys, **_SELECTED_ANCHOR_STYLE)


def plot_constraint_failure_location(
    ax: Axes3D, trajectory: Trajectory, evaluation: TrajectoryEvaluation
) -> bool:
    """Mark the first sample where tension or load-factor margin is negative, if any.

    `evaluation.tension_margins`/`load_factor_margins` are NaN during
    ballistic samples; a NaN comparison is always False, so ballistic
    samples never spuriously trigger this without any special-casing.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        Target 3D data axes.
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
    position = trajectory.positions[index]
    ax.scatter([position[0]], [0.0], [position[1]], **_FAILURE_STYLE)
    return True


def scene_extent(
    city: City, trajectory: Trajectory, start_position: Point, margin_fraction: float = 0.12
) -> tuple[float, float, float, float, float]:
    """Return explicit (x, depth, z) axis limits covering `city`, `trajectory`, and `start_position`.

    `mpl_toolkits.mplot3d`'s autoscaling does not reliably include
    `Poly3DCollection` faces (unlike `plot`/`scatter` data), so callers set
    axis limits explicitly from this rather than relying on it. The depth
    half-extent is derived only from the buildings' own render depth (see
    `_building_render_depth`) since nothing else in the scene has depth
    extent -- the trajectory, anchors, and start marker all sit exactly on
    the depth=0 plane.

    Parameters
    ----------
    city : City
        Supplies building/destination bounds.
    trajectory : Trajectory
        Supplies the trajectory's own (x, y) extent, which can exceed the
        city's building bounds (e.g. an elevated starting anchor).
    start_position : tuple[float, float]
        Included so a start position outside both the city and the
        trajectory bounds (degenerate inputs) is still not clipped.
    margin_fraction : float, optional
        Fraction of each axis's data range added as margin on both ends.

    Returns
    -------
    tuple[float, float, float, float, float]
        `(x_min, x_max, y_half_depth, z_min, z_max)`.
    """
    xs = [x for b in city.buildings for x, _y in b.vertices]
    zs = [y for b in city.buildings for _x, y in b.vertices]
    xs += [city.destination.x_min, city.destination.x_max, start_position[0]]
    zs += [city.destination.y_min, city.destination.y_max, start_position[1], 0.0]
    xs += list(trajectory.positions[:, 0])
    zs += list(trajectory.positions[:, 1])

    x_min, x_max = min(xs), max(xs)
    z_min, z_max = min(zs), max(zs)
    x_margin = margin_fraction * max(x_max - x_min, 1.0)
    z_margin = margin_fraction * max(z_max - z_min, 1.0)
    y_half_depth = max((_building_render_depth(b) / 2.0 for b in city.buildings), default=5.0)
    return x_min - x_margin, x_max + x_margin, y_half_depth, z_min - z_margin, z_max + z_margin


def render_static_overview(
    city: City,
    path: tuple[SearchEdge, ...],
    trajectory: Trajectory,
    start_position: Point,
    figsize: tuple[float, float] = (10.0, 8.0),
) -> Figure:
    """Render a complete static overview figure.

    Combines `plot_city`, `plot_trajectory`, `plot_selected_anchors`, and
    `plot_start` onto one new 3D data axes with a dark theme
    (`apply_dark_theme`) and a legend.

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
    fig = plt.figure(figsize=figsize, dpi=140)
    ax = fig.add_subplot(projection="3d")
    apply_dark_theme(fig, ax)

    plot_city(ax, city)
    plot_trajectory(ax, trajectory)
    plot_selected_anchors(ax, path)
    plot_start(ax, start_position)

    x_min, x_max, y_half_depth, z_min, z_max = scene_extent(city, trajectory, start_position)
    ax.set_xlim3d(x_min, x_max)
    ax.set_ylim3d(-y_half_depth, y_half_depth)
    ax.set_zlim3d(z_min, z_max)
    # Box aspect follows the true (x, z) data proportions (so a tall
    # narrow tower still reads as tall and narrow) but exaggerates the
    # depth axis to a fixed presentation fraction of the x-extent -- the
    # real depth span is often thin, and rendering it at its literal
    # proportion would flatten the scene back toward 2D.
    ax.set_box_aspect((x_max - x_min, 0.5 * (x_max - x_min), z_max - z_min))
    ax.view_init(elev=22.0, azim=-55.0)

    ax.set_title(f"Planned Route Overview — total travel time {trajectory.times[-1]:.2f} s", fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_zlabel("y (m)")
    legend = ax.legend(
        loc="upper right",
        fontsize="small",
        framealpha=0.9,
        facecolor=_AXES_BG,
        edgecolor=_SPINE_COLOR,
        bbox_to_anchor=(1.0, 1.0),
    )
    for text in legend.get_texts():
        text.set_color(_TEXT_COLOR)
    return fig
