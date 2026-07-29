r"""Frame-by-frame playback of a stored trajectory via `matplotlib.animation`.

Combines `static.py`'s city/route rendering (drawn once, unchanging across
frames) with per-frame elements CLAUDE.md lists that a single static image
cannot show: the active web line (the anchor-to-body segment during a
swing sample, absent during ballistic samples), the current position
marker, the current velocity vector, and `hud.py`'s per-frame HUD text.

Per CLAUDE.md, this module must not rerun the optimizer or numerical
integrator while rendering: `render_animation` consumes an already-
assembled `Trajectory`/`TrajectoryEvaluation`/`SearchEdge` path exactly as
produced by `simulation.runner.run_simulation`, and every per-frame update
only reads already-computed arrays.

`update_frame` is a standalone, independently callable function (not a
closure captured inside `render_animation`), so a single frame's behavior
can be tested directly without depending on `matplotlib.animation.
FuncAnimation`'s internal call mechanics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Text

from webswing.config import SwingConstraints
from webswing.geometry.buildings import City
from webswing.planning.astar import GOAL_NODE, PlannedTransfer, SearchEdge
from webswing.simulation.evaluator import TrajectoryEvaluation
from webswing.simulation.trajectory import Trajectory
from webswing.visualization.hud import build_hud_frame, draw_hud, edge_indices_for_path, format_hud_text
from webswing.visualization.static import (
    apply_dark_theme,
    plot_city,
    plot_constraint_failure_location,
    plot_selected_anchors,
    plot_start,
    plot_trajectory,
)

Point = tuple[float, float]

_VELOCITY_DISPLAY_SECONDS = 0.15
"""Velocity arrows are drawn as position -> position + velocity * this many
seconds: a display heuristic (arrow length = distance travelled in that
time), not a physical quantity."""

_POSITION_COLOR = "#ECEFF1"
_POSITION_EDGE_COLOR = "#FF1744"
_VELOCITY_COLOR = "#FFC400"
_WEB_LINE_COLOR = "#CFD8DC"
_LEGEND_BG = "#05070D"
_LEGEND_EDGE = "#2A3F5A"
_LEGEND_TEXT = "#ECEFF1"


def _glow(linewidth: float, color: str, alpha: float = 0.6) -> list:
    """Return a `path_effects` list giving an artist a soft glow halo, plus normal rendering on top."""
    return [path_effects.withStroke(linewidth=linewidth, foreground=color, alpha=alpha), path_effects.Normal()]


_WRITERS: dict[str, tuple[type, str]] = {
    ".mp4": (FFMpegWriter, "ffmpeg"),
    ".gif": (PillowWriter, "Pillow"),
}


def _anchor_positions_from_path(
    path: tuple[SearchEdge, ...], start_anchor_id: str, start_anchor_position: Point
) -> dict[str, Point]:
    positions: dict[str, Point] = {start_anchor_id: start_anchor_position}
    for edge in path:
        if not isinstance(edge.label, PlannedTransfer):
            raise TypeError(
                "render_animation requires a path whose edge labels are "
                f"PlannedTransfer (as produced by planning.astar.plan_route); "
                f"got {type(edge.label)!r}"
            )
        positions[edge.from_node.anchor_id] = edge.label.problem.current_anchor
        if edge.to_node != GOAL_NODE:
            positions[edge.to_node.anchor_id] = edge.label.problem.target_anchor
    return positions


@dataclass(frozen=True)
class AnimationArtists:
    """The per-frame-mutable matplotlib artists `update_frame` refreshes each frame.

    Parameters
    ----------
    position_marker : matplotlib.lines.Line2D
        Marker at the current position.
    velocity_arrow : matplotlib.patches.FancyArrowPatch
        Arrow from the current position along the current velocity.
    web_line : matplotlib.lines.Line2D
        Segment from the active anchor to the current position; hidden
        during ballistic samples.
    hud_text : matplotlib.text.Text
        HUD text block.
    """

    position_marker: Line2D
    velocity_arrow: FancyArrowPatch
    web_line: Line2D
    hud_text: Text


def update_frame(
    artists: AnimationArtists,
    trajectory: Trajectory,
    evaluation: TrajectoryEvaluation,
    edge_indices: tuple[int | None, ...],
    anchor_positions: dict[str, Point],
    constraints: SwingConstraints,
    frame_index: int,
    velocity_display_seconds: float = _VELOCITY_DISPLAY_SECONDS,
) -> tuple:
    """Update `artists` in place for one frame and return them (blit-compatible).

    Parameters
    ----------
    artists : AnimationArtists
        Artists to mutate.
    trajectory : Trajectory
        Assembled trajectory.
    evaluation : TrajectoryEvaluation
        Force-derived quantities, index-aligned with `trajectory`.
    edge_indices : tuple[int or None, ...]
        From `hud.edge_indices_for_path`, for the same path.
    anchor_positions : dict[str, tuple[float, float]]
        Anchor identifier to position, covering every anchor visited (see
        `render_animation`).
    constraints : SwingConstraints
        Supplies `tension_max` for the HUD.
    frame_index : int
        Sample index to render.
    velocity_display_seconds : float, optional
        Display-only velocity arrow scale (see module docstring).

    Returns
    -------
    tuple
        `(position_marker, velocity_arrow, web_line, hud_text)`, as
        `matplotlib.animation.FuncAnimation`'s blitting contract expects.
    """
    frame = build_hud_frame(trajectory, evaluation, edge_indices, constraints, frame_index)
    x, y = frame.position

    artists.position_marker.set_data([x], [y])

    vx, vy = trajectory.velocities[frame_index]
    artists.velocity_arrow.set_positions(
        (x, y), (x + vx * velocity_display_seconds, y + vy * velocity_display_seconds)
    )

    if frame.mode == "swing" and frame.active_anchor_id is not None:
        anchor_position = anchor_positions[frame.active_anchor_id]
        artists.web_line.set_data([anchor_position[0], x], [anchor_position[1], y])
        artists.web_line.set_visible(True)
    else:
        artists.web_line.set_data([], [])
        artists.web_line.set_visible(False)

    artists.hud_text.set_text(format_hud_text(frame))

    return artists.position_marker, artists.velocity_arrow, artists.web_line, artists.hud_text


def render_animation(
    city: City,
    path: tuple[SearchEdge, ...],
    trajectory: Trajectory,
    evaluation: TrajectoryEvaluation,
    constraints: SwingConstraints,
    start_anchor_id: str,
    start_anchor_position: Point,
    interval_ms: float = 50.0,
    velocity_display_seconds: float = _VELOCITY_DISPLAY_SECONDS,
    figsize: tuple[float, float] = (10.0, 8.0),
) -> tuple[Figure, FuncAnimation]:
    """Build a `FuncAnimation` replaying an assembled trajectory over a static city/route backdrop.

    Parameters
    ----------
    city : City
        Environment to draw.
    path : tuple[SearchEdge, ...]
        `SearchResult.path` from `planning.astar.plan_route`.
    trajectory : Trajectory
        Assembled trajectory to replay.
    evaluation : TrajectoryEvaluation
        Force-derived quantities, index-aligned with `trajectory`.
    constraints : SwingConstraints
        Supplies `tension_max` for the HUD and the thresholds for the
        constraint-failure marker.
    start_anchor_id : str
        Identifier of the anchor `trajectory` starts attached to.
    start_anchor_position : tuple[float, float]
        Position of that anchor, in meters.
    interval_ms : float, optional
        Delay between frames, in milliseconds (playback speed; not a
        physical quantity).
    velocity_display_seconds : float, optional
        Forwarded to `update_frame`.
    figsize : tuple[float, float], optional
        Figure size in inches.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.animation.FuncAnimation]
        The figure and the animation object. Use `save_animation` for
        deterministic export, or display `fig` interactively.
    """
    edge_indices = edge_indices_for_path(path)
    anchor_positions = _anchor_positions_from_path(path, start_anchor_id, start_anchor_position)

    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    apply_dark_theme(fig, ax)
    plot_city(ax, city)
    plot_trajectory(ax, trajectory)
    plot_selected_anchors(ax, path)
    plot_start(ax, start_anchor_position)
    plot_constraint_failure_location(ax, trajectory, evaluation)

    (position_marker,) = ax.plot(
        [],
        [],
        marker="o",
        markersize=12,
        markerfacecolor=_POSITION_COLOR,
        markeredgecolor=_POSITION_EDGE_COLOR,
        markeredgewidth=1.4,
        linestyle="none",
        zorder=7,
        label="current position",
        path_effects=_glow(9.0, _POSITION_COLOR, alpha=0.65),
    )
    velocity_arrow = FancyArrowPatch(
        (0.0, 0.0),
        (0.0, 0.0),
        color=_VELOCITY_COLOR,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=2.2,
        zorder=6,
        path_effects=_glow(7.0, _VELOCITY_COLOR, alpha=0.55),
    )
    ax.add_patch(velocity_arrow)
    (web_line,) = ax.plot(
        [],
        [],
        color=_WEB_LINE_COLOR,
        linewidth=2.0,
        linestyle=":",
        zorder=5,
        label="active web",
        path_effects=_glow(6.0, _WEB_LINE_COLOR, alpha=0.5),
    )

    initial_frame = build_hud_frame(trajectory, evaluation, edge_indices, constraints, 0)
    hud_text = draw_hud(ax, initial_frame)

    artists = AnimationArtists(position_marker, velocity_arrow, web_line, hud_text)

    ax.set_title("Trajectory Playback", fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    legend = ax.legend(loc="lower right", fontsize="small", framealpha=0.9, facecolor=_LEGEND_BG, edgecolor=_LEGEND_EDGE)
    for text in legend.get_texts():
        text.set_color(_LEGEND_TEXT)

    def _update(frame_index: int) -> tuple:
        return update_frame(
            artists,
            trajectory,
            evaluation,
            edge_indices,
            anchor_positions,
            constraints,
            frame_index,
            velocity_display_seconds,
        )

    anim = FuncAnimation(
        fig, _update, frames=len(trajectory.times), interval=interval_ms, blit=False
    )
    return fig, anim


def save_animation(anim: FuncAnimation, filepath: str, fps: int = 20) -> bool:
    """Export `anim` to `filepath` using the writer matching its file extension.

    Dispatch is by extension (deterministic; no trial-and-error across
    writers): `.mp4` uses `FFMpegWriter`, `.gif` uses `PillowWriter`. If the
    corresponding encoder is not installed locally, this returns False
    rather than raising -- CLAUDE.md's "when the necessary local encoder is
    available" language anticipates environments without one.

    Parameters
    ----------
    anim : matplotlib.animation.FuncAnimation
        Animation to export, from `render_animation`.
    filepath : str
        Output path; its extension selects the writer.
    fps : int, optional
        Playback frame rate for the exported file.

    Returns
    -------
    bool
        True if the export succeeded, False if the required encoder is not
        available locally.

    Raises
    ------
    ValueError
        If `filepath`'s extension is not one of the supported formats.
    """
    extension = os.path.splitext(filepath)[1].lower()
    if extension not in _WRITERS:
        raise ValueError(
            f"unsupported animation file extension {extension!r}; supported: {sorted(_WRITERS)}"
        )
    writer_cls, _encoder_name = _WRITERS[extension]
    if not writer_cls.isAvailable():
        return False
    anim.save(filepath, writer=writer_cls(fps=fps))
    return True
