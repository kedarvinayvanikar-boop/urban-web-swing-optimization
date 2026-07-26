"""Tests for frame-by-frame trajectory animation and export.

Uses the `Agg` backend and a module-scoped fixture (one real planned route,
same small city as `test_astar.py`/`test_static.py`/`test_hud.py`).
`update_frame` is tested directly (not through `FuncAnimation`'s internal
call mechanics) at a known swing sample and a known ballistic sample, and
`save_animation`'s MP4 branch is checked against
`FFMpegWriter.isAvailable()` rather than assuming any particular
environment has (or lacks) ffmpeg installed; its GIF branch is exercised
as a genuine export (Pillow is a matplotlib dependency, always available)
and checked for a real, non-empty output file.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.animation import FFMpegWriter, FuncAnimation

from webswing.config import BallisticDomain, PhysicalParameters, SwingConstraints
from webswing.geometry.buildings import Building, City, DestinationRegion
from webswing.planning.state import PlanningStateResolution
from webswing.simulation.runner import run_simulation
from webswing.visualization.animation import (
    AnimationArtists,
    _anchor_positions_from_path,
    render_animation,
    save_animation,
    update_frame,
)
from webswing.visualization.hud import build_hud_frame, draw_hud, edge_indices_for_path, format_hud_text

PARAMS = PhysicalParameters(mass=1.0, gravity=9.80665)
CONSTRAINTS = SwingConstraints(
    tension_max=200.0, load_factor_max=20.0, ell_min=1.0, ell_max=30.0, nu_max=5.0
)
DOMAIN = BallisticDomain(x_min=-200.0, x_max=200.0, y_max=200.0, max_duration=10.0)
RESOLUTION = PlanningStateResolution(
    theta_bin_width=0.1, omega_bin_width=0.2, ell_bin_width=0.5, nu_bin_width=0.1
)
START_ANCHOR_ID = "START"
START_ANCHOR_POSITION = (0.0, 50.0)
START_STATE = np.array([-0.3, 1.0, 10.0, 0.0])


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


@pytest.fixture(scope="module")
def successful_run():
    return run_simulation(
        start_anchor_id=START_ANCHOR_ID,
        start_anchor_position=START_ANCHOR_POSITION,
        start_state=START_STATE,
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


# --- _anchor_positions_from_path ------------------------------------------------------


def test_anchor_positions_includes_start(successful_run) -> None:
    positions = _anchor_positions_from_path(
        successful_run.search_result.path, START_ANCHOR_ID, START_ANCHOR_POSITION
    )
    assert positions[START_ANCHOR_ID] == START_ANCHOR_POSITION


# --- update_frame ----------------------------------------------------------------------


def _make_artists():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    fig, ax = plt.subplots()
    (position_marker,) = ax.plot([], [])
    velocity_arrow = FancyArrowPatch((0.0, 0.0), (0.0, 0.0))
    ax.add_patch(velocity_arrow)
    (web_line,) = ax.plot([], [])
    return fig, ax, position_marker, velocity_arrow, web_line


def test_update_frame_swing_sample_shows_web_line_to_active_anchor(successful_run) -> None:
    import matplotlib.pyplot as plt

    path = successful_run.search_result.path
    edge_indices = edge_indices_for_path(path)
    anchor_positions = _anchor_positions_from_path(path, START_ANCHOR_ID, START_ANCHOR_POSITION)

    fig, ax, position_marker, velocity_arrow, web_line = _make_artists()
    hud_text = draw_hud(
        ax, build_hud_frame(successful_run.trajectory, successful_run.evaluation, edge_indices, CONSTRAINTS, 0)
    )
    artists = AnimationArtists(position_marker, velocity_arrow, web_line, hud_text)

    update_frame(artists, successful_run.trajectory, successful_run.evaluation, edge_indices, anchor_positions, CONSTRAINTS, 0)

    assert web_line.get_visible() is True
    xs, ys = web_line.get_data()
    assert (xs[0], ys[0]) == START_ANCHOR_POSITION
    expected_position = tuple(successful_run.trajectory.positions[0])
    assert (xs[1], ys[1]) == pytest.approx(expected_position)

    pm_xs, pm_ys = position_marker.get_data()
    assert (pm_xs[0], pm_ys[0]) == pytest.approx(expected_position)
    plt.close(fig)


def test_update_frame_ballistic_sample_hides_web_line(successful_run) -> None:
    import matplotlib.pyplot as plt

    path = successful_run.search_result.path
    edge_indices = edge_indices_for_path(path)
    anchor_positions = _anchor_positions_from_path(path, START_ANCHOR_ID, START_ANCHOR_POSITION)
    ballistic_index = successful_run.trajectory.modes.index("ballistic")

    fig, ax, position_marker, velocity_arrow, web_line = _make_artists()
    hud_text = draw_hud(
        ax,
        build_hud_frame(
            successful_run.trajectory, successful_run.evaluation, edge_indices, CONSTRAINTS, ballistic_index
        ),
    )
    artists = AnimationArtists(position_marker, velocity_arrow, web_line, hud_text)

    update_frame(
        artists,
        successful_run.trajectory,
        successful_run.evaluation,
        edge_indices,
        anchor_positions,
        CONSTRAINTS,
        ballistic_index,
    )

    assert web_line.get_visible() is False
    plt.close(fig)


def test_update_frame_updates_hud_text_to_match_frame(successful_run) -> None:
    import matplotlib.pyplot as plt

    path = successful_run.search_result.path
    edge_indices = edge_indices_for_path(path)
    anchor_positions = _anchor_positions_from_path(path, START_ANCHOR_ID, START_ANCHOR_POSITION)

    fig, ax, position_marker, velocity_arrow, web_line = _make_artists()
    frame0 = build_hud_frame(successful_run.trajectory, successful_run.evaluation, edge_indices, CONSTRAINTS, 0)
    hud_text = draw_hud(ax, frame0)
    artists = AnimationArtists(position_marker, velocity_arrow, web_line, hud_text)

    last = len(successful_run.trajectory.times) - 1
    update_frame(artists, successful_run.trajectory, successful_run.evaluation, edge_indices, anchor_positions, CONSTRAINTS, last)

    expected_frame = build_hud_frame(
        successful_run.trajectory, successful_run.evaluation, edge_indices, CONSTRAINTS, last
    )
    assert hud_text.get_text() == format_hud_text(expected_frame)
    plt.close(fig)


# --- render_animation --------------------------------------------------------------------


def test_render_animation_produces_expected_artist_counts(successful_run) -> None:
    city = make_city()
    fig, anim = render_animation(
        city,
        successful_run.search_result.path,
        successful_run.trajectory,
        successful_run.evaluation,
        CONSTRAINTS,
        START_ANCHOR_ID,
        START_ANCHOR_POSITION,
    )
    ax = fig.axes[0]

    assert isinstance(anim, FuncAnimation)
    # 1 building + 1 destination rectangle + 1 velocity-arrow patch
    assert len(ax.patches) == len(city.buildings) + 2
    # ground + swing run + ballistic run + position marker + web line
    assert len(ax.lines) == 5
    # candidate anchors, selected anchors, start (no constraint-failure marker: feasible route)
    assert len(ax.collections) == 3
    assert len(ax.texts) == 1

    import matplotlib.pyplot as plt

    plt.close(fig)


# --- save_animation --------------------------------------------------------------------------


def test_save_animation_rejects_unsupported_extension(successful_run, tmp_path) -> None:
    fig, anim = render_animation(
        make_city(),
        successful_run.search_result.path,
        successful_run.trajectory,
        successful_run.evaluation,
        CONSTRAINTS,
        START_ANCHOR_ID,
        START_ANCHOR_POSITION,
    )
    with pytest.raises(ValueError):
        save_animation(anim, str(tmp_path / "out.avi"))

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_save_animation_gif_produces_a_real_nonempty_file(successful_run, tmp_path) -> None:
    fig, anim = render_animation(
        make_city(),
        successful_run.search_result.path,
        successful_run.trajectory,
        successful_run.evaluation,
        CONSTRAINTS,
        START_ANCHOR_ID,
        START_ANCHOR_POSITION,
    )
    output_path = tmp_path / "out.gif"
    ok = save_animation(anim, str(output_path), fps=10)

    assert ok is True
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_save_animation_mp4_matches_ffmpeg_availability(successful_run, tmp_path) -> None:
    fig, anim = render_animation(
        make_city(),
        successful_run.search_result.path,
        successful_run.trajectory,
        successful_run.evaluation,
        CONSTRAINTS,
        START_ANCHOR_ID,
        START_ANCHOR_POSITION,
    )
    output_path = tmp_path / "out.mp4"
    ok = save_animation(anim, str(output_path), fps=10)

    assert ok == FFMpegWriter.isAvailable()

    import matplotlib.pyplot as plt

    plt.close(fig)
