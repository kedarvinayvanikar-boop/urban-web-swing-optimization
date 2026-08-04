"""Tests for static overview plotting.

Uses the `Agg` (non-interactive) backend, set before importing pyplot
anywhere, since this environment has no display. A real planned route
(same small city as `test_astar.py`) is computed once via a module-scoped
fixture and reused across tests, since each `run_simulation` call performs
real `solve_ivp`/`scipy.optimize` work.

Checks focus on structural correctness of what gets added to the 3D data
axes (`Poly3DCollection`/`Line3D`/`Path3DCollection` counts, and specific
3D coordinates via `get_data_3d`/`_offsets3d`) rather than pixel-level
rendering, which is the standard way to test `matplotlib` code without a
display or golden-image comparisons. Every plotting function under test
takes an `Axes3D` (built via `fig.add_subplot(projection="3d")`), per this
module's rendering-only depth-axis convention (see `static.py`'s module
docstring): the physics stays two-dimensional, and every plotted element
here sits on the depth=0 trajectory plane unless it is an extruded
building.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from dataclasses import replace

import numpy as np
import pytest
from mpl_toolkits.mplot3d.art3d import Path3DCollection, Poly3DCollection

from webswing.config import BallisticDomain, PhysicalParameters, SwingConstraints
from webswing.geometry.buildings import Building, City, DestinationRegion
from webswing.planning.state import PlanningStateResolution
from webswing.simulation.runner import run_simulation
from webswing.simulation.evaluator import TrajectoryEvaluation
from webswing.simulation.trajectory import Trajectory
from webswing.visualization.static import (
    _contiguous_mode_runs,
    _extrude_polygon_faces,
    _face_kinds_for_polygon,
    plot_city,
    plot_constraint_failure_location,
    plot_selected_anchors,
    plot_start,
    plot_trajectory,
    render_static_overview,
    scene_extent,
)

PARAMS = PhysicalParameters(mass=1.0, gravity=9.80665)
CONSTRAINTS = SwingConstraints(
    tension_max=200.0, load_factor_max=20.0, ell_min=1.0, ell_max=30.0, nu_max=5.0
)
DOMAIN = BallisticDomain(x_min=-200.0, x_max=200.0, y_max=200.0, max_duration=10.0)
RESOLUTION = PlanningStateResolution(
    theta_bin_width=0.1, omega_bin_width=0.2, ell_bin_width=0.5, nu_bin_width=0.1
)
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


def make_3d_axes():
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    return fig, ax


@pytest.fixture(scope="module")
def successful_run():
    city = make_city()
    return run_simulation(
        start_anchor_id="START",
        start_anchor_position=START_ANCHOR_POSITION,
        start_state=START_STATE,
        city=city,
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
    ), city


# --- _contiguous_mode_runs -----------------------------------------------------------


def test_contiguous_mode_runs_single_mode() -> None:
    assert _contiguous_mode_runs(("swing", "swing", "swing")) == [("swing", 0, 2)]


def test_contiguous_mode_runs_alternating() -> None:
    modes = ("swing", "swing", "ballistic", "ballistic", "ballistic", "swing")
    assert _contiguous_mode_runs(modes) == [("swing", 0, 1), ("ballistic", 2, 4), ("swing", 5, 5)]


def test_contiguous_mode_runs_empty() -> None:
    assert _contiguous_mode_runs(()) == []


def test_contiguous_mode_runs_single_sample() -> None:
    assert _contiguous_mode_runs(("swing",)) == [("swing", 0, 0)]


# --- _extrude_polygon_faces --------------------------------------------------------------


def test_extrude_polygon_faces_unit_square_front_and_back_caps() -> None:
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    faces = _extrude_polygon_faces(square, half_depth=2.0)
    assert len(faces) == 6  # 2 caps + 4 side walls, one per edge

    front_cap, back_cap = faces[0], faces[1]
    np.testing.assert_allclose(front_cap[:, 1], 2.0)  # render_depth = +half_depth
    np.testing.assert_allclose(back_cap[:, 1], -2.0)
    np.testing.assert_allclose(front_cap[:, [0, 2]], square)  # (render_x, render_z) = (x, y)
    np.testing.assert_allclose(back_cap[:, [0, 2]], square)


def test_extrude_polygon_faces_side_wall_connects_matching_front_back_edge() -> None:
    square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    faces = _extrude_polygon_faces(square, half_depth=0.5)
    first_wall = faces[2]  # edge (vertex 0 -> vertex 1)
    expected = np.array(
        [
            [0.0, 0.5, 0.0],
            [1.0, 0.5, 0.0],
            [1.0, -0.5, 0.0],
            [0.0, -0.5, 0.0],
        ]
    )
    np.testing.assert_allclose(first_wall, expected)


def test_extrude_polygon_faces_zero_depth_collapses_caps_onto_plane() -> None:
    triangle = np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]])
    faces = _extrude_polygon_faces(triangle, half_depth=0.0)
    assert len(faces) == 5  # 2 caps + 3 side walls
    for face in faces:
        np.testing.assert_allclose(face[:, 1], 0.0)


# --- _face_kinds_for_polygon -------------------------------------------------------------


def test_face_kinds_for_polygon_rectangle_has_one_roof_edge() -> None:
    verts = np.array([[20.0, 0.0], [24.0, 0.0], [24.0, 27.0], [20.0, 27.0]])
    kinds = _face_kinds_for_polygon(verts, roof_elevation=27.0)
    assert kinds[:2] == ["cap", "cap"]
    assert kinds.count("roof") == 1  # only the top edge (24,27)->(20,27)
    assert kinds[2:].count("wall") == 3  # bottom + two vertical sides


def test_face_kinds_for_polygon_ground_edge_is_not_roof() -> None:
    verts = np.array([[20.0, 0.0], [24.0, 0.0], [24.0, 27.0], [20.0, 27.0]])
    kinds = _face_kinds_for_polygon(verts, roof_elevation=27.0)
    # edge 0 connects (20,0)->(24,0): both at y=0, not roof_elevation
    assert kinds[2] == "wall"


# --- scene_extent -------------------------------------------------------------------------


def test_scene_extent_includes_trajectory_above_city_bounds(successful_run) -> None:
    run, city = successful_run
    x_min, x_max, y_half_depth, z_min, z_max = scene_extent(city, run.trajectory, START_ANCHOR_POSITION)
    # the start anchor (y=50) sits well above the building's roof (y=27);
    # scene_extent must not clip it to the city's own bounds.
    assert z_max > 27.0
    assert x_min <= 0.0 <= x_max
    assert y_half_depth > 0.0


def test_scene_extent_covers_start_position_outside_other_bounds() -> None:
    city = make_city()
    trajectory = _make_trajectory(2)
    x_min, x_max, y_half_depth, z_min, z_max = scene_extent(city, trajectory, (1000.0, 2000.0))
    assert x_max >= 1000.0
    assert z_max >= 2000.0
    assert y_half_depth > 0.0


# --- plot_city -------------------------------------------------------------------------


def test_plot_city_adds_one_poly3dcollection_per_building_plus_destination_and_ground() -> None:
    city = make_city()
    fig, ax = make_3d_axes()
    plot_city(ax, city)
    poly_collections = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
    assert len(poly_collections) == len(city.buildings) + 2
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_city_adds_ground_line() -> None:
    city = make_city()
    fig, ax = make_3d_axes()
    plot_city(ax, city)
    assert len(ax.lines) == 1  # the ground line
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_city_scatters_all_candidate_anchors_on_depth_zero_plane() -> None:
    city = make_city()
    fig, ax = make_3d_axes()
    plot_city(ax, city)
    scatters = [c for c in ax.collections if isinstance(c, Path3DCollection)]
    assert len(scatters) == 1
    xs, depths, zs = scatters[0]._offsets3d
    assert len(xs) == len(city.all_candidate_anchors())
    np.testing.assert_allclose(depths, 0.0)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_city_can_hide_candidate_anchors() -> None:
    city = make_city()
    fig, ax = make_3d_axes()
    plot_city(ax, city, show_candidate_anchors=False)
    scatters = [c for c in ax.collections if isinstance(c, Path3DCollection)]
    assert len(scatters) == 0
    import matplotlib.pyplot as plt

    plt.close(fig)


# --- plot_start --------------------------------------------------------------------------


def test_plot_start_adds_one_point_at_given_position_depth_zero() -> None:
    fig, ax = make_3d_axes()
    plot_start(ax, (5.0, 10.0))
    scatters = [c for c in ax.collections if isinstance(c, Path3DCollection)]
    assert len(scatters) == 1
    xs, depths, zs = scatters[0]._offsets3d
    assert len(xs) == 1
    np.testing.assert_allclose([xs[0], depths[0], zs[0]], [5.0, 0.0, 10.0])
    import matplotlib.pyplot as plt

    plt.close(fig)


# --- plot_trajectory -----------------------------------------------------------------------


def test_plot_trajectory_draws_one_line_per_contiguous_run(successful_run) -> None:
    run, _city = successful_run
    fig, ax = make_3d_axes()
    plot_trajectory(ax, run.trajectory)
    expected_n_lines = len(_contiguous_mode_runs(run.trajectory.modes))
    assert len(ax.lines) == expected_n_lines
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_trajectory_first_segment_matches_swing_positions_on_depth_zero(successful_run) -> None:
    run, _city = successful_run
    fig, ax = make_3d_axes()
    plot_trajectory(ax, run.trajectory)

    first_run_mode, start, end = _contiguous_mode_runs(run.trajectory.modes)[0]
    assert first_run_mode == "swing"
    expected_x = run.trajectory.positions[start : end + 1, 0]
    expected_z = run.trajectory.positions[start : end + 1, 1]
    xs, depths, zs = ax.lines[0].get_data_3d()
    np.testing.assert_allclose(xs, expected_x)
    np.testing.assert_allclose(zs, expected_z)
    np.testing.assert_allclose(depths, 0.0)
    import matplotlib.pyplot as plt

    plt.close(fig)


# --- plot_selected_anchors -----------------------------------------------------------------


def test_plot_selected_anchors_matches_edge_anchor_positions(successful_run) -> None:
    run, _city = successful_run
    fig, ax = make_3d_axes()
    plot_selected_anchors(ax, run.search_result.path)

    expected = set()
    for edge in run.search_result.path:
        expected.add(edge.label.problem.current_anchor)
        from webswing.planning.astar import GOAL_NODE

        if edge.to_node != GOAL_NODE:
            expected.add(edge.label.problem.target_anchor)

    scatters = [c for c in ax.collections if isinstance(c, Path3DCollection)]
    xs, depths, zs = scatters[0]._offsets3d
    plotted = {(float(x), float(z)) for x, z in zip(xs, zs)}
    assert plotted == expected
    np.testing.assert_allclose(depths, 0.0)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_selected_anchors_rejects_non_planned_transfer_label(successful_run) -> None:
    run, _city = successful_run
    fig, ax = make_3d_axes()
    bad_path = (replace(run.search_result.path[0], label="not-a-planned-transfer"),)
    with pytest.raises(TypeError):
        plot_selected_anchors(ax, bad_path)
    import matplotlib.pyplot as plt

    plt.close(fig)


# --- render_static_overview ------------------------------------------------------------------


def test_render_static_overview_produces_expected_artist_counts(successful_run) -> None:
    run, city = successful_run
    fig = render_static_overview(city, run.search_result.path, run.trajectory, START_ANCHOR_POSITION)
    ax = fig.axes[0]

    poly_collections = [c for c in ax.collections if isinstance(c, Poly3DCollection)]
    scatter_collections = [c for c in ax.collections if isinstance(c, Path3DCollection)]
    # buildings + destination + ground
    assert len(poly_collections) == len(city.buildings) + 2
    # candidate anchors, selected anchors, start
    assert len(scatter_collections) == 3
    # ground line + trajectory runs
    assert len(ax.lines) == 1 + len(_contiguous_mode_runs(run.trajectory.modes))
    assert len(ax.texts) == 1  # destination label
    assert len(ax.patches) == 0  # everything is a 3D collection now, not a 2D patch

    legend = ax.get_legend()
    assert legend is not None
    labels = {t.get_text() for t in legend.get_texts()}
    assert {"candidate anchor", "swing", "ballistic", "selected anchor", "start"} <= labels

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_render_static_overview_background_axes_carries_theme_decoration(successful_run) -> None:
    run, city = successful_run
    fig = render_static_overview(city, run.search_result.path, run.trajectory, START_ANCHOR_POSITION)
    assert len(fig.axes) == 2
    bg_ax = fig.axes[1]
    assert len(bg_ax.collections) == 1  # star field
    assert len(bg_ax.patches) == 3  # spider emblem ring + abdomen + head
    assert len(bg_ax.lines) == 7  # thread + 6 legs

    import matplotlib.pyplot as plt

    plt.close(fig)


# --- plot_constraint_failure_location -------------------------------------------------------


def _make_trajectory(n: int) -> Trajectory:
    return Trajectory(
        times=np.arange(float(n)),
        positions=np.array([[float(i), float(i)] for i in range(n)]),
        velocities=np.zeros((n, 2)),
        modes=("swing",) * n,
        active_anchor_ids=("A",) * n,
        web_lengths=np.full(n, 2.0),
        angular_rates=np.zeros(n),
        radial_rates=np.zeros(n),
    )


def test_plot_constraint_failure_location_returns_false_when_feasible(successful_run) -> None:
    run, _city = successful_run
    fig, ax = make_3d_axes()
    found = plot_constraint_failure_location(ax, run.trajectory, run.evaluation)
    assert found is False
    scatters = [c for c in ax.collections if isinstance(c, Path3DCollection)]
    assert len(scatters) == 0
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_constraint_failure_location_marks_first_violation() -> None:
    trajectory = _make_trajectory(4)
    evaluation = TrajectoryEvaluation(
        tensions=np.array([10.0, 10.0, 10.0, 10.0]),
        load_factors=np.array([1.0, 1.0, 1.0, 1.0]),
        tension_margins=np.array([5.0, -1.0, -2.0, 5.0]),
        load_factor_margins=np.array([1.0, 1.0, 1.0, 1.0]),
    )
    fig, ax = make_3d_axes()
    found = plot_constraint_failure_location(ax, trajectory, evaluation)
    assert found is True
    scatters = [c for c in ax.collections if isinstance(c, Path3DCollection)]
    assert len(scatters) == 1
    xs, depths, zs = scatters[0]._offsets3d
    np.testing.assert_allclose([xs[0], zs[0]], trajectory.positions[1])  # first violation is index 1
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_plot_constraint_failure_location_ignores_nan_ballistic_margins() -> None:
    trajectory = _make_trajectory(3)
    evaluation = TrajectoryEvaluation(
        tensions=np.array([10.0, float("nan"), 10.0]),
        load_factors=np.array([1.0, float("nan"), 1.0]),
        tension_margins=np.array([5.0, float("nan"), 5.0]),
        load_factor_margins=np.array([1.0, float("nan"), 1.0]),
    )
    fig, ax = make_3d_axes()
    found = plot_constraint_failure_location(ax, trajectory, evaluation)
    assert found is False
    import matplotlib.pyplot as plt

    plt.close(fig)
