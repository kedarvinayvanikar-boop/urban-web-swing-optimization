"""Tests for static overview plotting.

Uses the `Agg` (non-interactive) backend, set before importing pyplot
anywhere, since this environment has no display. A real planned route
(same small city as `test_astar.py`) is computed once via a module-scoped
fixture and reused across tests, since each `run_simulation` call performs
real `solve_ivp`/`scipy.optimize` work.

Checks focus on structural correctness of what gets added to the Axes
(patch/line/collection counts, and specific line/point coordinates) rather
than pixel-level rendering, which is the standard way to test `matplotlib`
code without a display or golden-image comparisons.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from dataclasses import replace

import numpy as np
import pytest

from webswing.config import BallisticDomain, PhysicalParameters, SwingConstraints
from webswing.geometry.buildings import Building, City, DestinationRegion
from webswing.planning.state import PlanningStateResolution
from webswing.simulation.runner import run_simulation
from webswing.visualization.static import (
    _contiguous_mode_runs,
    plot_city,
    plot_selected_anchors,
    plot_start,
    plot_trajectory,
    render_static_overview,
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


# --- plot_city -------------------------------------------------------------------------


def test_plot_city_adds_building_and_destination_patches() -> None:
    import matplotlib.pyplot as plt

    city = make_city()
    fig, ax = plt.subplots()
    plot_city(ax, city)
    # one polygon per building, plus one rectangle for the destination region
    assert len(ax.patches) == len(city.buildings) + 1
    plt.close(fig)


def test_plot_city_adds_ground_line() -> None:
    import matplotlib.pyplot as plt

    city = make_city()
    fig, ax = plt.subplots()
    plot_city(ax, city)
    assert len(ax.lines) == 1  # the ground axhline
    plt.close(fig)


def test_plot_city_scatters_all_candidate_anchors() -> None:
    import matplotlib.pyplot as plt

    city = make_city()
    fig, ax = plt.subplots()
    plot_city(ax, city)
    assert len(ax.collections) == 1
    offsets = ax.collections[0].get_offsets()
    assert len(offsets) == len(city.all_candidate_anchors())
    plt.close(fig)


def test_plot_city_can_hide_candidate_anchors() -> None:
    import matplotlib.pyplot as plt

    city = make_city()
    fig, ax = plt.subplots()
    plot_city(ax, city, show_candidate_anchors=False)
    assert len(ax.collections) == 0
    plt.close(fig)


# --- plot_start --------------------------------------------------------------------------


def test_plot_start_adds_one_point_at_given_position() -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plot_start(ax, (5.0, 10.0))
    assert len(ax.collections) == 1
    offsets = ax.collections[0].get_offsets()
    assert len(offsets) == 1
    np.testing.assert_allclose(offsets[0], [5.0, 10.0])
    plt.close(fig)


# --- plot_trajectory -----------------------------------------------------------------------


def test_plot_trajectory_draws_one_line_per_contiguous_run(successful_run) -> None:
    import matplotlib.pyplot as plt

    run, _city = successful_run
    fig, ax = plt.subplots()
    plot_trajectory(ax, run.trajectory)
    expected_n_lines = len(_contiguous_mode_runs(run.trajectory.modes))
    assert len(ax.lines) == expected_n_lines
    plt.close(fig)


def test_plot_trajectory_first_segment_matches_swing_positions(successful_run) -> None:
    import matplotlib.pyplot as plt

    run, _city = successful_run
    fig, ax = plt.subplots()
    plot_trajectory(ax, run.trajectory)

    first_run_mode, start, end = _contiguous_mode_runs(run.trajectory.modes)[0]
    assert first_run_mode == "swing"
    expected_x = run.trajectory.positions[start : end + 1, 0]
    np.testing.assert_allclose(ax.lines[0].get_xdata(), expected_x)
    plt.close(fig)


# --- plot_selected_anchors -----------------------------------------------------------------


def test_plot_selected_anchors_matches_edge_anchor_positions(successful_run) -> None:
    import matplotlib.pyplot as plt

    run, _city = successful_run
    fig, ax = plt.subplots()
    plot_selected_anchors(ax, run.search_result.path)

    expected = set()
    for edge in run.search_result.path:
        expected.add(edge.label.problem.current_anchor)
        from webswing.planning.astar import GOAL_NODE

        if edge.to_node != GOAL_NODE:
            expected.add(edge.label.problem.target_anchor)

    offsets = ax.collections[0].get_offsets()
    plotted = {tuple(row) for row in offsets}
    assert plotted == {tuple(p) for p in expected}
    plt.close(fig)


def test_plot_selected_anchors_rejects_non_planned_transfer_label(successful_run) -> None:
    import matplotlib.pyplot as plt

    run, _city = successful_run
    fig, ax = plt.subplots()
    bad_path = (replace(run.search_result.path[0], label="not-a-planned-transfer"),)
    with pytest.raises(TypeError):
        plot_selected_anchors(ax, bad_path)
    plt.close(fig)


# --- render_static_overview ------------------------------------------------------------------


def test_render_static_overview_produces_expected_artist_counts(successful_run) -> None:
    run, city = successful_run
    fig = render_static_overview(city, run.search_result.path, run.trajectory, START_ANCHOR_POSITION)
    ax = fig.axes[0]

    assert len(ax.patches) == len(city.buildings) + 1
    assert len(ax.lines) == 1 + len(_contiguous_mode_runs(run.trajectory.modes))  # ground + trajectory runs
    assert len(ax.collections) == 3  # candidate anchors, selected anchors, start

    legend = ax.get_legend()
    assert legend is not None
    labels = {t.get_text() for t in legend.get_texts()}
    assert {"candidate anchor", "swing", "ballistic", "selected anchor", "start"} <= labels

    import matplotlib.pyplot as plt

    plt.close(fig)
