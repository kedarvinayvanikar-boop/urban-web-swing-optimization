"""Tests for dynamic HUD text data and formatting.

Uses the `Agg` backend and a module-scoped fixture (one real planned route,
same small city as `test_astar.py`/`test_static.py`), since HUD field
values are read directly from a real `Trajectory`/`TrajectoryEvaluation`
rather than fabricated. Checks cross-reference `build_hud_frame`'s output
against the underlying arrays at specific indices, verify NaN fields render
as "n/a" during ballistic samples, and check the trivial (no-edge) case.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import math

import numpy as np
import pytest

from webswing.config import BallisticDomain, PhysicalParameters, SwingConstraints
from webswing.geometry.buildings import Building, City, DestinationRegion
from webswing.planning.state import PlanningStateResolution
from webswing.simulation.runner import run_simulation
from webswing.visualization.hud import (
    HudFrame,
    build_hud_frame,
    draw_hud,
    edge_indices_for_path,
    format_hud_text,
)

PARAMS = PhysicalParameters(mass=1.0, gravity=9.80665)
CONSTRAINTS = SwingConstraints(
    tension_max=200.0, load_factor_max=20.0, ell_min=1.0, ell_max=30.0, nu_max=5.0
)
DOMAIN = BallisticDomain(x_min=-200.0, x_max=200.0, y_max=200.0, max_duration=10.0)
RESOLUTION = PlanningStateResolution(
    theta_bin_width=0.1, omega_bin_width=0.2, ell_bin_width=0.5, nu_bin_width=0.1
)
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
        start_anchor_id="START",
        start_anchor_position=(0.0, 50.0),
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


@pytest.fixture(scope="module")
def trivial_run():
    return run_simulation(
        start_anchor_id="START",
        start_anchor_position=(20.0, 27.0),
        start_state=np.array([0.0, 0.0, 1.0, 0.0]),
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


# --- edge_indices_for_path ------------------------------------------------------------


def test_edge_indices_length_matches_trajectory(successful_run) -> None:
    edge_indices = edge_indices_for_path(successful_run.search_result.path)
    assert len(edge_indices) == len(successful_run.trajectory.times)


def test_edge_indices_are_all_zero_for_single_edge_path(successful_run) -> None:
    assert len(successful_run.search_result.path) == 1
    edge_indices = edge_indices_for_path(successful_run.search_result.path)
    assert set(edge_indices) == {0}


def test_edge_indices_trivial_path_is_none(trivial_run) -> None:
    edge_indices = edge_indices_for_path(trivial_run.search_result.path)
    assert edge_indices == (None,)


# --- build_hud_frame -----------------------------------------------------------------


def test_hud_frame_at_first_sample_matches_trajectory_and_evaluation(successful_run) -> None:
    edge_indices = edge_indices_for_path(successful_run.search_result.path)
    frame = build_hud_frame(successful_run.trajectory, successful_run.evaluation, edge_indices, CONSTRAINTS, 0)

    assert isinstance(frame, HudFrame)
    assert frame.simulation_time == successful_run.trajectory.times[0]
    assert frame.mode == successful_run.trajectory.modes[0]
    assert frame.active_anchor_id == successful_run.trajectory.active_anchor_ids[0]
    assert frame.web_length == successful_run.trajectory.web_lengths[0]
    assert frame.tension == pytest.approx(successful_run.evaluation.tensions[0])
    assert frame.tension_max == CONSTRAINTS.tension_max
    assert frame.edge_index == 0
    assert frame.cumulative_travel_time == frame.simulation_time

    vx, vy = successful_run.trajectory.velocities[0]
    assert frame.speed == pytest.approx(math.hypot(vx, vy))


def test_hud_frame_at_ballistic_sample_has_nan_swing_fields(successful_run) -> None:
    edge_indices = edge_indices_for_path(successful_run.search_result.path)
    ballistic_index = successful_run.trajectory.modes.index("ballistic")
    frame = build_hud_frame(
        successful_run.trajectory, successful_run.evaluation, edge_indices, CONSTRAINTS, ballistic_index
    )
    assert frame.mode == "ballistic"
    assert frame.active_anchor_id is None
    assert math.isnan(frame.web_length)
    assert math.isnan(frame.radial_rate)
    assert math.isnan(frame.angular_rate)
    assert math.isnan(frame.tension)
    assert math.isnan(frame.load_factor)


def test_hud_frame_rejects_out_of_range_index(successful_run) -> None:
    edge_indices = edge_indices_for_path(successful_run.search_result.path)
    with pytest.raises(IndexError):
        build_hud_frame(successful_run.trajectory, successful_run.evaluation, edge_indices, CONSTRAINTS, -1)
    with pytest.raises(IndexError):
        build_hud_frame(
            successful_run.trajectory,
            successful_run.evaluation,
            edge_indices,
            CONSTRAINTS,
            len(successful_run.trajectory.times),
        )


def test_hud_frame_trivial_case_has_no_edge(trivial_run) -> None:
    edge_indices = edge_indices_for_path(trivial_run.search_result.path)
    frame = build_hud_frame(trivial_run.trajectory, trivial_run.evaluation, edge_indices, CONSTRAINTS, 0)
    assert frame.edge_index is None


# --- format_hud_text -----------------------------------------------------------------


def test_format_hud_text_swing_sample_shows_numeric_values(successful_run) -> None:
    edge_indices = edge_indices_for_path(successful_run.search_result.path)
    frame = build_hud_frame(successful_run.trajectory, successful_run.evaluation, edge_indices, CONSTRAINTS, 0)
    text = format_hud_text(frame)
    assert "n/a" not in text
    assert "mode: swing" in text
    assert "anchor: START" in text
    assert "planner edge: edge 0" in text


def test_format_hud_text_ballistic_sample_shows_na_for_swing_only_fields(successful_run) -> None:
    edge_indices = edge_indices_for_path(successful_run.search_result.path)
    ballistic_index = successful_run.trajectory.modes.index("ballistic")
    frame = build_hud_frame(
        successful_run.trajectory, successful_run.evaluation, edge_indices, CONSTRAINTS, ballistic_index
    )
    text = format_hud_text(frame)
    assert "mode: ballistic" in text
    assert "anchor: none (in flight)" in text
    assert "web length: n/a" in text
    assert "tension: n/a" in text
    assert "load factor: n/a" in text


# --- draw_hud --------------------------------------------------------------------------


def test_draw_hud_adds_one_text_artist_matching_formatted_text(successful_run) -> None:
    import matplotlib.pyplot as plt

    edge_indices = edge_indices_for_path(successful_run.search_result.path)
    frame = build_hud_frame(successful_run.trajectory, successful_run.evaluation, edge_indices, CONSTRAINTS, 0)

    fig, ax = plt.subplots()
    artist = draw_hud(ax, frame)
    assert len(ax.texts) == 1
    assert artist.get_text() == format_hud_text(frame)
    plt.close(fig)
