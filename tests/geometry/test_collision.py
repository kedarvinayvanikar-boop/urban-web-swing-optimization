"""Tests for polygon/segment collision-detection primitives.

Covers point-in-polygon (interior, exterior, edge, vertex), segment-segment
intersection (crossing, parallel, touching endpoint, collinear overlap and
non-overlap), segment-to-polygon distance and intersection (including the
documented tangent-contact policy), bounding-box acceleration, ground
collision, destination-region containment, and the combined city-level
segment/trajectory collision checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from webswing.geometry.buildings import Building, City, DestinationRegion
from webswing.geometry.collision import (
    ground_collision,
    point_in_destination,
    point_in_polygon,
    point_to_segment_distance,
    segment_bounding_boxes_overlap,
    segment_collides_with_city,
    segment_intersects_polygon,
    segment_intersects_segment,
    segment_polygon_min_distance,
    segment_to_segment_distance,
    segment_violates_building_clearance,
    trajectory_collides_with_city,
)

SQUARE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])


def make_building(safety_margin: float = 0.0) -> Building:
    return Building(
        building_id="B",
        vertices=SQUARE,
        width=10.0,
        height=10.0,
        roof_elevation=10.0,
        safety_margin=safety_margin,
    )


def make_destination() -> DestinationRegion:
    return DestinationRegion(x_min=50.0, x_max=60.0, y_min=0.0, y_max=5.0)


# --- point_in_polygon ---------------------------------------------------------


def test_point_in_polygon_strictly_inside() -> None:
    assert point_in_polygon((5.0, 5.0), SQUARE) is True


def test_point_in_polygon_strictly_outside() -> None:
    assert point_in_polygon((15.0, 5.0), SQUARE) is False


def test_point_in_polygon_on_edge_midpoint() -> None:
    assert point_in_polygon((5.0, 0.0), SQUARE) is True


def test_point_in_polygon_on_vertex() -> None:
    assert point_in_polygon((0.0, 0.0), SQUARE) is True


def test_point_in_polygon_just_outside_edge() -> None:
    assert point_in_polygon((5.0, -1.0e-3), SQUARE) is False


# --- point_to_segment_distance -------------------------------------------------


def test_point_to_segment_distance_perpendicular_projection() -> None:
    d = point_to_segment_distance((5.0, 3.0), (0.0, 0.0), (10.0, 0.0))
    assert d == pytest.approx(3.0, rel=1e-12)


def test_point_to_segment_distance_beyond_endpoint() -> None:
    d = point_to_segment_distance((15.0, 0.0), (0.0, 0.0), (10.0, 0.0))
    assert d == pytest.approx(5.0, rel=1e-12)


# --- segment_intersects_segment ------------------------------------------------


def test_segments_cross() -> None:
    assert segment_intersects_segment((0.0, 0.0), (10.0, 10.0), (0.0, 10.0), (10.0, 0.0)) is True


def test_segments_parallel_non_intersecting() -> None:
    assert segment_intersects_segment((0.0, 0.0), (10.0, 0.0), (0.0, 1.0), (10.0, 1.0)) is False


def test_segments_touch_at_shared_endpoint() -> None:
    assert segment_intersects_segment((0.0, 0.0), (5.0, 5.0), (5.0, 5.0), (10.0, 0.0)) is True


def test_segments_collinear_overlapping() -> None:
    assert segment_intersects_segment((0.0, 0.0), (5.0, 0.0), (3.0, 0.0), (8.0, 0.0)) is True


def test_segments_collinear_non_overlapping() -> None:
    assert segment_intersects_segment((0.0, 0.0), (2.0, 0.0), (5.0, 0.0), (8.0, 0.0)) is False


# --- segment_to_segment_distance -----------------------------------------------


def test_segment_to_segment_distance_zero_when_intersecting() -> None:
    d = segment_to_segment_distance((0.0, 0.0), (10.0, 10.0), (0.0, 10.0), (10.0, 0.0))
    assert d == 0.0


def test_segment_to_segment_distance_parallel_gap() -> None:
    d = segment_to_segment_distance((0.0, 0.0), (10.0, 0.0), (0.0, 1.0), (10.0, 1.0))
    assert d == pytest.approx(1.0, rel=1e-12)


# --- segment_polygon_min_distance ----------------------------------------------


def test_segment_polygon_min_distance_gap_to_the_left() -> None:
    d = segment_polygon_min_distance((-5.0, 5.0), (-2.0, 5.0), SQUARE)
    assert d == pytest.approx(2.0, rel=1e-12)


def test_segment_polygon_min_distance_zero_when_touching_boundary() -> None:
    d = segment_polygon_min_distance((5.0, 0.0), (5.0, -5.0), SQUARE)
    assert d == pytest.approx(0.0, abs=1e-12)


# --- segment_intersects_polygon (including tangent-contact policy) ------------


def test_segment_crosses_through_polygon_interior() -> None:
    assert segment_intersects_polygon((-5.0, 5.0), (15.0, 5.0), SQUARE) is True


def test_segment_entirely_outside_polygon() -> None:
    assert segment_intersects_polygon((-5.0, 5.0), (-2.0, 5.0), SQUARE) is False


def test_segment_with_endpoint_inside_polygon() -> None:
    assert segment_intersects_polygon((5.0, 5.0), (20.0, 5.0), SQUARE) is True


def test_segment_touching_vertex_corner() -> None:
    # p2 lands exactly on the polygon's (0, 0) vertex.
    assert segment_intersects_polygon((-5.0, -5.0), (0.0, 0.0), SQUARE) is True


def test_segment_tangent_along_edge_line_touches_only_at_shared_vertex() -> None:
    # Collinear with the right edge (x=10, y in [0, 10]) but spans y in
    # [10, 20], so it shares only the (10, 10) vertex with the polygon and
    # never enters the interior. Policy: this tangent contact still counts
    # as an intersection (module docstring, Boundary policy).
    assert segment_intersects_polygon((10.0, 10.0), (10.0, 20.0), SQUARE) is True


# --- segment_bounding_boxes_overlap --------------------------------------------


def test_bounding_boxes_overlap_when_segment_crosses_box() -> None:
    bbox = (0.0, 10.0, 0.0, 10.0)
    assert segment_bounding_boxes_overlap((-5.0, 5.0), (15.0, 5.0), bbox) is True


def test_bounding_boxes_do_not_overlap_when_far_apart() -> None:
    bbox = (0.0, 10.0, 0.0, 10.0)
    assert segment_bounding_boxes_overlap((100.0, 100.0), (110.0, 110.0), bbox) is False


def test_bounding_boxes_overlap_once_margin_closes_the_gap() -> None:
    bbox = (0.0, 10.0, 0.0, 10.0)
    assert segment_bounding_boxes_overlap((11.0, 5.0), (15.0, 5.0), bbox, margin=0.0) is False
    assert segment_bounding_boxes_overlap((11.0, 5.0), (15.0, 5.0), bbox, margin=2.0) is True


# --- segment_violates_building_clearance ---------------------------------------


def test_clearance_violation_for_crossing_segment_regardless_of_margin() -> None:
    building = make_building(safety_margin=0.0)
    assert segment_violates_building_clearance((-5.0, 5.0), (15.0, 5.0), building) is True


def test_clearance_violation_within_safety_margin() -> None:
    building = make_building(safety_margin=1.0)
    # Segment 0.5 m to the left of the square: within the 1.0 m margin.
    assert segment_violates_building_clearance((-0.5, 5.0), (-0.5, 6.0), building) is True


def test_no_clearance_violation_outside_safety_margin() -> None:
    building = make_building(safety_margin=1.0)
    # Segment 1.5 m to the left of the square: outside the 1.0 m margin.
    assert segment_violates_building_clearance((-1.5, 5.0), (-1.5, 6.0), building) is False


def test_extra_clearance_extends_the_violation_zone() -> None:
    building = make_building(safety_margin=1.0)
    # 1.5 m away violates only once extra_clearance brings the total to >1.5 m.
    assert segment_violates_building_clearance((-1.5, 5.0), (-1.5, 6.0), building, extra_clearance=0.0) is False
    assert segment_violates_building_clearance((-1.5, 5.0), (-1.5, 6.0), building, extra_clearance=1.0) is True


# --- ground_collision -----------------------------------------------------------


def test_no_ground_collision_above_ground() -> None:
    assert ground_collision((0.0, 5.0), (1.0, 6.0)) is False


def test_ground_collision_endpoint_on_ground() -> None:
    assert ground_collision((0.0, 0.0), (1.0, 5.0)) is True


def test_ground_collision_segment_crosses_ground() -> None:
    assert ground_collision((0.0, -1.0), (1.0, 1.0)) is True


# --- point_in_destination --------------------------------------------------------


def test_point_in_destination_inside() -> None:
    assert point_in_destination((55.0, 2.0), make_destination()) is True


def test_point_in_destination_outside() -> None:
    assert point_in_destination((70.0, 2.0), make_destination()) is False


def test_point_in_destination_on_boundary() -> None:
    assert point_in_destination((50.0, 0.0), make_destination()) is True


# --- segment_collides_with_city / trajectory_collides_with_city ----------------


def test_segment_collides_with_city_through_building() -> None:
    city = City(buildings=(make_building(),), destination=make_destination())
    assert segment_collides_with_city((-5.0, 5.0), (15.0, 5.0), city) is True


def test_segment_collides_with_city_through_ground_away_from_buildings() -> None:
    city = City(buildings=(make_building(),), destination=make_destination())
    assert segment_collides_with_city((30.0, -1.0), (30.0, 1.0), city) is True


def test_segment_clear_of_city() -> None:
    city = City(buildings=(make_building(),), destination=make_destination())
    assert segment_collides_with_city((30.0, 5.0), (35.0, 6.0), city) is False


def test_trajectory_collides_with_city_when_one_segment_crosses_building() -> None:
    city = City(buildings=(make_building(),), destination=make_destination())
    path = [(-5.0, 5.0), (-2.0, 5.0), (15.0, 5.0), (20.0, 5.0)]
    assert trajectory_collides_with_city(path, city) is True


def test_trajectory_clear_when_all_segments_avoid_city() -> None:
    city = City(buildings=(make_building(),), destination=make_destination())
    path = [(-5.0, 20.0), (5.0, 20.0), (15.0, 20.0)]
    assert trajectory_collides_with_city(path, city) is False


def test_trajectory_with_fewer_than_two_points_never_collides() -> None:
    city = City(buildings=(make_building(),), destination=make_destination())
    assert trajectory_collides_with_city([(0.0, 5.0)], city) is False
    assert trajectory_collides_with_city([], city) is False
