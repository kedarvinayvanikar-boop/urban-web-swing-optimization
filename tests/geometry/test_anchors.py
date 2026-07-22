"""Tests for roof-corner anchor derivation and anchor line-of-sight validation.

Covers roof-corner extraction for a flat-topped and a single-apex building,
line-of-sight clearance against an intervening building, the self-occlusion
exclusion at an anchor's own building (which would otherwise always
register as blocked, since collision boundaries are treated as inside),
and the aggregate `visible_candidate_anchors` filter over a small city.
"""

from __future__ import annotations

import numpy as np
import pytest

from webswing.geometry.anchors import (
    anchor_has_line_of_sight,
    line_of_sight_clear,
    roof_corners,
    visible_candidate_anchors,
)
from webswing.geometry.buildings import Building, City, DestinationRegion

FLAT_ROOF = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
APEX_ROOF = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [5.0, 10.0], [0.0, 5.0]])


def make_flat_building(building_id: str = "A", candidate_anchors=(), safety_margin: float = 0.0) -> Building:
    return Building(
        building_id=building_id,
        vertices=FLAT_ROOF,
        width=10.0,
        height=10.0,
        roof_elevation=10.0,
        candidate_anchors=candidate_anchors,
        safety_margin=safety_margin,
    )


def make_apex_building(building_id: str = "APEX") -> Building:
    return Building(
        building_id=building_id,
        vertices=APEX_ROOF,
        width=10.0,
        height=10.0,
        roof_elevation=10.0,
    )


def make_destination() -> DestinationRegion:
    return DestinationRegion(x_min=100.0, x_max=110.0, y_min=0.0, y_max=5.0)


# --- roof_corners ---------------------------------------------------------------


def test_roof_corners_flat_roof_returns_both_top_vertices() -> None:
    corners = roof_corners(make_flat_building())
    assert corners == ((10.0, 10.0), (0.0, 10.0))


def test_roof_corners_single_apex_returns_one_vertex() -> None:
    corners = roof_corners(make_apex_building())
    assert corners == ((5.0, 10.0),)


def test_roof_corners_includes_vertex_within_tolerance() -> None:
    vertices = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0 - 1e-9], [0.0, 10.0]])
    building = Building(
        building_id="B", vertices=vertices, width=10.0, height=10.0, roof_elevation=10.0
    )
    corners = roof_corners(building)
    assert (10.0, 10.0 - 1e-9) in corners
    assert (0.0, 10.0) in corners


# --- line_of_sight_clear --------------------------------------------------------


def test_line_of_sight_clear_when_no_building_intervenes() -> None:
    city = City(buildings=(make_flat_building(),), destination=make_destination())
    assert line_of_sight_clear((30.0, 5.0), (40.0, 5.0), city) is True


def test_line_of_sight_blocked_by_intervening_building() -> None:
    city = City(buildings=(make_flat_building(),), destination=make_destination())
    assert line_of_sight_clear((-5.0, 5.0), (15.0, 5.0), city) is False


def test_line_of_sight_excludes_named_building() -> None:
    city = City(buildings=(make_flat_building(),), destination=make_destination())
    # Same segment as the blocked case above, but with building "A" excluded.
    assert line_of_sight_clear((-5.0, 5.0), (15.0, 5.0), city, exclude_building_id="A") is True


def test_line_of_sight_still_blocked_by_other_building_when_excluding_a_different_one() -> None:
    a = make_flat_building(building_id="A")
    b = Building(
        building_id="B",
        vertices=FLAT_ROOF + np.array([20.0, 0.0]),
        width=10.0,
        height=10.0,
        roof_elevation=10.0,
    )
    city = City(buildings=(a, b), destination=make_destination())
    # Segment crosses building B; excluding A must not clear it.
    assert line_of_sight_clear((15.0, 5.0), (35.0, 5.0), city, exclude_building_id="A") is False


# --- anchor_has_line_of_sight (self-occlusion exclusion) ------------------------


def test_anchor_on_own_roof_is_not_self_occluded() -> None:
    building = make_flat_building(candidate_anchors=((0.0, 10.0),))
    city = City(buildings=(building,), destination=make_destination())
    # Observer directly "inside" the building's horizontal span, looking up
    # at its own roof corner: without self-exclusion this would always
    # register as blocked by its own building.
    assert anchor_has_line_of_sight((5.0, 5.0), "A", (0.0, 10.0), city) is True


def test_anchor_visibility_blocked_by_a_different_building() -> None:
    a = make_flat_building(building_id="A", candidate_anchors=((0.0, 10.0),))
    blocker = Building(
        building_id="BLOCKER",
        vertices=FLAT_ROOF + np.array([-20.0, 0.0]),
        width=10.0,
        height=10.0,
        roof_elevation=10.0,
    )
    city = City(buildings=(a, blocker), destination=make_destination())
    # Observer on the far side of BLOCKER from A's anchor.
    assert anchor_has_line_of_sight((-25.0, 5.0), "A", (0.0, 10.0), city) is False


# --- visible_candidate_anchors ---------------------------------------------------


def test_visible_candidate_anchors_filters_blocked_and_keeps_clear() -> None:
    # A sits at x in [0, 10] with both roof corners as candidate anchors;
    # BLOCKER sits at x in [20, 30] with its own roof-corner anchor. From an
    # observer beyond BLOCKER, both of A's anchors are shadowed by BLOCKER,
    # while BLOCKER's own anchor is visible (self-occlusion excluded).
    a = make_flat_building(building_id="A", candidate_anchors=((0.0, 10.0), (10.0, 10.0)))
    blocker = Building(
        building_id="BLOCKER",
        vertices=FLAT_ROOF + np.array([20.0, 0.0]),
        width=10.0,
        height=10.0,
        roof_elevation=10.0,
        candidate_anchors=((25.0, 10.0),),
    )
    city = City(buildings=(a, blocker), destination=make_destination())

    observer = (40.0, 5.0)
    visible = visible_candidate_anchors(observer, city)

    assert visible == (("BLOCKER", (25.0, 10.0)),)
