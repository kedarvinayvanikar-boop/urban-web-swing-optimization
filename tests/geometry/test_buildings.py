"""Tests for the building/city environment data model.

Validates successful construction and cross-field consistency of `Building`
against its own polygon (width, height, roof elevation, ground contact),
polygon boundary cases (too few vertices, degenerate/zero-area polygon,
non-finite vertices), `DestinationRegion` bound validation, and `City`
aggregation (duplicate-id rejection, lookup, anchor collection).
"""

from __future__ import annotations

import numpy as np
import pytest

from webswing.exceptions import InvalidGeometryError
from webswing.geometry.buildings import Building, City, DestinationRegion

RECTANGLE = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 20.0], [0.0, 20.0]])


def make_building(**overrides) -> Building:
    defaults = dict(
        building_id="B1",
        vertices=RECTANGLE,
        width=10.0,
        height=20.0,
        roof_elevation=20.0,
    )
    defaults.update(overrides)
    return Building(**defaults)


def test_valid_rectangle_building_bounding_box_and_area() -> None:
    b = make_building()
    assert b.bounding_box == pytest.approx((0.0, 10.0, 0.0, 20.0))
    assert b.area == pytest.approx(200.0, rel=1e-12)


def test_building_rejects_empty_id() -> None:
    with pytest.raises(InvalidGeometryError):
        make_building(building_id="")


def test_building_rejects_too_few_vertices() -> None:
    with pytest.raises(InvalidGeometryError):
        make_building(vertices=np.array([[0.0, 0.0], [1.0, 0.0]]), width=1.0, height=1.0, roof_elevation=1.0)


def test_building_rejects_nonfinite_vertices() -> None:
    bad = RECTANGLE.copy()
    bad[0, 0] = float("nan")
    with pytest.raises(InvalidGeometryError):
        make_building(vertices=bad)


def test_building_rejects_degenerate_polygon() -> None:
    collinear = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    with pytest.raises(InvalidGeometryError):
        make_building(vertices=collinear, width=10.0, height=0.0, roof_elevation=0.0)


def test_building_rejects_width_inconsistent_with_polygon() -> None:
    with pytest.raises(InvalidGeometryError):
        make_building(width=999.0)


def test_building_rejects_height_inconsistent_with_polygon() -> None:
    with pytest.raises(InvalidGeometryError):
        make_building(height=999.0)


def test_building_rejects_roof_elevation_inconsistent_with_polygon() -> None:
    with pytest.raises(InvalidGeometryError):
        make_building(roof_elevation=999.0)


def test_building_rejects_base_not_on_ground() -> None:
    floating = RECTANGLE + np.array([0.0, 5.0])
    with pytest.raises(InvalidGeometryError):
        make_building(vertices=floating, width=10.0, height=20.0, roof_elevation=25.0)


def test_building_rejects_nonpositive_width() -> None:
    with pytest.raises(InvalidGeometryError):
        make_building(width=0.0)


def test_building_rejects_nonpositive_height() -> None:
    with pytest.raises(InvalidGeometryError):
        make_building(height=-1.0)


def test_building_rejects_negative_safety_margin() -> None:
    with pytest.raises(InvalidGeometryError):
        make_building(safety_margin=-0.1)


def test_building_accepts_and_stores_candidate_anchors() -> None:
    b = make_building(candidate_anchors=((0.0, 20.0), (10.0, 20.0)))
    assert b.candidate_anchors == ((0.0, 20.0), (10.0, 20.0))


def test_building_rejects_nonfinite_candidate_anchor() -> None:
    with pytest.raises(InvalidGeometryError):
        make_building(candidate_anchors=((0.0, float("nan")),))


def test_building_vertices_array_is_read_only() -> None:
    b = make_building()
    with pytest.raises(ValueError):
        b.vertices[0, 0] = 5.0


def make_destination(**overrides) -> DestinationRegion:
    defaults = dict(x_min=50.0, x_max=60.0, y_min=0.0, y_max=5.0)
    defaults.update(overrides)
    return DestinationRegion(**defaults)


def test_destination_region_accepts_valid_bounds() -> None:
    d = make_destination()
    assert d.x_max > d.x_min


def test_destination_region_rejects_x_max_not_greater_than_x_min() -> None:
    with pytest.raises(InvalidGeometryError):
        make_destination(x_min=10.0, x_max=10.0)


def test_destination_region_rejects_y_max_not_greater_than_y_min() -> None:
    with pytest.raises(InvalidGeometryError):
        make_destination(y_min=5.0, y_max=5.0)


def test_destination_region_rejects_nonfinite_bound() -> None:
    with pytest.raises(InvalidGeometryError):
        make_destination(x_min=float("nan"))


def test_destination_region_rejects_below_ground() -> None:
    with pytest.raises(InvalidGeometryError):
        make_destination(y_min=-1.0)


def test_city_aggregates_buildings_and_destination() -> None:
    b1 = make_building(building_id="B1", candidate_anchors=((0.0, 20.0),))
    b2 = make_building(building_id="B2", candidate_anchors=((10.0, 20.0), (0.0, 20.0)))
    city = City(buildings=(b1, b2), destination=make_destination())
    assert city.get_building("B1") is b1
    assert city.get_building("B2") is b2
    assert city.all_candidate_anchors() == (
        ("B1", (0.0, 20.0)),
        ("B2", (10.0, 20.0)),
        ("B2", (0.0, 20.0)),
    )


def test_city_rejects_duplicate_building_ids() -> None:
    b1 = make_building(building_id="B1")
    b2 = make_building(building_id="B1")
    with pytest.raises(InvalidGeometryError):
        City(buildings=(b1, b2), destination=make_destination())


def test_city_get_building_raises_for_missing_id() -> None:
    city = City(buildings=(make_building(building_id="B1"),), destination=make_destination())
    with pytest.raises(KeyError):
        city.get_building("does-not-exist")


def test_city_allows_empty_buildings() -> None:
    city = City(buildings=(), destination=make_destination())
    assert city.buildings == ()
    assert city.all_candidate_anchors() == ()
