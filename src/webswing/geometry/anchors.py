r"""Candidate web-attachment anchor derivation and line-of-sight validation.

Implements the two anchor-related responsibilities of the Urban Environment
that `geometry.buildings` (data model) and `geometry.collision` (segment/
polygon algorithms) do not: deriving candidate anchor points from a
building's roof geometry, and validating that a straight line between two
points (a candidate web segment) is unobstructed by the city's buildings.

Self-occlusion at an anchor's own building
--------------------------------------------
`geometry.collision` treats a polygon's boundary as inside it (see that
module's Boundary policy). A roof-corner anchor sits exactly on its own
building's boundary by construction, so a naive line-of-sight check from
any point to that anchor would always report a collision with the anchor's
own building, regardless of whether anything actually blocks the view.
Every line-of-sight function in this module therefore accepts (or, for
anchors drawn from `City.all_candidate_anchors`, automatically applies) an
`exclude_building_id` that omits the anchor's own building from the
obstruction check. This is a deliberate exclusion of self-occlusion at the
anchor point, not a weakening of collision detection against other
buildings, which are still fully checked.

Line-of-sight here checks buildings only, not the ground; ground contact is
a termination condition handled by `dynamics.events`, not a visibility
concern.
"""

from __future__ import annotations

from webswing.geometry.buildings import Building, City
from webswing.geometry.collision import (
    Point,
    segment_bounding_boxes_overlap,
    segment_violates_building_clearance,
)

_ROOF_ELEVATION_TOLERANCE_M = 1.0e-6


def roof_corners(building: Building, tolerance: float = _ROOF_ELEVATION_TOLERANCE_M) -> tuple[Point, ...]:
    """Return the building's polygon vertices lying at (or within `tolerance` of) its roof.

    This is a simplification of "geometrically valid roof corners": every
    vertex at the roof elevation is returned, in the polygon's original
    vertex order, without filtering for convexity. Buildings with a
    non-planar roofline (e.g. a single apex, or a flat roof with two
    corners) are both handled correctly by this height-based selection;
    concave-roof-corner filtering is not performed and is left as a future
    refinement if it proves necessary.

    Parameters
    ----------
    building : Building
        Building whose roof corners are being derived.
    tolerance : float, optional
        Maximum vertical distance from `building.roof_elevation` for a
        vertex to be considered a roof corner, in meters.

    Returns
    -------
    tuple[tuple[float, float], ...]
        Roof-corner vertex positions, in meters, in polygon order. Never
        empty, since `Building` validation guarantees at least one vertex
        sits at the roof elevation.
    """
    return tuple(
        (float(x), float(y))
        for x, y in building.vertices
        if abs(y - building.roof_elevation) <= tolerance
    )


def line_of_sight_clear(
    p1: Point,
    p2: Point,
    city: City,
    extra_clearance: float = 0.0,
    exclude_building_id: str | None = None,
) -> bool:
    """Return whether the straight segment p1-p2 is unobstructed by any building in `city`.

    Checks every building in `city` except `exclude_building_id` (see
    module docstring on self-occlusion), using bounding-box acceleration
    before the exact segment-polygon clearance test.

    Parameters
    ----------
    p1, p2 : tuple[float, float]
        Segment endpoints, in meters.
    city : City
        Environment to check against.
    extra_clearance : float, optional
        Additional required clearance beyond each building's own
        `safety_margin`, in meters.
    exclude_building_id : str or None, optional
        A building identifier to omit from the check, e.g. the building an
        anchor endpoint rests on.

    Returns
    -------
    bool
        True if the segment does not collide with, or come within
        clearance of, any checked building.
    """
    for building in city.buildings:
        if building.building_id == exclude_building_id:
            continue
        margin = building.safety_margin + extra_clearance
        if not segment_bounding_boxes_overlap(p1, p2, building.bounding_box, margin):
            continue
        if segment_violates_building_clearance(p1, p2, building, extra_clearance):
            return False
    return True


def anchor_has_line_of_sight(
    from_point: Point,
    anchor_building_id: str,
    anchor_point: Point,
    city: City,
    extra_clearance: float = 0.0,
) -> bool:
    """Return whether `from_point` has an unobstructed view of a candidate anchor.

    Convenience wrapper over `line_of_sight_clear` that automatically
    excludes the anchor's own building from the obstruction check (see
    module docstring on self-occlusion); other buildings, including
    buildings the anchor happens to be near but does not belong to, are
    still checked normally.

    Parameters
    ----------
    from_point : tuple[float, float]
        Observer position (e.g. current body position), in meters.
    anchor_building_id : str
        Identifier of the building the candidate anchor belongs to.
    anchor_point : tuple[float, float]
        Candidate anchor position, in meters.
    city : City
        Environment to check against.
    extra_clearance : float, optional
        Additional required clearance beyond each building's own
        `safety_margin`, in meters.

    Returns
    -------
    bool
        True if the anchor is visible (unobstructed by any other building).
    """
    return line_of_sight_clear(
        from_point, anchor_point, city, extra_clearance, exclude_building_id=anchor_building_id
    )


def visible_candidate_anchors(
    from_point: Point, city: City, extra_clearance: float = 0.0
) -> tuple[tuple[str, Point], ...]:
    """Return every (building_id, anchor) candidate in `city` visible from `from_point`.

    Parameters
    ----------
    from_point : tuple[float, float]
        Observer position, in meters.
    city : City
        Environment supplying the candidate anchors (via
        `City.all_candidate_anchors`) and the buildings to check
        obstruction against.
    extra_clearance : float, optional
        Additional required clearance beyond each building's own
        `safety_margin`, in meters.

    Returns
    -------
    tuple[tuple[str, tuple[float, float]], ...]
        The subset of `city.all_candidate_anchors()` with an unobstructed
        line of sight from `from_point`, in the same order.
    """
    return tuple(
        (building_id, anchor)
        for building_id, anchor in city.all_candidate_anchors()
        if anchor_has_line_of_sight(from_point, building_id, anchor, city, extra_clearance)
    )
