r"""Polygon and segment intersection algorithms for collision detection.

Implements the geometric primitives needed to check a straight-line motion
segment (a single integration step of either the attached or ballistic
dynamics) against the deterministic urban environment defined in
`geometry.buildings`: point-in-polygon, segment-segment intersection,
segment-polygon intersection, segment-to-polygon minimum distance (for
obstacle-clearance margins), ground collision, destination-region
containment, and axis-aligned bounding-box acceleration.

Boundary policy
----------------
This module treats polygon and region boundaries as closed (inclusive): a
point exactly on a building's edge counts as inside it, and a segment that
merely touches a building's boundary tangentially -- without crossing into
the interior -- is still reported as intersecting it. This is the
conservative choice for a collision system (a trajectory that grazes a
wall is not treated as clear of it) and is applied uniformly by
`point_in_polygon`, `segment_intersects_polygon`, and `ground_collision`.

Continuous vs. sampled collision checking
------------------------------------------
`segment_collides_with_city` and `trajectory_collides_with_city` test the
exact line segment between two given states (not a set of interior sample
points), so they do not suffer tunnelling between the two endpoints
themselves. Tunnelling can still occur if the *endpoints supplied* are
spaced farther apart than the integrator's actual path curvature justifies;
callers assembling a trajectory from `solve_ivp` output are responsible for
choosing a sampling/event-refinement interval fine enough that consecutive
points bound the true path tightly, per CLAUDE.md's requirement to document
this possibility when segment sampling is used as a proxy for the
continuous path.
"""

from __future__ import annotations

import math

import numpy as np

from webswing.geometry.buildings import Building, City, DestinationRegion

Point = tuple[float, float]
BoundingBox = tuple[float, float, float, float]

_COINCIDENCE_TOLERANCE_M = 1.0e-9


def _polygon_edges(vertices: np.ndarray) -> list[tuple[Point, Point]]:
    n = len(vertices)
    return [(tuple(vertices[i]), tuple(vertices[(i + 1) % n])) for i in range(n)]


def point_to_segment_distance(point: Point, seg_start: Point, seg_end: Point) -> float:
    """Return the Euclidean distance from `point` to the closest point on a segment.

    Parameters
    ----------
    point : tuple[float, float]
        Query point, in meters.
    seg_start, seg_end : tuple[float, float]
        Segment endpoints, in meters.

    Returns
    -------
    float
        Minimum distance, in meters. Zero if `point` lies on the segment.
    """
    p = np.asarray(point, dtype=float)
    a = np.asarray(seg_start, dtype=float)
    b = np.asarray(seg_end, dtype=float)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= _COINCIDENCE_TOLERANCE_M**2:
        return float(np.linalg.norm(p - a))
    t = float(np.dot(p - a, ab) / denom)
    t = min(1.0, max(0.0, t))
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))


def point_in_polygon(point: Point, vertices: np.ndarray) -> bool:
    """Return whether `point` lies inside or on the boundary of a polygon.

    Uses the standard even-odd (crossing-number) ray-casting test for the
    strict-interior case, preceded by an explicit boundary check so that
    points on an edge are reliably classified as inside (see module
    docstring, Boundary policy).

    Parameters
    ----------
    point : tuple[float, float]
        Query point, in meters.
    vertices : np.ndarray, shape (N, 2)
        Polygon boundary vertices, in meters.

    Returns
    -------
    bool
        True if `point` is inside the polygon or on its boundary.
    """
    for a, b in _polygon_edges(vertices):
        if point_to_segment_distance(point, a, b) <= _COINCIDENCE_TOLERANCE_M:
            return True

    x, y = point
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = vertices[i]
        xj, yj = vertices[j]
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _orientation(a: Point, b: Point, c: Point) -> float:
    """Signed area of triangle (a, b, c); sign gives turn direction, 0 collinear."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    """Whether `b` lies within the bounding box of segment a-c, given a, b, c collinear."""
    return (
        min(a[0], c[0]) - _COINCIDENCE_TOLERANCE_M <= b[0] <= max(a[0], c[0]) + _COINCIDENCE_TOLERANCE_M
        and min(a[1], c[1]) - _COINCIDENCE_TOLERANCE_M <= b[1] <= max(a[1], c[1]) + _COINCIDENCE_TOLERANCE_M
    )


def segment_intersects_segment(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """Return whether segments p1-p2 and q1-q2 intersect, including touching endpoints.

    Uses the standard orientation-based test with an explicit collinear/
    on-segment fallback, so a segment endpoint that lands exactly on the
    other segment (a "touching corner") is correctly reported as an
    intersection rather than missed by floating-point orientation noise.

    Parameters
    ----------
    p1, p2, q1, q2 : tuple[float, float]
        Segment endpoints, in meters.

    Returns
    -------
    bool
        True if the two segments share at least one point.
    """
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    def sign(v: float) -> int:
        if abs(v) <= _COINCIDENCE_TOLERANCE_M:
            return 0
        return 1 if v > 0 else -1

    s1, s2, s3, s4 = sign(o1), sign(o2), sign(o3), sign(o4)

    if s1 != s2 and s3 != s4:
        return True
    if s1 == 0 and _on_segment(p1, q1, p2):
        return True
    if s2 == 0 and _on_segment(p1, q2, p2):
        return True
    if s3 == 0 and _on_segment(q1, p1, q2):
        return True
    if s4 == 0 and _on_segment(q1, p2, q2):
        return True
    return False


def segment_to_segment_distance(p1: Point, p2: Point, q1: Point, q2: Point) -> float:
    """Return the minimum Euclidean distance between two segments.

    Zero if the segments intersect; otherwise the minimum of the four
    endpoint-to-opposite-segment distances, which is exact for two straight
    segments in the plane.

    Parameters
    ----------
    p1, p2, q1, q2 : tuple[float, float]
        Segment endpoints, in meters.

    Returns
    -------
    float
        Minimum distance, in meters.
    """
    if segment_intersects_segment(p1, p2, q1, q2):
        return 0.0
    return min(
        point_to_segment_distance(p1, q1, q2),
        point_to_segment_distance(p2, q1, q2),
        point_to_segment_distance(q1, p1, p2),
        point_to_segment_distance(q2, p1, p2),
    )


def segment_polygon_min_distance(p1: Point, p2: Point, vertices: np.ndarray) -> float:
    """Return the minimum distance from a segment to a polygon's boundary.

    Parameters
    ----------
    p1, p2 : tuple[float, float]
        Segment endpoints, in meters.
    vertices : np.ndarray, shape (N, 2)
        Polygon boundary vertices, in meters.

    Returns
    -------
    float
        Minimum distance over all polygon edges, in meters. Zero if the
        segment intersects or lies on the boundary.
    """
    return min(segment_to_segment_distance(p1, p2, a, b) for a, b in _polygon_edges(vertices))


def segment_intersects_polygon(p1: Point, p2: Point, vertices: np.ndarray) -> bool:
    """Return whether a segment intersects a polygon, per the module's boundary policy.

    True if either endpoint lies inside or on the polygon, or if the
    segment crosses or touches any polygon edge (see module docstring,
    Boundary policy: tangent contact counts as intersection).

    Parameters
    ----------
    p1, p2 : tuple[float, float]
        Segment endpoints, in meters.
    vertices : np.ndarray, shape (N, 2)
        Polygon boundary vertices, in meters.

    Returns
    -------
    bool
        True if the segment intersects the polygon.
    """
    if point_in_polygon(p1, vertices) or point_in_polygon(p2, vertices):
        return True
    return any(segment_intersects_segment(p1, p2, a, b) for a, b in _polygon_edges(vertices))


def segment_bounding_boxes_overlap(
    p1: Point, p2: Point, bbox: BoundingBox, margin: float = 0.0
) -> bool:
    """Return whether a segment's bounding box overlaps another, expanded by `margin`.

    A cheap necessary (not sufficient) pre-filter: if the boxes do not
    overlap, the segment cannot intersect or come within `margin` of
    whatever `bbox` bounds, so the expensive exact test can be skipped.

    Parameters
    ----------
    p1, p2 : tuple[float, float]
        Segment endpoints, in meters.
    bbox : tuple[float, float, float, float]
        (min_x, max_x, min_y, max_y) of the other geometry, in meters.
    margin : float, optional
        Non-negative clearance margin to expand both boxes by, in meters.

    Returns
    -------
    bool
        True if the (expanded) bounding boxes overlap.
    """
    seg_min_x, seg_max_x = sorted((p1[0], p2[0]))
    seg_min_y, seg_max_y = sorted((p1[1], p2[1]))
    b_min_x, b_max_x, b_min_y, b_max_y = bbox
    return not (
        seg_max_x + margin < b_min_x
        or seg_min_x - margin > b_max_x
        or seg_max_y + margin < b_min_y
        or seg_min_y - margin > b_max_y
    )


def segment_violates_building_clearance(
    p1: Point, p2: Point, building: Building, extra_clearance: float = 0.0
) -> bool:
    """Return whether a segment collides with or comes too close to a building.

    Combines exact geometric intersection (`segment_intersects_polygon`)
    with the building's configured `safety_margin` and an additional
    caller-supplied `extra_clearance`.

    Parameters
    ----------
    p1, p2 : tuple[float, float]
        Segment endpoints, in meters.
    building : Building
        Obstacle to check against.
    extra_clearance : float, optional
        Additional required clearance beyond `building.safety_margin`, in
        meters. Must be non-negative.

    Returns
    -------
    bool
        True if the segment intersects the building or falls within the
        combined clearance margin of it.
    """
    clearance = building.safety_margin + extra_clearance
    if segment_intersects_polygon(p1, p2, building.vertices):
        return True
    if clearance <= 0.0:
        return False
    return segment_polygon_min_distance(p1, p2, building.vertices) < clearance


def ground_collision(p1: Point, p2: Point) -> bool:
    """Return whether a segment touches or crosses the ground plane y = 0.

    Parameters
    ----------
    p1, p2 : tuple[float, float]
        Segment endpoints, in meters.

    Returns
    -------
    bool
        True if either endpoint has y <= 0, or the segment spans y = 0.
    """
    return min(p1[1], p2[1]) <= 0.0


def point_in_destination(point: Point, region: DestinationRegion) -> bool:
    """Return whether a point lies inside or on the boundary of a destination region.

    Parameters
    ----------
    point : tuple[float, float]
        Query point, in meters.
    region : DestinationRegion
        Axis-aligned destination region.

    Returns
    -------
    bool
        True if `region.x_min <= x <= region.x_max` and
        `region.y_min <= y <= region.y_max`.
    """
    x, y = point
    return region.x_min <= x <= region.x_max and region.y_min <= y <= region.y_max


def segment_collides_with_city(
    p1: Point, p2: Point, city: City, extra_clearance: float = 0.0
) -> bool:
    """Return whether a motion segment collides with the ground or any building in `city`.

    Bounding-box acceleration is applied per building before the exact
    segment-polygon test.

    Parameters
    ----------
    p1, p2 : tuple[float, float]
        Segment endpoints, in meters.
    city : City
        Environment to check against.
    extra_clearance : float, optional
        Additional required clearance beyond each building's own
        `safety_margin`, in meters. Must be non-negative.

    Returns
    -------
    bool
        True if the segment touches/crosses the ground, or collides with
        or violates the clearance of any building in `city`.
    """
    if ground_collision(p1, p2):
        return True
    for building in city.buildings:
        margin = building.safety_margin + extra_clearance
        if not segment_bounding_boxes_overlap(p1, p2, building.bounding_box, margin):
            continue
        if segment_violates_building_clearance(p1, p2, building, extra_clearance):
            return True
    return False


def trajectory_collides_with_city(
    points: list[Point], city: City, extra_clearance: float = 0.0
) -> bool:
    """Return whether any consecutive-point segment of a trajectory collides with `city`.

    Each consecutive pair of `points` is checked with the exact segment
    test in `segment_collides_with_city`, not by sampling interior points;
    see the module docstring for the tunnelling caveat this still leaves
    between the supplied points themselves.

    Parameters
    ----------
    points : list[tuple[float, float]]
        Ordered trajectory positions, in meters. Must contain at least two
        points to define a segment; a shorter list yields False.
    city : City
        Environment to check against.
    extra_clearance : float, optional
        Additional required clearance beyond each building's own
        `safety_margin`, in meters.

    Returns
    -------
    bool
        True if any consecutive segment collides with `city`.
    """
    return any(
        segment_collides_with_city(points[i], points[i + 1], city, extra_clearance)
        for i in range(len(points) - 1)
    )
